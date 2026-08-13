# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dual priority-stream execution: two engines, one process, one copy of the weights.

Enabled with VLLM_DUAL_STREAM=1. Off by default and inert when off.

Each TP worker holds ONE set of model weights and TWO model-runner states:

    LATENCY     small batch, high-priority CUDA stream, small KV reservation
    THROUGHPUT  the normal engine, default-priority stream, the rest of the KV pool

Sharing weights is a requirement, not an optimisation: for a large MoE the
weights dominate device memory (193.75 GiB per GPU of a 267.69 GiB card on the
model this was built against), so two independent processes cannot both hold a
copy. One copy plus two runner states can.

Model-runner state must be per-role because input batches, block tables, and persistent
buffers are mutable. The runners share one TP communicator, with collective kernels
serialized by CUDA events while compute overlaps on the role streams.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

LATENCY = "latency"
THROUGHPUT = "throughput"

_thread_state = threading.local()
_streams: dict = {}
_tp_groups: dict = {}
_collective_sequencer = None


def enabled() -> bool:
    return os.environ.get("VLLM_DUAL_STREAM", "0") == "1"


def latency_kv_fraction() -> float:
    """Share of KV blocks reserved for the latency engine.

    Small on purpose: the latency engine runs a handful of sequences, so giving
    it more only steals capacity from the throughput engine, which is the side
    that actually converts KV into tokens.
    """
    return float(os.environ.get("VLLM_DUAL_STREAM_LATENCY_KV_FRAC", "0.10"))


def latency_max_num_seqs() -> int:
    return int(os.environ.get("VLLM_DUAL_STREAM_LATENCY_MAX_NUM_SEQS", "1"))


def sequential_warmup_steps() -> int:
    return int(os.environ.get("VLLM_DUAL_STREAM_WARMUP_STEPS", "0"))


def validate_config(vllm_config) -> None:
    errors = []
    if vllm_config.scheduler_config.async_scheduling:
        errors.append("async scheduling")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        errors.append("pipeline parallelism")
    if vllm_config.parallel_config.use_ubatching:
        errors.append("ubatching/DBO")
    if vllm_config.parallel_config.enable_expert_parallel:
        errors.append("expert parallelism")
    if vllm_config.parallel_config.data_parallel_size != 1:
        errors.append("data parallelism")
    if vllm_config.parallel_config.prefill_context_parallel_size != 1:
        errors.append("prefill context parallelism")
    if vllm_config.parallel_config.decode_context_parallel_size != 1:
        errors.append("decode context parallelism")
    if vllm_config.kv_transfer_config is not None:
        errors.append("KV transfer connectors")
    if vllm_config.ec_transfer_config is not None:
        errors.append("encoder-cache transfer connectors")
    if vllm_config.compilation_config.cudagraph_mode.name != "NONE":
        errors.append("CUDA graphs")
    if errors:
        unsupported = ", ".join(errors)
        raise ValueError(
            "VLLM_DUAL_STREAM does not yet support "
            f"{unsupported}. Use TP-only execution with --enforce-eager and "
            "--no-async-scheduling."
        )


class _CollectiveSequencer:
    """Order collectives across role streams and serialize their GPU execution."""

    def __init__(self):
        self._condition = threading.Condition()
        self._index = 0
        self._role = LATENCY
        self._aborted = False
        self._previous_event = None
        self._events = []
        self._done_at = {}

    def _skip_finished_role(self):
        for _ in range(2):
            if self._done_at.get(self._role, self._index + 1) > self._index:
                return
            if self._role == LATENCY:
                self._role = THROUGHPUT
            else:
                self._index += 1
                self._role = LATENCY

    def run(self, role, operation):
        index = getattr(_thread_state, "collective_index", 0)
        _thread_state.collective_index = index + 1
        with self._condition:

            def ready():
                self._skip_finished_role()
                return self._aborted or (self._index == index and self._role == role)

            self._condition.wait_for(ready)
            if self._aborted:
                raise RuntimeError("dual-stream collective sequence aborted")
            try:
                import torch

                current_stream = torch.cuda.current_stream()
                if self._previous_event is not None:
                    current_stream.wait_event(self._previous_event)
                result = operation()
                event = torch.cuda.Event()
                event.record(current_stream)
                self._previous_event = event
                self._events.append(event)
            except BaseException:
                self._aborted = True
                self._condition.notify_all()
                raise
            if role == LATENCY:
                self._role = THROUGHPUT
            else:
                self._index += 1
                self._role = LATENCY
            self._skip_finished_role()
            self._condition.notify_all()
            return result

    def mark_done(self, role, count):
        with self._condition:
            self._done_at[role] = count
            self._skip_finished_role()
            self._condition.notify_all()

    def abort(self):
        with self._condition:
            self._aborted = True
            self._condition.notify_all()


@contextlib.contextmanager
def sequenced_collectives():
    global _collective_sequencer
    assert _collective_sequencer is None
    sequencer = _CollectiveSequencer()
    _collective_sequencer = sequencer
    try:
        yield
    finally:
        sequencer.abort()
        _collective_sequencer = None


def abort_collectives():
    if _collective_sequencer is not None:
        _collective_sequencer.abort()


def finish_collectives(role):
    if _collective_sequencer is not None:
        _collective_sequencer.mark_done(
            role, getattr(_thread_state, "collective_index", 0)
        )


class _RoleTPGroup:
    """Route one role's collectives through the shared TP communicator."""

    _SEQUENCED_METHODS = {
        "all_gather",
        "all_gatherv",
        "broadcast",
        "gather",
        "reduce_scatter",
        "reduce_scatterv",
    }

    def __init__(self, role, primary):
        self._role = role
        self._primary = primary
        self.device_group = primary.device_group

    def __getattr__(self, name):
        target = getattr(self._primary, name)
        if name not in self._SEQUENCED_METHODS or not callable(target):
            return target

        def wrapped(*args, **kwargs):
            def operation():
                return target(*args, **kwargs)

            if _collective_sequencer is None:
                return operation()
            return _collective_sequencer.run(self._role, operation)

        return wrapped

    def all_reduce(self, input_):
        import torch.distributed as dist

        if input_.numel() == 0:
            return input_

        def operation():
            output = input_.clone()
            dist.all_reduce(output, group=self.device_group)
            return output

        if _collective_sequencer is None:
            return operation()
        return _collective_sequencer.run(self._role, operation)


def init_worker(model_runner_factory, primary_runner):
    """Build the latency-role runner, communicator and streams inside a TP worker.

    `model_runner_factory` returns a fresh, unloaded runner; its `.model` is
    then pointed at the primary runner's weights so nothing is loaded twice.
    """
    import torch
    import torch.distributed as dist

    from vllm.distributed import parallel_state as ps

    if not enabled() or LATENCY in _tp_groups:
        return _tp_groups.get(LATENCY), None

    primary = ps.get_tp_group()
    _tp_groups[THROUGHPUT] = _RoleTPGroup(THROUGHPUT, primary)
    ranks = dist.get_process_group_ranks(primary.device_group)
    _tp_groups[LATENCY] = _RoleTPGroup(LATENCY, primary)

    least, greatest = torch.cuda.Stream.priority_range()
    _streams[LATENCY] = torch.cuda.Stream(priority=greatest)
    _streams[THROUGHPUT] = torch.cuda.Stream(priority=least)

    latency_runner = model_runner_factory()

    # Assigning `.model` is not enough: load_model() also derives per-runner
    # state (e.g. decode_query_len, speculator wiring, memory accounting). Run
    # it with a loader that returns the already-resident module: all side
    # effects, no duplicate weight allocation or disk reads.
    import sys as _sys

    mod: Any = _sys.modules[type(latency_runner).__module__]
    original_get_loader = mod.get_model_loader
    original_loader = original_get_loader(latency_runner.vllm_config.load_config)

    class _SharedWeightLoader:
        def __init__(self, model):
            self._model = model

        def load_model(self, *args, **kwargs):
            return self._model

        def __getattr__(self, name):
            return getattr(original_loader, name)

    mod.get_model_loader = lambda *a, **k: _SharedWeightLoader(primary_runner.model)
    try:
        latency_runner.load_model()
    finally:
        mod.get_model_loader = original_get_loader
    logger.info(
        "dual-stream: latency runner ready (shared TP comm over ranks %s, "
        "stream priorities lat=%d thr=%d)",
        ranks,
        greatest,
        least,
    )
    return _tp_groups[LATENCY], latency_runner


def split_kv_cache_config(kv_cache_config, fraction: float):
    """(throughput_config, latency_config) carved out of one pool.

    num_blocks alone is not enough. KVCacheTensor.size is a byte count computed
    from the original block count, and initialize_kv_cache allocates from that.
    A shallow copy changing only num_blocks makes each runner allocate the full
    pool and OOM. Rescale tensor byte sizes to the role's block count as well.
    """
    import copy

    total = kv_cache_config.num_blocks
    lat_blocks = max(1, int(total * fraction))
    thr_blocks = total - lat_blocks

    def carve(n_blocks):
        cfg = copy.deepcopy(kv_cache_config)
        cfg.num_blocks = n_blocks
        for t in cfg.kv_cache_tensors:
            if getattr(t, "block_stride", 0):
                # Packed layout: bytes are exactly stride * blocks.
                t.size = t.block_stride * n_blocks
            else:
                # Scale proportionally, preserving per-block granularity.
                per_block = t.size // max(1, total)
                t.size = per_block * n_blocks
        return cfg

    thr, lat = carve(thr_blocks), carve(lat_blocks)
    thr_bytes = sum(t.size for t in thr.kv_cache_tensors)
    lat_bytes = sum(t.size for t in lat.kv_cache_tensors)
    orig_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
    logger.info(
        "dual-stream: KV split -> throughput %d blocks (%.1f GiB), latency %d blocks "
        "(%.1f GiB); original %.1f GiB",
        thr_blocks,
        thr_bytes / 2**30,
        lat_blocks,
        lat_bytes / 2**30,
        orig_bytes / 2**30,
    )
    return thr, lat


@contextlib.contextmanager
def on_stream(role: str):
    """Execute a forward as `role`: its CUDA stream AND its TP communicator.

    Both together. Running on the latency stream while collectives go to the throughput
    communicator is exactly the race this module exists to prevent.
    """
    import torch

    from vllm.distributed import parallel_state as ps

    if not enabled() or role not in _streams:
        yield None
        return
    prev = getattr(_thread_state, "role", None)
    prev_index = getattr(_thread_state, "collective_index", None)
    _thread_state.role = role
    _thread_state.collective_index = 0
    try:
        with (
            ps.override_tp_group(_tp_groups[role]),
            torch.cuda.stream(_streams[role]),
        ):
            yield _streams[role]
    finally:
        _thread_state.role = prev
        _thread_state.collective_index = prev_index


def stream(role: str):
    return _streams.get(role)


def current_role() -> str | None:
    """Return the dual-stream role bound to the current host thread."""
    return getattr(_thread_state, "role", None)
