"""Dual priority-stream execution: two engines, one process, one copy of the weights.

Enabled with VLLM_DUAL_STREAM=1. Off by default and inert when off.

Each TP worker holds ONE set of model weights and TWO model-runner states:

    LATENCY     small batch, high-priority CUDA stream, small KV reservation
    THROUGHPUT  the normal engine, default-priority stream, the rest of the KV pool

Sharing weights is a requirement, not an optimisation: for a large MoE the weights dominate
device memory (193.75 GiB per GPU of a 267.69 GiB card on the model this was built against),
so two independent processes cannot both hold a copy. One copy plus two runner states can.

Two things must be per-role or the two forwards corrupt each other:
  * the TP communicator - vLLM resolves it through a module-level singleton, which is fine
    for one in-flight forward and undefined for two;
  * the model-runner state - input batches, block tables and persistent buffers are mutable
    per-instance, so the runners must be distinct objects even though `.model` is shared.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading

logger = logging.getLogger(__name__)

LATENCY = "latency"
THROUGHPUT = "throughput"

_thread_state = threading.local()
_streams: dict = {}
_tp_groups: dict = {}


def enabled() -> bool:
    return os.environ.get("VLLM_DUAL_STREAM", "0") == "1"


def latency_kv_fraction() -> float:
    """Share of KV blocks reserved for the latency engine.

    Small on purpose: the latency engine runs a handful of sequences, so giving it more only
    steals capacity from the throughput engine, which is the side that actually converts KV
    into tokens.
    """
    return float(os.environ.get("VLLM_DUAL_STREAM_LATENCY_KV_FRAC", "0.10"))


def _patch_tp_group_resolution() -> None:
    from vllm.distributed import parallel_state as ps

    if getattr(ps, "_dual_stream_patched", False):
        return
    original = ps.get_tp_group

    def get_tp_group_threadaware():
        role = getattr(_thread_state, "role", None)
        if role is not None:
            g = _tp_groups.get(role)
            if g is not None:
                return g
        return original()

    ps.get_tp_group = get_tp_group_threadaware
    ps._dual_stream_original_get_tp_group = original
    ps._dual_stream_patched = True


class _SecondaryTPGroup:
    """Primary GroupCoordinator, but collectives routed to a second NCCL communicator.

    Deliberately bypasses vLLM's custom all-reduce: that path uses one preallocated
    signal/staging buffer per process, so two concurrent forwards would corrupt each other's.
    Plain NCCL is correct but slower; per-role custom-allreduce buffers would win it back.
    """

    def __init__(self, primary, device_group):
        self._primary = primary
        self.device_group = device_group

    def __getattr__(self, name):
        return getattr(self._primary, name)

    def all_reduce(self, input_):
        import torch.distributed as dist

        if input_.numel() == 0:
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_


def init_worker(model_runner_factory, primary_runner):
    """Build the latency-role runner, communicator and streams inside a TP worker.

    `model_runner_factory` returns a fresh, unloaded runner; its `.model` is then pointed at
    the primary runner's weights so nothing is loaded twice.
    """
    import torch
    import torch.distributed as dist
    from vllm.distributed import parallel_state as ps

    if not enabled() or LATENCY in _tp_groups:
        return _tp_groups.get(LATENCY), None

    _patch_tp_group_resolution()
    primary = ps._dual_stream_original_get_tp_group()
    _tp_groups[THROUGHPUT] = primary

    # new_group is itself collective: every rank must reach it, in the same order.
    ranks = dist.get_process_group_ranks(primary.device_group)
    secondary = dist.new_group(ranks=ranks, backend="nccl")
    _tp_groups[LATENCY] = _SecondaryTPGroup(primary, secondary)

    least, greatest = torch.cuda.Stream.priority_range()
    _streams[LATENCY] = torch.cuda.Stream(priority=greatest)
    _streams[THROUGHPUT] = torch.cuda.Stream(priority=least)

    latency_runner = model_runner_factory()

    # Assigning `.model` is NOT enough: load_model() also derives per-runner state (e.g.
    # decode_query_len, speculator wiring, memory accounting) and a runner missing it fails
    # later in initialize_kv_cache. So run the real load_model, but swap the loader for one
    # that hands back the already-resident module - all side effects, zero disk reads.
    import importlib
    import sys as _sys

    mod = _sys.modules[type(latency_runner).__module__]
    original_get_loader = mod.get_model_loader

    class _SharedWeightLoader:
        def __init__(self, model):
            self._model = model

        def load_model(self, *args, **kwargs):
            return self._model

        def __getattr__(self, name):
            return getattr(original_get_loader(_LOAD_CFG[0]), name)

    _LOAD_CFG = [latency_runner.vllm_config.load_config]
    mod.get_model_loader = lambda *a, **k: _SharedWeightLoader(primary_runner.model)
    try:
        latency_runner.load_model()
    finally:
        mod.get_model_loader = original_get_loader
    logger.info(
        "dual-stream: latency runner ready (secondary TP comm over ranks %s, "
        "stream priorities lat=%d thr=%d)",
        ranks, greatest, least,
    )
    return _tp_groups[LATENCY], latency_runner


def split_kv_cache_config(kv_cache_config, fraction: float):
    """(throughput_config, latency_config) carved out of one pool.

    num_blocks alone is NOT enough. KVCacheTensor.size is a byte count computed from the
    original block count, and initialize_kv_cache allocates from THAT. A shallow copy with
    only num_blocks changed makes each runner allocate the whole pool, which OOMs at 2x.
    So the per-tensor byte sizes are rescaled to match each side's block count.
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
                # Otherwise scale proportionally, keeping the per-block granularity intact.
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
        thr_blocks, thr_bytes / 2**30, lat_blocks, lat_bytes / 2**30,
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

    if not enabled() or role not in _streams:
        yield None
        return
    prev = getattr(_thread_state, "role", None)
    _thread_state.role = role
    try:
        with torch.cuda.stream(_streams[role]):
            yield _streams[role]
    finally:
        _thread_state.role = prev


def stream(role: str):
    return _streams.get(role)
