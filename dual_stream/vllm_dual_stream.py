"""
Dual-priority-stream support for vLLM (foundational layer).

Implements the plumbing the dual-engine design needs, WITHOUT editing the installed wheel:
a second tensor-parallel NCCL communicator, thread-aware resolution of "which TP group am I
on", and two CUDA streams at different priorities. Enable with:

    PYTHONPATH=/path/to/this VLLM_DUAL_STREAM=1 vllm serve ...

Why an overlay rather than a source patch: it keeps an installed wheel byte-identical, so
rollback is "unset the env var" rather than "reinstall the wheel". That matters when the same
environment is also serving traffic.

WHY THIS LAYER FIRST
--------------------
vLLM resolves the TP group through a module-level singleton `_TP` in
vllm.distributed.parallel_state. That is fine when exactly one forward pass is in flight. The
moment two forwards run concurrently - which is the entire point of the dual-engine design -
both would issue collectives on the SAME communicator from two threads, which is undefined
and deadlock-prone. So every higher layer (dual scheduler, split KV, request routing) is built
on the assumption that this layer works.

Whether it works is an empirical question about NCCL, not a design question: NCCL collective
kernels are persistent and hold SMs for their duration, so two collectives on separate
communicators can deadlock if they cannot co-reside. Validate with dual_stream_probe.py on the
target hardware BEFORE building anything on top of this.

STATUS: untested. Written against vllm 20260803.dev23+g9d083cdd6. The scheduler/KV layers are
deliberately NOT implemented yet - see MISSING below.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading

logger = logging.getLogger(__name__)

# Which TP group the *current thread* should use. Unset -> fall back to vLLM's global _TP,
# so any code path that is not part of a dual-stream forward behaves exactly as before.
_thread_state = threading.local()

LATENCY = "latency"      # batch-1, high stream priority
THROUGHPUT = "throughput"  # batched, default priority

_streams: dict[str, "object"] = {}
_tp_groups: dict[str, object] = {}
_installed = False


def enabled() -> bool:
    return os.environ.get("VLLM_DUAL_STREAM", "0") == "1"


def _install_tp_group_override() -> None:
    """Make get_tp_group() thread-aware.

    Kept deliberately minimal: if the current thread has not opted in via `on_stream`, this is
    the original function. That keeps warmup, profiling, CUDA graph capture and every other
    single-forward path on exactly the original code path.
    """
    from vllm.distributed import parallel_state as ps

    if getattr(ps, "_dual_stream_patched", False):
        return

    original_get_tp_group = ps.get_tp_group

    def get_tp_group_threadaware():
        role = getattr(_thread_state, "role", None)
        if role is not None:
            group = _tp_groups.get(role)
            if group is not None:
                return group
        return original_get_tp_group()

    ps.get_tp_group = get_tp_group_threadaware
    ps._dual_stream_original_get_tp_group = original_get_tp_group
    ps._dual_stream_patched = True
    logger.info("dual-stream: get_tp_group() is now thread-aware")


def init_secondary_tp_group() -> None:
    """Create a second TP communicator over the same ranks as the primary.

    Every rank must call this at the same point in program order - NCCL communicator creation
    is a collective operation, and creating them in different orders on different ranks is
    itself a deadlock.
    """
    import torch.distributed as dist
    from vllm.distributed import parallel_state as ps

    if THROUGHPUT in _tp_groups:
        return

    primary = ps._dual_stream_original_get_tp_group() if getattr(
        ps, "_dual_stream_patched", False) else ps.get_tp_group()

    # The throughput role reuses the existing communicator, so the common path is unchanged
    # and only the latency role pays for a new one.
    _tp_groups[THROUGHPUT] = primary

    ranks = dist.get_process_group_ranks(primary.device_group)
    secondary = dist.new_group(ranks=ranks, backend="nccl")

    # Wrap so callers see the same GroupCoordinator API while collectives land on the new comm.
    _tp_groups[LATENCY] = _SecondaryTPGroup(primary, secondary)
    logger.info("dual-stream: created secondary TP communicator over ranks %s", ranks)


class _SecondaryTPGroup:
    """Delegates to the primary GroupCoordinator but routes collectives to a second comm.

    Only the collectives actually used on the decode path are overridden; everything else
    (rank bookkeeping, device info) is inherited unchanged from the primary coordinator.
    """

    def __init__(self, primary, device_group):
        self._primary = primary
        self.device_group = device_group

    def __getattr__(self, name):
        return getattr(self._primary, name)

    def all_reduce(self, input_):
        import torch.distributed as dist

        # NOTE: this bypasses vLLM's custom all-reduce fast path on purpose. The custom
        # allreduce uses a single preallocated signal/staging buffer per process; two
        # concurrent forwards would corrupt each other's buffers. Falling back to plain NCCL
        # is correct but slower, and reclaiming that speed means per-role custom-allreduce
        # buffers.
        if input_.numel() == 0:
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_


def init_streams() -> None:
    """Two CUDA streams: latency at greatest priority, throughput at least priority."""
    import torch

    if _streams:
        return
    least, greatest = torch.cuda.Stream.priority_range()
    _streams[LATENCY] = torch.cuda.Stream(priority=greatest)
    _streams[THROUGHPUT] = torch.cuda.Stream(priority=least)
    logger.info("dual-stream: streams created (latency prio=%d, throughput prio=%d)",
                greatest, least)


@contextlib.contextmanager
def on_stream(role: str):
    """Run a forward pass as `role`: its stream, its TP communicator.

    Both must be set together. Executing on the latency stream while collectives go to the
    throughput communicator is exactly the race this module exists to prevent.
    """
    import torch

    assert role in (LATENCY, THROUGHPUT), role
    if not _streams:
        init_streams()

    prev_role = getattr(_thread_state, "role", None)
    _thread_state.role = role
    try:
        with torch.cuda.stream(_streams[role]):
            yield _streams[role]
    finally:
        _thread_state.role = prev_role


def install() -> None:
    """Idempotent. Safe to call from every worker process."""
    global _installed
    if _installed or not enabled():
        return
    _install_tp_group_override()
    _installed = True
    logger.info("dual-stream: overlay installed")


# ---------------------------------------------------------------------------------------
# MISSING - deliberately not implemented until dual_stream_probe.py passes on the target box.
#
# 1. KV PARTITIONING. EngineCore builds one Scheduler from one kv_cache_config
#    (v1/engine/core.py:153). Dual engines need that block pool split, e.g. a small fixed
#    reservation for the latency engine and the remainder for throughput. Splitting by block
#    count is straightforward; what is NOT is that both schedulers then believe they own a
#    KVCacheManager, so prefix-cache hashing must not be shared across them or one engine will
#    hand the other's blocks out.
#
# 2. DUAL SCHEDULER + CONCURRENT STEP. EngineCore.step() (v1/engine/core.py:576) is a strict
#    schedule -> execute -> update cycle over a single scheduler. Dual mode needs both
#    schedulers stepped with their forwards in flight simultaneously, each under on_stream().
#    The subtlety is that update_from_output() must not be serialized behind the *other*
#    engine's forward, or the latency engine inherits the throughput engine's step time and
#    the whole design buys nothing.
#
# 3. REQUEST ROUTING. vLLM already carries a per-request `priority` field, so routing can key
#    off it (priority < 0 -> latency engine) without new API surface.
#
# 4. CUDA GRAPHS. Graphs are captured against a stream; replaying on another is invalid. Either
#    capture per role (roughly doubles CUDA graph memory) or run
#    the latency engine eager. Eager is the safer first cut, but then the mns=1 and mns=64
#    baselines must be re-measured eager or the comparison confounds architecture with graphs.
#
# 5. PER-ROLE KERNEL WORKSPACES. FlashInfer/MoE scratch buffers are typically process-global.
#    Two concurrent forwards will race on them. This needs an audit of the model's MoE/MLA
#    kernel path, including any vendor-specific fused kernels. A shared workspace here is
#    silent numerical corruption, not a crash, which makes it the most dangerous item listed.
# ---------------------------------------------------------------------------------------
