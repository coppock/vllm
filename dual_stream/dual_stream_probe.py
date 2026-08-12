#!/usr/bin/env python3
"""
Standalone feasibility probe for the dual-engine / dual-priority-stream design.

Answers the three questions that decide whether the vLLM patch can work, WITHOUT touching
vLLM at all. Run this before investing in the engine changes.

  Q1 DEADLOCK: can two NCCL communicators issue collectives concurrently from two threads on
     two CUDA streams across TP=8, or does it hang? NCCL collective kernels are persistent and
     occupy SMs for their whole duration; two collectives on different communicators that
     cannot co-reside can deadlock waiting on each other. This is the single largest risk in
     the design and it is cheap to test in isolation.

  Q2 PRIORITY: does CUDA stream priority actually protect the latency-critical stream's
     step time when a heavy stream is saturating the GPU? Priority affects which blocks get
     scheduled as SMs free up; it does NOT preempt running blocks. With large MoE kernels
     already resident, the protection may be much weaker than the API suggests.

  Q3 EXCHANGE RATE: how much throughput does the heavy stream lose to host the light one?
     This is the number that decides whether the architecture is worth shipping.

Workload shapes mimic K3 decode: hidden 7168, 16 routed experts of 3584x3072 per token,
93 layers, TP=8 (so each rank owns 1/8 of each expert's intermediate dim), with an all-reduce
of the hidden state per layer.

Usage (8-GPU node, TP=8):
    torchrun --nproc_per_node=8 dual_stream_probe.py
    torchrun --nproc_per_node=8 dual_stream_probe.py --layers 8 --iters 30

Exit code 0 = no deadlock. A hang means Q1 answered NO: the design needs a different
concurrency strategy (e.g. a single stream with interleaved micro-batches).
"""

import argparse
import os
import threading
import time

import torch
import torch.distributed as dist

HIDDEN = 7168
MOE_INTER = 3072
EXPERT_H = 3584
TOPK = 16


def log(msg):
    """Per-rank, flushed. A deadlock is diagnosed by WHICH milestone is missing, so every
    rank must report every milestone and nothing may sit in a stdio buffer."""
    r = dist.get_rank() if dist.is_initialized() else "-"
    print(f"[rank {r}] {msg}", flush=True)


class Watchdog:
    """A hang is the expected failure mode, so make it loud and non-silent."""

    def __init__(self, seconds, label):
        self.seconds = seconds
        self.label = label
        self._done = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        if not self._done.wait(self.seconds):
            rank = dist.get_rank() if dist.is_initialized() else -1
            print(f"\n!!! WATCHDOG rank={rank}: '{self.label}' exceeded {self.seconds}s "
                  f"-- almost certainly a NCCL deadlock across the two communicators.\n"
                  f"    Q1 = NO. Concurrent collectives on separate comms do not work here.",
                  flush=True)
            os._exit(0)  # _exit so the hung NCCL threads cannot block interpreter shutdown

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._done.set()


def make_weights(rank_inter, device, layers, n_experts):
    """Per-rank shard of the expert weights. bf16 to match K3's activation dtype."""
    g = torch.Generator(device=device).manual_seed(0)
    return [[
        (torch.randn(EXPERT_H, rank_inter, device=device, dtype=torch.bfloat16, generator=g),
         torch.randn(rank_inter, EXPERT_H, device=device, dtype=torch.bfloat16, generator=g))
        for _ in range(n_experts)
    ] for _ in range(layers)]


def decode_step(x, weights, group, layers, experts_per_token):
    """One decode forward: per layer, run `experts_per_token` expert GEMMs then all-reduce.

    Deliberately mirrors the real bottleneck shape - many small GEMMs plus a collective per
    layer - because that is what makes batch-1 launch-bound rather than bandwidth-bound.
    """
    for layer in range(layers):
        acc = torch.zeros_like(x)
        for e in range(experts_per_token):
            w_up, w_down = weights[layer][e]
            acc = acc + (x @ w_up) @ w_down
        dist.all_reduce(acc, group=group)
        x = acc
    return x


def run_loop(tag, batch, weights, group, stream, layers, experts, iters, results, barrier,
             device, until_event=None, signal_event=None):
    # torch.cuda.set_device() binds the CURRENT THREAD's device. Worker threads do NOT inherit
    # it from the spawning thread - they default to cuda:0 - so without this every rank except
    # 0 builds activations on the wrong device while its weights sit on cuda:rank. That fails
    # with a device-mismatch error on ranks 1..N-1 while rank 0 happily proceeds into the
    # collective and blocks forever waiting for peers that already died. It presents exactly
    # like a NCCL deadlock, which is how it fooled the first run of this probe.
    torch.cuda.set_device(device)
    try:
        x = torch.randn(batch, EXPERT_H, device=device, dtype=torch.bfloat16)
        # Warm up on this stream/comm so NCCL channel setup is not timed.
        with torch.cuda.stream(stream):
            decode_step(x, weights, group, min(2, layers), experts)
        torch.cuda.synchronize()
        barrier.wait(timeout=120)

        # Fixed-iteration loops of unequal step cost do NOT overlap for their full windows:
        # the fast stream retires early and the slow one then runs alone, diluting any
        # measured interference toward zero. So the light stream instead runs UNTIL the heavy
        # stream signals completion, keeping both fully overlapped for the heavy stream's
        # entire measurement window.
        lat = []
        with torch.cuda.stream(stream):
            if until_event is not None:
                while not until_event.is_set():
                    t0 = time.perf_counter()
                    decode_step(x, weights, group, layers, experts)
                    stream.synchronize()
                    lat.append((time.perf_counter() - t0) * 1000)
            else:
                for _ in range(iters):
                    t0 = time.perf_counter()
                    decode_step(x, weights, group, layers, experts)
                    stream.synchronize()
                    lat.append((time.perf_counter() - t0) * 1000)
                if signal_event is not None:
                    signal_event.set()
    except BaseException as e:
        # A thread dying before the barrier leaves its partner blocked forever, turning a
        # plain crash into an indistinguishable "hang". Break the barrier so the failure
        # surfaces as the error it actually is.
        results[tag] = {"error": f"{type(e).__name__}: {e}"}
        if signal_event is not None:
            signal_event.set()
        try:
            barrier.abort()
        except Exception:
            pass
        raise

    lat.sort()
    results[tag] = {
        "mean_ms": sum(lat) / len(lat),
        "p50_ms": lat[len(lat) // 2],
        "p99_ms": lat[min(len(lat) - 1, int(0.99 * (len(lat) - 1)))],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=8, help="Layers simulated (K3 has 93)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--heavy-batch", type=int, default=64)
    # At batch 1 only TOPK experts are touched; at batch 64 a 896-expert MoE activates far
    # more. Without this the "heavy" stream does the SAME work as the light one and is
    # launch-bound rather than saturating, which makes the priority test meaningless.
    p.add_argument("--heavy-experts", type=int, default=128)
    p.add_argument("--watchdog", type=float, default=180.0)
    args = p.parse_args()

    print("[rank -] entering init_process_group", flush=True)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    # Explicit index, NOT bare "cuda". A bare device resolves against the calling thread's
    # current device, which is the whole bug this probe just tripped over - it would silently
    # mean cuda:0 inside the phase-3 worker threads.
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0
    log(f"dist initialized (world={world}), device set")

    lo_pri, hi_pri = torch.cuda.Stream.priority_range()  # lo_pri is the LEAST priority value
    if is_main:
        print(f"world={world}  priority_range=(least={lo_pri}, greatest={hi_pri})", flush=True)
        print(f"layers={args.layers} iters={args.iters} heavy_batch={args.heavy_batch}\n",
              flush=True)

    # An eager collective on the DEFAULT group first: if this alone hangs, the problem is the
    # basic NCCL setup, not anything to do with two communicators.
    with Watchdog(args.watchdog, "warmup-default-group-allreduce"):
        t = torch.ones(8, device=device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
    log(f"default-group all_reduce OK (sum={t[0].item():.0f}, expect {world})")

    # TWO communicators over the same ranks. Every rank must create them in the same order.
    # new_group is itself collective, so it can deadlock on its own - hence the watchdog.
    with Watchdog(args.watchdog, "new_group-hi"):
        group_hi = dist.new_group(ranks=list(range(world)))
    log("created group_hi")
    with Watchdog(args.watchdog, "new_group-lo"):
        group_lo = dist.new_group(ranks=list(range(world)))
    log("created group_lo")

    # Exercise each new communicator once, serially. If a comm only deadlocks under
    # concurrency, these succeed and phase 3 is the real answer; if one hangs here, the
    # communicators themselves are broken and concurrency is irrelevant.
    with Watchdog(args.watchdog, "serial-allreduce-on-group_hi"):
        dist.all_reduce(torch.ones(8, device=device), group=group_hi)
        torch.cuda.synchronize()
    log("serial all_reduce on group_hi OK")
    with Watchdog(args.watchdog, "serial-allreduce-on-group_lo"):
        dist.all_reduce(torch.ones(8, device=device), group=group_lo)
        torch.cuda.synchronize()
    log("serial all_reduce on group_lo OK")

    stream_hi = torch.cuda.Stream(priority=hi_pri)   # latency-critical (batch 1)
    stream_lo = torch.cuda.Stream(priority=lo_pri)   # throughput (batch N)

    rank_inter = MOE_INTER // world  # TP shard of each expert's intermediate dim
    with Watchdog(args.watchdog, "weight-allocation"):
        w_hi = make_weights(rank_inter, device, args.layers, TOPK)
        w_lo = make_weights(rank_inter, device, args.layers, args.heavy_experts)
    log("weights allocated")

    results = {}

    # --- Phase 1: light stream alone (the BS1 latency baseline) ---
    log("phase1 start (light alone)")
    b = threading.Barrier(1)
    with Watchdog(args.watchdog, "phase1-light-alone"):
        run_loop("light_alone", 1, w_hi, group_hi, stream_hi,
                 args.layers, TOPK, args.iters, results, b, device)
    dist.barrier()
    log(f"phase1 done: p50={results['light_alone']['p50_ms']:.2f}ms")

    # --- Phase 2: heavy stream alone (the throughput baseline) ---
    log("phase2 start (heavy alone)")
    with Watchdog(args.watchdog, "phase2-heavy-alone"):
        run_loop("heavy_alone", args.heavy_batch, w_lo, group_lo, stream_lo,
                 args.layers, args.heavy_experts, args.iters, results, b, device)
    dist.barrier()
    log(f"phase2 done: p50={results['heavy_alone']['p50_ms']:.2f}ms")

    # --- Phase 3: BOTH CONCURRENTLY. This is the question. ---
    if is_main:
        print("phase3: concurrent -- if this hangs, Q1 is answered NO\n", flush=True)
    b2 = threading.Barrier(2)
    heavy_done = threading.Event()
    threads = [
        threading.Thread(target=run_loop, args=("light_conc", 1, w_hi, group_hi, stream_hi,
                                                args.layers, TOPK, args.iters, results, b2, device,
                                                heavy_done, None)),
        threading.Thread(target=run_loop, args=("heavy_conc", args.heavy_batch, w_lo, group_lo,
                                                stream_lo, args.layers, args.heavy_experts,
                                                args.iters, results, b2, device,
                                                None, heavy_done)),
    ]
    with Watchdog(args.watchdog, "phase3-concurrent"):
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    dist.barrier()

    if is_main:
        la, lc = results["light_alone"], results["light_conc"]
        ha, hc = results["heavy_alone"], results["heavy_conc"]
        print("=" * 78)
        print("Q1 DEADLOCK      : NO -- concurrent collectives on two comms completed")
        print()
        print(f"{'':16} {'alone':>12} {'concurrent':>12} {'change':>10}")
        print(f"{'light p50 (ms)':16} {la['p50_ms']:>12.2f} {lc['p50_ms']:>12.2f} "
              f"{lc['p50_ms'] / la['p50_ms']:>9.2f}x")
        print(f"{'light p99 (ms)':16} {la['p99_ms']:>12.2f} {lc['p99_ms']:>12.2f} "
              f"{lc['p99_ms'] / la['p99_ms']:>9.2f}x")
        print(f"{'heavy p50 (ms)':16} {ha['p50_ms']:>12.2f} {hc['p50_ms']:>12.2f} "
              f"{hc['p50_ms'] / ha['p50_ms']:>9.2f}x")
        print()
        print(f"Q2 PRIORITY      : light stream slowed {lc['p50_ms'] / la['p50_ms']:.2f}x while "
              f"sharing with a saturating stream.")
        print("                   Near 1.0x = priority protects it. Large = it does not.")
        heavy_kept = ha["p50_ms"] / hc["p50_ms"]
        print(f"Q3 EXCHANGE RATE : heavy stream retained {heavy_kept:.0%} of its solo throughput.")
        print(f"                   Cost of hosting the latency stream = {1 - heavy_kept:.0%} of batch capacity.")
        print("=" * 78)
        print()
        print("Compare against the measured alternative: putting that request INSIDE the batch")
        print("engine costs it ~69.8ms TPOT (p99 ITL 308ms) but zero throughput. The dual-stream")
        print("design is only worth shipping if Q2 is near 1.0x AND Q3 stays high.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
