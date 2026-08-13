# Dual priority-stream execution

This opt-in vLLM execution mode runs a latency-critical batch and a throughput
batch concurrently in one worker process. Both roles share one copy of the
model weights, but use separate scheduler, runner, KV-cache, workspace, and
CUDA-stream state.

The latency role uses a small scheduler batch on a high-priority CUDA stream.
The throughput role uses the normal scheduler batch on a low-priority stream.
Requests with `priority < 0` go to the latency scheduler; all other requests go
to the throughput scheduler.

## Status

Concurrent execution is working end to end with tensor parallelism. It was
validated on vLLM 0.27.1 with Qwen3.5-0.8B at TP=4 on four NVIDIA A10Gs. The
durable upstream branch also passes its focused unit tests, Ruff, formatting,
mypy, repository policy hooks, Python compilation, and `git diff --check`.

An earlier sequential version was validated with Kimi-K3 at TP=8 on eight
B300s. The current concurrent vendor patch is ready, but has not yet been run
with Kimi-K3.

## How it works

`EngineCore` owns two schedulers over matching partitions of the available KV
cache. When both roles have work, it sends both scheduler outputs in one RPC.
Each GPU worker fans that RPC out to two host threads and launches the model
runners on separate CUDA streams.

The two runners share the model module and weights. Their mutable state is not
shared: input batches, block tables, KV tensors, forward contexts, attention
workspaces, and FlashInfer workspaces are role-specific.

Both roles use the existing TP process group. Host-side collective calls are
ordered identically on every rank, and CUDA events serialize collective kernel
execution while allowing non-collective model compute to overlap. Plain
`torch.distributed` all-reduce is used in this path because vLLM's custom
all-reduce owns shared staging buffers that are not safe for two in-flight
forwards.

This collective sequencing is required for correctness. The original
prototype created a second NCCL communicator and launched from two host
threads. With `NCCL_LAUNCH_ORDER_IMPLICIT=1` and
`NCCL_LAUNCH_RACE_FATAL=1`, NCCL reported a host-thread launch race. Stack
traces at sampled-token copies were only where the CPU first observed the
unfinished GPU work; those copies were not the root cause.

## Running it

Use eager, synchronous scheduling:

```bash
VLLM_DUAL_STREAM=1 \
VLLM_DUAL_STREAM_LATENCY_KV_FRAC=0.10 \
VLLM_DUAL_STREAM_LATENCY_MAX_NUM_SEQS=1 \
vllm serve MODEL \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --no-async-scheduling
```

Send a negative OpenAI request priority for latency-sensitive work:

```json
{
  "model": "MODEL",
  "prompt": "...",
  "priority": -1,
  "max_tokens": 16
}
```

Environment variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `VLLM_DUAL_STREAM` | `0` | Enable the feature when set to `1`. |
| `VLLM_DUAL_STREAM_LATENCY_KV_FRAC` | `0.10` | Fraction of KV blocks reserved for latency requests. |
| `VLLM_DUAL_STREAM_LATENCY_MAX_NUM_SEQS` | `1` | Maximum latency-role batch size. |
| `VLLM_DUAL_STREAM_WARMUP_STEPS` | `0` | Optional sequential mixed steps for diagnosis. Normal operation should leave this at zero. |

Startup rejects unsupported combinations instead of silently falling back:
CUDA graphs, async scheduling, ubatching/DBO, KV or encoder-cache transfer
connectors, and parallel modes other than tensor parallelism.

## Validation results

The TP=4 stress workload used eight throughput requests at 160 output tokens
and four staggered latency requests at 16 output tokens.

| Workload | Result |
| --- | ---: |
| Throughput only | 1,280 output tokens in 5.940 s, 215.48 tok/s |
| Latency only | completions at 0.610, 1.203, 1.800, 2.395 s |
| Mixed, repeated run | latency completions at 1.664, 3.261, 4.826, 6.414 s |
| Mixed, repeated run | throughput requests completed in 10.395-10.440 s |
| Mixed, repeated run | 128.73 combined tok/s; no NCCL races or engine errors |

These numbers establish correctness and real overlap, not ideal isolation. CUDA
stream priority affects pending block scheduling and does not preempt already
running blocks. On this small A10G model, mixed latency was about 2.7x solo and
throughput retained about 57% of its solo rate. The B300/Kimi-K3 workload must
be benchmarked before drawing production capacity conclusions.

## Files

| File | Purpose |
| --- | --- |
| `vllm/v1/worker/dual_stream.py` | Role streams, shared-TP collective sequencing, KV split, and shared-weight runner setup. |
| `vllm/v1/engine/core.py` | Dual schedulers, priority routing, and merged mixed-role steps. |
| `vllm/v1/worker/gpu_worker.py` | Per-role runners and concurrent worker execution. |
| `vllm/forward_context.py` | Thread-safe forward context using `ContextVar`. |
| `vllm/distributed/parallel_state.py` | Thread-safe TP-group override visible through existing imports. |
| `vllm/v1/worker/workspace.py` | Separate latency scratch workspace. |
| `vllm/v1/attention/backends/flashinfer.py` | Separate latency FlashInfer workspace. |
| `dual_engine.patch` | Current combined patch for vendor vLLM `20260803.dev23+g9d083cdd6`. |
| `dual_stream_probe.py` | Standalone CUDA/NCCL feasibility benchmark. |

`vllm_dual_stream.py` is retained only as a historical overlay from the first
prototype. The in-tree implementation above is authoritative.

## Current limits

- CUDA/NCCL and tensor parallelism only. Pipeline, data, context, and expert
  parallel collectives have not been integrated with the role sequencer.
- CUDA graphs, async scheduling, ubatching/DBO, and cache-transfer connectors
  are disabled.
- The latency KV partition is fixed at startup. A latency request exceeding
  that reservation cannot borrow blocks from the throughput partition.
- LoRA mutation, sleep/wake, elastic scaling, speculative decoding, pooling,
  and structured outputs need dedicated concurrent stress coverage before
  production use.
- Stream priority is advisory scheduling, not GPU preemption.
