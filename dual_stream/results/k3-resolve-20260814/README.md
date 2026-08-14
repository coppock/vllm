# Kimi-K3 Resolve A/B validation — 2026-08-14

## Result

The dual engine passed functional validation but failed the performance
criterion for this configuration. Both modes completed all 62 requests without
an engine error, collective timeout, or NCCL race, but the stock baseline was
substantially faster for both traffic tiers.

| Metric | Stock baseline | Dual engine | Dual / baseline |
| --- | ---: | ---: | ---: |
| Successful requests | 62/62 | 62/62 | — |
| Total elapsed | 237.787 s | 386.027 s | 1.623x |
| Output throughput | 16.502 tok/s | 10.178 tok/s | 0.617x |
| Server mean TTFT, all requests | 3.697 s | 33.378 s | 9.029x |
| Latency-tier E2E p50 | 14.547 s | 141.734 s | 9.743x |
| Latency-tier E2E p95 | 43.640 s | 194.931 s | 4.467x |
| Throughput-tier E2E p50 | 14.333 s | 41.863 s | 2.921x |
| Throughput-tier E2E p95 | 53.116 s | 78.418 s | 1.476x |

## Method

- Hardware: 8 NVIDIA B300 GPUs; Kimi-K3 at tensor parallel size 8.
- Traffic: a 224.269-second production Resolve slice containing 62 requests:
  47 `flex`/throughput and 15 latency requests.
- Both runs used production arrival timing, the same request priorities, eager
  execution, synchronous priority scheduling, prefix caching, a 32,768-token
  scheduler budget, and a 64-token output cap.
- Both processed exactly 7,754,855 input tokens. The baseline produced 3,924
  output tokens and dual produced 3,929 because a few requests naturally
  stopped before the cap.
- Dual configuration: 10% latency KV reservation, latency batch limit 1, CUDA
  stream priorities -3 (latency) and 0 (throughput).
- Each server was started fresh and the replay began only after API readiness.
  Dual ran first, followed by baseline. These are single runs, not confidence
  intervals.

Artifact checksums used for the run:

- `dual_engine.patch`:
  `e284edfc89b6de97f3ced913cf3f711c130d4f818668e736c120bc8a81d1b8b9`
- `resolve_replay.py`:
  `5cb1335d58b460769a8f52b40ff69e96e1ca11af14d5383ec25a022a26865ae0`
- Resolve sample:
  `2bd399150c1a80e0af31b18ca44f87597f3390e02580732b72741f391da48d92`

The replay script recognizes the legacy `reasoning_content` SSE field, while
this vLLM build emits Kimi-K3 reasoning as `delta.reasoning`. Its per-request
TTFT values are therefore incomplete. The table's all-request mean TTFT is
computed from the server's before/after Prometheus counters, which covered all
62 requests. E2E and token usage in the JSON results are complete.

## Interpretation

Two effects dominate this result:

1. The latency scheduler's batch limit of one serialized the 15-request
   latency burst. It prevented co-batching and built a long queue; later latency
   requests reached roughly 200 seconds E2E.
2. Safe dual execution currently disables K3 latent-MoE tail fusion and the
   process-global FlashInfer fused all-reduce path. The sequenced plain
   tensor-parallel fallback removed the deadlock but reduced throughput enough
   to hurt both tiers. CUDA stream priority cannot preempt an already-running
   large kernel.

Before repeating the production A/B, the next implementation should provide a
role-safe fused collective/workspace path and sweep the latency batch limit
(at least 1, 2, 4, and 8). The replay should also recognize
`delta.reasoning`, and tuned candidates should be repeated in alternating run
order.

## Artifacts

- `baseline-full-result.json` and `dual-full-result.json`: per-request timing
  and usage, without generated response text.
- `baseline-full.{before,after}.metrics` and
  `dual-full.{before,after}.metrics`: server metric snapshots used for the
  all-request TTFT calculation.
