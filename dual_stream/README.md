# Dual priority-stream execution (foundational layer)

Lets one vLLM engine run a **latency-critical** stream and a **throughput** stream
concurrently on the same GPUs, sharing one copy of the weights, on two CUDA streams at
different priorities.

Sharing weights is not an optimisation here, it is a requirement. For a large MoE the weights
dominate device memory, so two independent processes cannot both hold a copy — on the model
this was developed against, weights plus non-torch memory were 193.75 GiB per GPU on a
267.69 GiB card. Two copies do not fit; one copy plus two schedulers does.

## Status

**The plumbing in `vllm_dual_stream.py` is untested as an engine integration.** What *has*
been measured is the primitive it depends on — see below. The scheduler, KV-partitioning and
request-routing layers are deliberately not implemented; they are listed as `MISSING` in the
module docstring.

## Files

| file | what it does |
|---|---|
| `vllm_dual_stream.py` | Opt-in overlay (`VLLM_DUAL_STREAM=1`): a second TP NCCL communicator, thread-aware `get_tp_group()`, and two CUDA streams at different priorities. Monkey-patches at import so the installed wheel stays byte-identical. |
| `dual_stream_probe.py` | Standalone feasibility probe. No vLLM dependency. Answers whether the design can work at all before any engine code is written. |

## Run the probe first

```
torchrun --nproc_per_node=8 dual_stream_probe.py --layers 8 --iters 30 --heavy-experts 128
```

It answers three questions:

1. **Deadlock?** Two NCCL communicators issuing concurrent collectives from two threads on two
   streams. NCCL collective kernels are persistent, and two that cannot co-reside can deadlock.
   This is the risk that can kill the design outright.
2. **Does stream priority protect the latency stream?** Priority schedules new blocks; it does
   not preempt running ones, so protection can be much weaker than the API implies.
3. **What does hosting the latency stream cost the throughput stream?**

### Measured on 8xB300, TP=8

| | result |
|---|---|
| deadlock | **no** — concurrent collectives on two comms completed |
| latency stream p50 | **1.02x** its solo latency while a saturating stream ran alongside |
| latency stream p99 | 1.64x — priority protects the median, not the tail |
| throughput stream | retained **87%** of solo throughput |

The 13% cost is not free capacity: at batch 64 the engine is ~8.8x more efficient per token
than at batch 1, so 13% of a 64-way batch is roughly the capacity of 8 sequences, spent to
serve 1 at low latency. The design buys **latency isolation**, not throughput.

Two probe bugs worth knowing about, both fixed and both of which first presented as a NCCL
deadlock: `torch.cuda.set_device()` binds per-thread and is **not** inherited by worker
threads, and fixed-iteration loops of unequal step cost stop overlapping once the fast stream
retires, diluting measured interference toward zero.

## Update: the engine integration now works

The layers listed as `MISSING` above are implemented and serving. Each TP worker holds one
copy of the weights and two model-runner states; `EngineCore` runs two schedulers over
matching KV partitions and routes by request priority (`priority < 0` -> latency engine).

Verified on 8xB300 / Kimi-K3 / TP=8: both roles return correct output over repeated
alternating requests, with the KV pool split 51.1 / 5.7 GiB out of an unchanged 56.8 GiB.

### Known limits

* **Execution is sequential** - the latency batch runs, then the throughput batch. Concurrent
  execution across the two streams, which is the entire point of the architecture, is not
  implemented. The probe numbers (1.02x latency p50, 87% throughput retention) therefore
  remain a projection, not an end-to-end measurement.
* **Requires `--no-async-scheduling`.** With async scheduling the engine drives
  `step_with_batch_queue` rather than `step`, which is not patched.
* **The secondary TP group bypasses the custom all-reduce**, whose single shared staging
  buffer would corrupt under two concurrent forwards. Plain NCCL is correct but slower.
* **A failed latency request kills the engine**, so this is not safe to leave serving.

### Provenance

Developed and verified against a vendor fork of vLLM (`20260803.dev23+g9d083cdd6`), not
against this tree's `main`. The diff applies cleanly here, but "applies cleanly" is textual -
it has **not** been run against upstream vLLM. `dual_engine.patch` is the original patch as
generated on the verified tree.

### The six defects, all found by running it

1. `load_model()` side effects (`decode_query_len`) missing on the second runner
2. `KVCacheConfig` shallow copy kept full tensor byte sizes -> 2x KV -> OOM
3. async scheduling bypassed the patched `step()` entirely
4. `WorkerWrapperBase.execute_model()` dropped the `role` argument
5. `_init_kv_zero_meta()` not run for the latency runner
6. `sample_tokens()` also needs the role - it is the second half of the same forward, so
   sampling against the other runner found no pending batch and silently returned `None`
