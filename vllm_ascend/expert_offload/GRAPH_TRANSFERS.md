# Static expert-pool graph transfers

The first version records full local EP-shard H2D transfers on the consuming
compute stream. Each layer reloads its assigned pool slot on every replay.
No Host callback, Host wait, residency shortcut, or secondary transfer stream
is used by this path. Stream ordering protects both read-after-load and
overwrite-after-consumption dependencies.

## Scheduler and adapters

- `PoolTransferScheduler` owns immutable per-layer/per-slot plans and their
  backing tensors. Eager warmup prepares a plan; capture rejects missing plans.
- `PoolCopy` aliases physical bytes through base-format uint8 tensors. Sources
  must be pinned CPU memory and destinations must be NPU memory.
- `ExpertPoolAdapter` maps the local EP shard and supplies packed weights,
  scales, offsets and scale biases present in the existing manager schema.
- Quant fields preserve the legacy logical copy semantics. If source and
  destination physical layouts differ, the adapter creates matching pinned
  scale staging once, outside capture. Expert weight payloads are not duplicated.
- Existing FP32 scale caches are refreshed by device-side dtype conversion
  into preallocated storage, after their H2D dependencies.
- A quant method can implement
  `build_pool_transfer_plan(manager, layer_idx, pool_slot)` to return a
  `PoolTransferPlan` with additional fields and post-copy device operations.
  Each post-operation implements `submit()`: enqueue fixed-address device
  work on the current stream, without CPU decisions or synchronization.

## Lifetime and scope

Weights, layouts, buffers and staging contents are immutable after preparation.
Retire all referencing graphs before closing the scheduler or reloading weights.
Workers must serialize eager/graph forwards that share the same pool.
Each graph shape records the complete layer-load sequence, even if a pool slot
contains that layer's weights during capture.

Integration covers the manager's torch-backed, full local-shard path. Native
MXFP4 ownership exchange, pointer-only MemFabric sources and delta/split-GMM
pool loading do not acquire graph support from this implementation. Unsupported
backends fail explicitly during capture.

Passing synthetic quant-field tests does not establish support for every
model/quantization combination. A new layout must satisfy the adapter contract
and pass model-level eager/graph comparisons.

## Validation

Unit tests:
`tests/ut/expert_offload/test_prefill_transfer.py`

NPU GLOBAL-capture tests:
`tests/e2e/pull_request/one_card/aclgraph/test_expert_pool_transfer.py`

The NPU tests verify changed Host contents across replay, alternating graph
sizes, two-slot reuse across three layers, rank-one shard mapping, strided
uint8 quant fields, and FP16-to-FP32 cache conversion. Manager integration is
exercised through `_prefill_load_layer_shard`.

Run only on an idle visible NPU with an allocator supporting pinned async H2D.
The current installation passed both `pinned_mem_register:True` and
`pin_memory_expandable_segments:True` capture probes (separately).

Full DeepSeek-V4 EP=2/DP=2 acceptance is pending: current model loading failed
allocating pinned Host buffers before the new transfer path ran. The machine's
other processes and global memory settings were not changed.

Transport benchmark (run separately on an idle device):
`benchmarks/expert_offload/benchmark_pool_transfer.py` reports device latency,
wall latency and bandwidth for eager and GLOBAL graph copies. No model
performance claim is based on this transport-only benchmark.

Current validation summary: 12 new unit tests plus 2 existing layout tests pass;
3 NPU GLOBAL-capture tests pass. The related manager suite retains the same
9 previously reproduced baseline failures. Full format CI cannot run because
pre-commit/ruff are absent and the container's package repositories do not
resolve. Syntax and whitespace checks pass.


## Full-model EP=2 / DP=2 graph routing (2026-11-13)

The full-shard torch-backed path now uses fixed-shape EP communication during
FULL graph execution. Each rank gathers activations, expert IDs and routing
weights, computes its contiguous local expert shard, then reduce-scatters
weighted contributions back to token owners. BF16 activations are quantized by
the existing MLP, preserving the earlier BF16 AllToAll path's quantization
boundary. The graph has no Host split-count read or data-dependent repeat.

With FULL decode graphs enabled, target and draft selection consistently use
the complete shard pool, including small batches. The resident expert cache is
not paged inside these graphs. Every graph replay still reloads all referenced
expert layers through the static H2D scheduler. This trades cache reuse and
activation communication cost for a capture-safe full-shard execution path.

The static_ep.py module accepts floating 2D activations and equal contiguous
EP shards. Prequantized inputs and MoE LoRA are rejected explicitly. Captured
token rows must be uniform across ranks, as ensured by DP graph padding.
Uncaptured prefill retains the existing dynamic AllToAll path.

Validation includes a real two-NPU GLOBAL-capture test with changing routes,
changing activations and alternating graph shapes (8, 32 and 48 token rows):
tests/e2e/pull_request/multi_card/aclgraph/test_static_ep_dispatch.py.
New regression tests cover graph dispatch selection and input contracts.

The complete DeepSeek-V4 model now captures all four FULL decode graphs on
both ranks (0.21 GiB per rank). API smoke tests produce correct single and
concurrent completions; worker logs confirm actual ACL graph replay.
This is FULL_DECODE_ONLY, not FULL prefill capture. Performance is not benchmarked.

Container-only service entry point:
bash /mnt/share/l00955465/multi-card/run_moe_offload_ep2_dp2_service.sh
It sources CANN in the subprocess and uses the graph launcher without an
eager-mode fallback. Current evidence and logs are in
diagnostics/fullgraph-20261113-114913/.
