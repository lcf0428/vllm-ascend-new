# Global Host expert store, version 1

This is an opt-in functional implementation. A synthetic BF16 MoE has passed
TP=2/EP=2 MC2 FULL capture/replay on two NPUs, including expert paging and
ALLTOALL fallback. DeepSeek FP4 inference, DSpark end-to-end generation, TP=4
FULL graphs and real RDMA transfers still require separate acceptance.

## Supported scope

- v1 runner, PP=PCP=DCP=1, static EP, identical expert layouts on peers.
- TP=1 retains the existing DP/MC2 graph path. TP>1 requires DP=1 and supports
  eager or MC2 FULL decode with `graph.mc2_decode=true`. Experts remain complete
  (`moe_config.tp_size=1`); physical TP distributes token rows before EP.
  Combined TP>1/DP>1 remains unsupported.
- One immutable canonical Host owner shard per EP rank; one Host copy per
  expert across the EP group. Placement on NPUs is independent of Host ownership.
- Target-model layers already eligible for expert offload. MTP layers skipped
  by the existing loader remain outside this store. Late offloaded layers are
  rejected after publication.
- Eager prefill/overflow through the existing full local EP pool. Small decode
  uses the existing MC2 dispatcher and stable global placement.
- Prefetch, substitution, pruning, CPU MoE, hot preload, exclusive storage,
  prefill delta/split and dynamic EPLB are outside version 1.

## Configuration

Set these fields inside `expert_offload_config` in `--additional-config`:

```json
{
  "expert_offload": true,
  "enable_multi_card": true,
  "h2d_backend": "torch",
  "num_device_experts": [128],
  "num_device_layers": 2,
  "cache_policy_enabled": true,
  "host_storage": {
    "mode": "global_sharded",
    "replicas": 1,
    "backing": "memfd",
    "runtime_dir": "/data/l00955465/multi-card/runtime/expert_store",
    "startup_timeout_s": 600
  },
  "transport": {
    "intra_node": "shared_memory",
    "inter_node": "disabled",
    "staging_buffer_experts": 4,
    "timeout_s": 60
  },
  "placement": {"scope": "ep_group", "policy": "stable"},
  "graph": {"mc2_decode": true}
}
```

`host_storage.mode` defaults to `legacy`; existing launchers stay on their
original path. `graph.mc2_decode` defaults to false and must be explicitly
enabled to exercise the new graph integration. Unknown nested settings fail.

`num_device_experts` remains the **global per-layer** NPU capacity. With EP=4,
128 means 32 slots per rank. The existing restriction remains: capacity must
be positive, below the layer's logical expert count, and divisible by EP.
This implementation does not enable 336 slots for a 256-expert layer.

`shard_per_rank` may be omitted (defaults to true). Explicit false conflicts
with `global_sharded`; true controls startup Host ownership only, and does
not constrain NPU placement in this mode.

`backing=memfd` uses anonymous shared physical pages. It does not consume
`/data` disk space or the container's `/dev/shm` mount capacity. Peers reopen
the owner FD through `/proc/<pid>/fd/<fd>`. This requires a compatible process
namespace and permission to open that FD. A nonce-file visibility probe and
successful mapping are required before treating a peer as local. Separate
containers/namespaces may need a network transport even on one physical host.

`backing=file` uses MAP_SHARED files in the job-specific runtime directory;
it requires enough backing filesystem space and can page to disk. No tmpfs
mount, container configuration or server environment is changed automatically.

Each rank has one pinned allocation split into two DMA banks. Configure
`transport.staging_buffer_experts` (integer >= 2, default 4): this is the
**total expert capacity per rank across both banks**, not the resident-device
expert count and not the capacity of each bank. With 4, each bank holds two
experts; odd budgets split floor/ceil (3 gives 1+2).

At startup, each field's actual `untyped_storage().nbytes()` is rounded up to
64-byte alignment. The largest complete expert payload across all published
layers, including weights, scales, offsets and biases, defines one budget
unit. Total pinned bytes = this unit times `staging_buffer_experts`. Raw/NZ
padding is included; there is no hardcoded model size or quantization type.
For a 12.75 MiB expert whose fields already meet alignment, 4 experts allocate
51 MiB per rank (25.5 MiB per bank). Startup logs the expert budget and both
bank sizes. Increasing this setting changes Host staging memory, not HBM.

Fields are packed into disjoint aligned ranges, submitted asynchronously on
the load stream, and protected by a completion event per bank. Filling the
other bank can overlap H2D. Only reuse of an unfinished bank waits; the final
layer synchronization and distributed READY/commit protocol remain in place.
Logical quant copies preserve shape/stride/dtype and retain temporary views
until completion. Raw fields can still be chunked for custom small buffers.
`staging_buffer_mb` is removed: replace it with `staging_buffer_experts` in old
configurations; old input fails with an explicit migration error.

There is no private full Host replica per reader. Device caches, bounded
staging, checkpoint mappings and temporary startup conversion buffers are
separate from the canonical store. Startup temporarily holds old owner
buffers and the new shared allocation; old buffers and their process-local
pinned allocator cache are released after publication. This is not a zero-copy
startup implementation. Published layers reject checkpoint reloading and
repeated finalization cannot convert their bytes a second time.

## Same-host control communication

Set `transport.control_backend="shared_memory"` to replace per-layer CPU Gloo
counts/router-score reductions and object-based materialization READY with a
small, CPU-polled shared control mapping. The default remains `gloo`. The DP2
launcher opts into `shared_memory`; its expert staging budget is unchanged.

All EP ranks must map each other's canonical regions and the same control
inode. Startup checks identical backend, expert count and timeout settings.
An explicitly selected shared backend fails startup if these conditions are
not met; it never changes to Gloo mid-execution. Multi-node groups use `gloo`.
Bootstrap and canonical-store shutdown still use the existing object collective.
Device dispatch/combine and worker-level DP synchronization are unchanged.

Rank zero builds the bundled `control.c` with `cc` into its private, randomly
named job directory during startup. This requires a C11 compiler, executable
file mappings, and lock-free 64-bit CPU atomics (x86-64/AArch64). There is no
CANN/Torch compilation, global installation or per-layer build. The control
file and DSO are mode 0600; the existing job directory is mode 0700. Each rank
maps the small file with MAP_SHARED; no explicit filesystem flush occurs in
the hot path. Only these control files are removed during normal retirement.

Counts and optional score sums share one publication. Native acquire/release
atomics protect the payload; ctypes releases the GIL during native peer waits.
Each call validates sequence, layer, operation and payload shape. Reader
acknowledgements prevent fast ranks overwriting unread slots. A short bounded
spin is followed by sleeping polls; this creates no dedicated spinning worker
or global CPU-affinity changes. A missed deadline or callback exception
publishes an immutable abort status so peers cannot silently continue.

Planner and LRC logic remain on the existing Python path, with fixed-rank-order
score summation in the shared backend. Floating-point sums may differ in their
last bits from Gloo's reduction order for larger EP groups. READY still waits
for every rank, including zero-miss ranks, to finish all weight/quantization
work before residency and mapping are committed. One persistent watchdog
replaces the per-callback Timer threads for the shared backend. Python graph
callbacks, the transfer executor and final device stream synchronization remain;
this stage does not implement native graph input polling or device-value release.

CPU-only communication microbenchmark (no model or NPU execution):

```bash
python benchmarks/expert_offload/benchmark_shared_control.py \
  --iterations 300 --output /path/to/private/control_benchmark.json
```

Tests cover two/four processes, repeated slot reuse, scores, operation/layer
mismatch, delayed readers, timeout, abort propagation, bootstrap, GIL release,
watchdog rearming and manager commit ordering. The TP2 FULL shared-control
system test is `test_global_store_tp2_ep2_full_shared_control` in
`tests/e2e/pull_request/four_card/test_global_expert_store_tp_eager.py`.
Run hardware acceptance only with dedicated idle NPUs.

## Multi-node transports

`inter_node=disabled` rejects peers whose canonical regions cannot be mapped
locally. `tcp` is an explicit reference transport for functionality/debugging.
There is no automatic fallback from RDMA to TCP.

For `rdma`, build the optional adapter inside the container:

```bash
cd /data/l00955465/multi-card/vllm-ascend
cc -shared -fPIC -O2 -Wall -Wextra -Werror \
  vllm_ascend/expert_offload/global_store/rdma_verbs.c \
  -libverbs -o /data/l00955465/multi-card/runtime/libexpert_rdma.so
```

Then set the transport section on each node, using its own address and device:

```json
{
  "intra_node": "shared_memory",
  "inter_node": "rdma",
  "staging_buffer_experts": 4,
  "timeout_s": 60,
  "bind_address": "0.0.0.0",
  "advertise_address": "NODE_REACHABLE_IP",
  "rdma_library": "/data/l00955465/multi-card/runtime/libexpert_rdma.so",
  "rdma_device": "DEVICE_NAME",
  "rdma_port": 1,
  "rdma_gid_index": 0
}
```

The backend uses libibverbs RC QPs and RDMA READ, from the remote registered
canonical Host region into a local registered pinned staging buffer. TCP
carries authenticated connection/QP metadata only. NPU H2D remains Torch.
Rebuild the adapter after updating: the client now requires the
`expert_rdma_read_into` entry point to address distinct banks and field ranges.
An older library is rejected rather than silently overwriting bank zero.
The same homogeneous native endpoint ABI is required on peers. Successful
MR/QP creation is checked at bootstrap; reads fail explicitly on CQ error or
timeout. InfiniBand/RoCE devices exposed through libibverbs are required;
Ascend URMA-only fabrics are not implemented by this adapter. Device, port,
GID, reachable IP, memlock and actual Host RDMA connectivity need validation
on the deployment. There are no RDMA devices exposed in the development
container used for the CPU validation.

## Decode admission and graphs

The DP metadata exchange retains independent unpadded row counts. Admission
uses their sum times top-k; missing metadata uses a conservative DP-wide
bound. The padded execution shape must also fit a single-source bound and
the existing MC2 kernel limits. For top-k=6 and global capacity=128:

- `[8, 0, 0, 0]` contributes at most 48 expert routes and can select MC2.
- `[8, 8, 8, 8]` contributes at most 192 routes and selects eager ALLTOALL.

Admission bounds routes, not the observed number of distinct experts; it is
deliberately conservative. `bs=1` must specify whether it is global or per-DP.
Capture warmup uses one active DP source so a shape can be recorded without
pretending that every padded rank owns independent requests. Runtime masks
are refreshed before replay. ALLTOALL bypasses FULL graphs; graph cache keys
also include communication method, target/draft role and layout version.

Graph callbacks stage router IDs, compute a deterministic global placement,
and submit materialization to a separate NPU worker thread/load stream. That
thread waits only before reusing an unfinished DMA bank and refreshes derived scales.
Every rank participates in a READY/error exchange, including ranks with no
misses. Residency and log2phy commit only after success. Callback failure or
timeout terminates the service worker; it never continues with stale weights
and never switches collectives after entering MC2. A timeout does not mean
DMA cancellation: registered memory remains alive until QPs/process teardown.
Version 1 also synchronizes each graph forward before returning, keeping
forwards serialized while they share Host buffers. Overlap is deferred.

## Validation and acceptance

CPU regression command (does not initialize a model or allocate NPU memory):

```bash
cd /data/l00955465/multi-card/vllm-ascend
/data/l00955465/multi-card/.venv-dp4-ep4/bin/python -m pytest --noconftest -q \
  tests/ut/expert_offload/test_global_store.py \
  tests/ut/expert_offload/test_staging_pipeline.py \
  tests/ut/expert_offload/test_global_store_integration.py \
  tests/ut/expert_offload/test_multi_card_planner.py \
  tests/ut/expert_offload/test_h2d_transfer.py
```

Before production use, run on dedicated available NPUs: eager cross-owner
decode and prefill; compare all quant fields and logits against the existing
eager reference; capture and repeatedly replay MC2 with changing routes,
cache evictions, padding and different DP request distributions; verify
target/draft separation, overflow fallback and profiler evidence of MC2 in
the graph. Repeat on two nodes with actual RDMA, and test owner failure,
timeouts, partial transfers and teardown. Report PSS/unique mapped regions
rather than summing RSS to assess single-copy ownership. Performance tuning
and benchmarks come after those correctness checks.

Graceful close drains the materialization executor, retires source views,
closes all reader QPs, synchronizes peer shutdown, then closes source MRs,
mappings and owned job files. Abrupt worker failure releases memfd memory on
process exit; small visibility metadata can remain in that job directory.
Never remove another service's runtime directory or change server-wide settings.

## TP=4 eager bring-up

The separate `/data/l00955465/multi-card/run_tp4_ep4_eager.sh` launcher uses
NPU 0–3, TP=4, DP=1, EP=4, PP=1, DSpark=7 and the same single-copy Host store.
It keeps graph capture disabled. Its initial context/batch limits are 4096 and
can be overridden with the script's `MAX_MODEL_LEN`/`MAX_BATCHED_TOKENS` inputs.
The original DP launcher is unchanged. NDE=128 remains 32 complete experts
per rank per layer; Host ownership remains 64 of the model's 256 experts.

Startup validates the effective full-expert layout and matching EP/MC2 rank
order before allocating Host weights. ALLTOALL pads hidden states, logits and
hash-route input IDs to the same TP multiple, including batches larger than TP.
Combined TP>1/DP>1 remains explicitly rejected.

### Eager validation completed

The four-card `test_global_expert_store_tp_eager.py` test passed on NPU 0–3.
It constructs real TP=4/EP=4 HCCL groups and the production manager, loads
expert weights and scale fields through the single-copy Host store, and uses
the production offload entry point, placement cache, dispatch/combine and
TP prepare/finalize. The synthetic MoE has 64 experts, hidden size 1024,
intermediate size 128, topk=6 and 48 global device slots.

Eight forwards include 1/3/5/8-row MC2, 9/22-row ALLTOALL, and switching back
to MC2 with changing experts. Every rank matches an independent CPU reference;
maximum absolute error is below 1.56e-4. Cross-owner reads are asserted.
The shared canonical region totals approximately 48.3 MiB, with 1 MiB staging
per rank. Expert arithmetic uses BF16 Torch SwiGLU and explicit scale fields;
this test does not exercise the DeepSeek FP4 grouped-matmul kernels, DSpark
proposal generation, full-model serving or graph capture.

The CPU offload regression selection also passes (92 tests). The separate
eager launcher's real CLI/engine configuration validation passes with TP=4,
DP=1, graphs disabled and DSpark=7, without constructing an engine.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /data/l00955465/multi-card/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V2_MODEL_RUNNER=0 \
  /data/l00955465/multi-card/.venv-dp4-ep4/bin/python -m pytest \
  --noconftest -s -q \
  tests/e2e/pull_request/four_card/test_global_expert_store_tp_eager.py -k tp4_ep4_eager
```

The shared-store path uses an inert diagnostic trace: it neither loads the
legacy private prefetch diagnostic library nor creates its trace directory.


## TP>1 FULL decode bring-up

`/data/l00955465/multi-card/run_tp4_ep4_graph.sh` uses NPU 0–3, TP=4, DP=1,
EP=4, DSpark=7, NDE=128 (32 slots per rank), context/batch limit 4096 and
`FULL_DECODE_ONLY` with capture sizes `[8,16]`. It enables `graph.mc2_decode`
and omits `--enforce-eager`. Its real CLI/engine configuration validation
passes without starting an engine. Launch inside `cpu_dspark_lcf`:

```bash
bash /data/l00955465/multi-card/run_tp4_ep4_graph.sh
```

DP=1 now publishes a CPU tensor with the padded execution size and the
independent valid-row count, without a DP collective. Dummy/speculative
forward contexts also resolve their own valid rows before admission and mask
updates, so inherited target metadata cannot make draft padding active.
Graph keys remain independent of valid-row count; the stable MC2 mask and
router staging are updated on every replay. Admission still conservatively
requires the padded shape times topk to fit the slots. Overflow/prefill uses
eager ALLTOALL; returning to MC2 reuses its captured graph.

### FULL hardware validation

At the user's direction, the new FULL test ran on physical NPU 6–7 with
TP=2/EP=2 rather than using the occupied NPU 0–3. The implementation is generic
for TP>1, but this run does not establish TP=4 FULL graph acceptance.
The synthetic configuration has 64 experts, hidden size 1024, intermediate
size 128, topk=6, 48 global device slots, one canonical Host copy and 1 MiB
staging per rank. Expert arithmetic is fixed-shape BF16 Torch SwiGLU with
explicit scale fields; production paging, MC2 dispatch/combine, TP token
slicing/gathering and the ACLGraphWrapper are exercised unchanged.

Thirteen forwards pass an independent CPU reference, with maximum absolute
error below 2.05e-4. Two graphs (4/8 execution rows) are captured and reused
through 11 MC2 callbacks, with valid-row counts 0/1/2/3/5/7/8 and changing
experts. The test asserts graph object identity and that Python model forward
is not invoked on replay. The 9/22-row steps use eager ALLTOALL, followed by
successful reuse of existing MC2 graphs. Cross-owner Host reads are asserted.
Capturing alone does not execute the graph; accuracy is checked after replay.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /data/l00955465/multi-card/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=6,7 VLLM_USE_V2_MODEL_RUNNER=0 \
  /data/l00955465/multi-card/.venv-dp4-ep4/bin/python -m pytest \
  --noconftest -s -q \
  tests/e2e/pull_request/four_card/test_global_expert_store_tp_eager.py -k tp2
```

The selected offload CPU suite passes 110 cases. Additional context, graph
wrapper and prepare/finalize tests have 91 passes and 14 pre-existing failures
caused by removed `enable_sp`/`AscendDeviceType` mock targets; the same failures
are reproduced with the pre-change forward-context source loaded in memory.
This validation does not cover DeepSeek FP4 kernels, attention/KV graph
updates, DSpark proposal generation or full-model serving.

## Ping-pong staging validation (2026-11-25)

The earlier bring-up results above predate expert-count budgeting and retain
those runs' original staging sizes. The new default is four experts per rank.

The selected CPU regressions pass 117 cases, including delayed-DMA overwrite
checks, bank reuse across calls, failure cleanup, odd budgets, physical storage
padding and RDMA destination offsets. Three single-NPU stress cases pass with
budgets 2/3/4, raw weights, strided quant fields and dtype conversion. The RDMA
adapter compiles with `-Wall -Wextra -Werror`; real RDMA hardware is untested.

A 20-iteration single-NPU microbenchmark with six 12.75 MiB experts per batch
and 51 MiB staging measures 6.80 ms median for the previous implementation
and 5.80 ms for ping-pong staging. Full stream synchronizations drop from 25
to one per batch; bank completion events still protect reuse. This measures
materialization only and is not a full-model throughput result. Reproduce on
an idle NPU after sourcing the container's CANN environment:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
  /data/l00955465/multi-card/.venv-dp4-ep4/bin/python \
  benchmarks/expert_offload/benchmark_staging_pipeline.py \
  --baseline-source /path/to/previous/global_store/staging.py
```

The two-NPU cross-owner FULL replay test passes after fixing its report
subscription lifecycle. The production TP2/EP2 FULL test also passes: 13
forwards, two captured graphs, 11 graph callbacks per rank, changing routes,
zero valid rows, eager ALLTOALL fallback and graph reuse. Every rank matches
an independent CPU reference, with maximum absolute error below 2.05e-4.
Initial failures were in callback report teardown; test fixtures now stop
report processing before captured graph callback owners are destroyed.
These synthetic tests do not cover DeepSeek FP4 grouped matmul or real serving.
Logs and the microbenchmark JSON are retained in
`/data/l00955465/multi-card/runtime/staging_validation`.
The full formatting hook could not initialize its Ruff environment because
the configured GitHub proxy returned HTTP 407.

## Native small-batch staging (2026-11-25)

The opt-in native backend starts persistent CPU packers and reuses the existing
materialization executor as its single submission thread. Canonical source
addresses and physical layouts are registered at startup. Each layer passes
descriptor IDs to a GIL-free C++ pipeline; the first completed chunk can be
submitted while later chunks are still packing. Destinations are registered
on first use and cached by address/layout, including fresh views of prefill slots.

```json
"transport": {
  "control_backend": "shared_memory",
  "staging_backend": "native",
  "staging_buffer_experts": 48,
  "staging_submit_experts": 2,
  "staging_pack_threads": 2,
  "timeout_s": 60
}
```

`staging_buffer_experts` remains the total pinned capacity **per rank**, sized
using the largest physical expert including quant attributes and aligned
padding. A budget of 48 and submission size 2 creates 24 disjoint chunks.
An odd remainder creates a smaller final chunk. `staging_submit_experts` must
be positive and smaller than the budget; use 1 with a budget of 2. Worker
count is an integer from 1 to 64. The default backend remains `torch`; the
DP2 launcher explicitly selects `native`. `inter_node=disabled` and locally
mapped canonical owners are required; TCP/RDMA continue using `torch`.

Chunk lifecycle is FREE → FILLING → READY → IN_FLIGHT → FREE. FIFO jobs favor
earlier chunks; event-protected reuse applies across both layer calls and
single-expert prefill calls. Backpressure bounds all queued work to the ring.
Weights use physical byte copies, preserving NZ storage. Quant attributes
use ATen strided views, reshape and typed `copy_`, retaining any temporary
allocation until completion. The native submitter drains torch-npu's host
launch queue before mixing direct ACL copies with queued tensor operations,
and before recording each completion event on the same load stream.

Derived scale refresh, final load-stream synchronization, cross-rank READY,
and placement/residency commit retain their existing order. This change does
not move the planner, input polling or device completion notification into C++.
Exceptions poison the native pipeline and propagate through the existing
callback abort path; there is no automatic retry or backend switch in flight.

The C++ extension is built once by rank 0 in the private job directory using
the active interpreter's Torch ABI, Torch-NPU and CANN headers/libraries.
All ranks check staging settings and startup failures. Compilation uses the
startup timeout; subsequent materialization uses the transport timeout.
The build needs a C++17 compiler and development headers. It does not install
packages, modify shared compiler caches, bind host CPUs, or change system settings.

To revert, set only `staging_backend` to `torch` and restart the service.
The existing two-bank staging and shared-memory control remain available;
`control_backend` is an independent choice. A running process keeps its loaded
implementation until restart.

`NativeStagingTransport.last_stats` exposes pack-worker CPU nanoseconds
(summed across workers), ready-wait time, submission CPU time, time to first
submission, total pipeline enqueue time and chunk count. These exclude the
Python planner/descriptor setup and final DMA drain; they are not service TPOT.

CPU regression tests exercise native workers with delayed asynchronous copies,
odd budgets, bounded reuse, quant strides/dtype conversion, close/error paths,
and GIL release. `benchmarks/expert_offload/benchmark_native_staging.py` compares
whole-layer packing with chunk sizes 1/2/4 and worker counts 1/2/4 using a CPU
DMA simulator. It measures CPU behavior, not PCIe or model throughput.
Hardware acceptance tests are provided in
`tests/e2e/pull_request/one_card/test_native_staging.py` (NZ plus quant fields)
and `test_global_store_tp2_ep2_full_native_staging` (TP/EP FULL graph reuse).
They have not been run for this backend because all NPUs were occupied.
