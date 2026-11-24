# Global Host expert store, version 1

This is an opt-in functional implementation. NPU inference, MC2 FULL graph
capture/replay, accuracy and real RDMA transfers still require hardware
acceptance. CPU tests and compiling the verbs adapter do not establish those
properties.

## Supported scope

- v1 runner, TP=PP=1, static EP, identical quantized expert layouts on peers.
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
    "staging_buffer_mb": 64,
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

Each rank has one pinned staging allocation, default 64 MiB. Raw weight fields
larger than this are copied in chunks. A quantization field must fit staging;
its logical shape/stride/dtype are preserved, including strided destinations.
There is no private full Host replica per reader. Device caches, bounded
staging, checkpoint mappings and temporary startup conversion buffers are
separate from the canonical store. Startup temporarily holds old owner
buffers and the new shared allocation; old buffers and their process-local
pinned allocator cache are released after publication. This is not a zero-copy
startup implementation. Published layers reject checkpoint reloading and
repeated finalization cannot convert their bytes a second time.

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
  "staging_buffer_mb": 64,
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
thread waits after each staging-buffer reuse and refreshes derived scales.
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
