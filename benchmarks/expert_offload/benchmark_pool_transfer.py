"""Microbenchmark static pool H2D in eager and GLOBAL graph modes.

Run only on an idle ASCEND_RT_VISIBLE_DEVICES with a supported pinned allocator.
This measures transport, not model latency or throughput.
"""
import argparse
import json
from time import perf_counter
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.prefill_transfer import PoolCopy, PoolTransferPlan, PoolTransferScheduler


def benchmark(nbytes, repeats):
    source = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True).fill_(37)
    destination = torch.empty(nbytes, dtype=torch.uint8, device="npu")
    plan = PoolTransferPlan((PoolCopy.prepare(source, destination, nbytes, "benchmark"),))
    scheduler = PoolTransferScheduler(SimpleNamespace(prepare=lambda *args: plan))
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        scheduler.submit(0, 0)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
        scheduler.submit(0, 0)
    rows = []
    for name, submit in (("eager", lambda: scheduler.submit(0, 0)), ("graph", graph.replay)):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        before = perf_counter()
        with torch.npu.stream(stream):
            start.record()
            for _ in range(repeats):
                submit()
            end.record()
        end.synchronize()
        milliseconds = start.elapsed_time(end) / repeats
        rows.append({
            "mode": name, "bytes": nbytes, "repeats": repeats, "device_ms": milliseconds,
            "GiB_per_second": nbytes / (1024 ** 3) / (milliseconds / 1000),
            "wall_ms_per_copy": (perf_counter() - before) * 1000 / repeats,
        })
    assert torch.all(destination.cpu() == 37)
    del graph
    scheduler.close()
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--mib", type=int, nargs="+", default=[1, 12, 64])
    args = parser.parse_args()
    if args.repeats <= 0 or any(size <= 0 for size in args.mib):
        parser.error("sizes and repeats must be positive")
    torch.npu.set_device(0)
    print(json.dumps([row for size in args.mib for row in benchmark(size * 1024 ** 2, args.repeats)], indent=2))
