# SPDX-License-Identifier: Apache-2.0
"""Compare bounded ping-pong staging to the previous serial implementation.

Run on an idle NPU. --baseline-source points to an unmodified old staging.py;
no installed modules or running services are replaced by the benchmark.
"""

import argparse
import importlib.util
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.global_store.directory import Directory
from vllm_ascend.expert_offload.global_store.shared_store import SharedRegion
from vllm_ascend.expert_offload.global_store.staging import StagingTransport, staging_buffer_sizes
from vllm_ascend.expert_offload.global_store.types import ExpertKey
from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport


class CountedDelegate(TorchCopyH2DTransport):
    def __init__(self):
        self.sync_calls = 0

    def synchronize(self, stream):
        self.sync_calls += 1
        super().synchronize(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--staging-experts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.experts <= 0 or args.iterations <= 0:
        parser.error("experts and iterations must be positive")
    spec = importlib.util.spec_from_file_location(
        "vllm_ascend.expert_offload.global_store._serial_benchmark", args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    stream = torch.npu.Stream()
    mib = 1 << 20
    experts = [(ExpertKey("benchmark", "target", "layer", eid), {
        "w13": torch.full((8 * mib,), eid + 1, dtype=torch.uint8),
        "w2": torch.full((4 * mib,), eid + 11, dtype=torch.uint8),
        "scale13": torch.full((128, 1024), eid + 1, dtype=torch.float32).t(),
        "scale2": torch.full((128, 512), eid + 2, dtype=torch.float32).t(),
    }) for eid in range(args.experts)]
    sizes = staging_buffer_sizes(experts, args.staging_experts)
    with tempfile.TemporaryDirectory(prefix="expert-staging-benchmark-") as directory:
        region, bundles = SharedRegion.publish(Path(directory) / "weights", 0, experts, backing="memfd")
        index = Directory([{"rank": 0, "size": region.size, "bundles": bundles}],
                          [key for key, _ in experts])

        class Store:
            def read_into(self, source, offset, target, size):
                region.read_into(source.field.offset + offset, target, size)

        destinations, references, tasks = [], [], []
        for key, fields in experts:
            for name, tensor in fields.items():
                destination = torch.empty_strided(tensor.shape, tensor.stride(),
                                                  dtype=tensor.dtype, device="npu")
                target = destination.untyped_storage() if name.startswith("w") else destination
                tasks.append(SimpleNamespace(source=index.source(key, name), destination=target,
                                             nbytes=tensor.untyped_storage().nbytes(), name=name))
                destinations.append(destination)
                references.append(tensor)
        results = {"experts_per_batch": args.experts, "staging_buffer_experts": args.staging_experts,
                   "bank_bytes": sizes, "payload_bytes": sum(t.nbytes for t in tasks),
                   "iterations": args.iterations, "runs": {}}
        try:
            for name, cls in (("serial", baseline.StagingTransport), ("pipeline", StagingTransport)):
                staging = torch.empty(sum(sizes), dtype=torch.uint8, pin_memory=True)
                delegate = CountedDelegate()
                transport = cls(Store(), staging, delegate)

                def run():
                    with torch.npu.stream(stream):
                        transport.copy_batch(tasks)
                        transport.synchronize(stream)

                try:
                    for _ in range(3):
                        run()
                    delegate.sync_calls = 0
                    times = []
                    for _ in range(args.iterations):
                        start = time.perf_counter()
                        run()
                        times.append((time.perf_counter() - start) * 1000)
                    for destination, reference in zip(destinations, references):
                        assert torch.equal(destination.cpu(), reference)
                    results["runs"][name] = {
                        "median_ms": statistics.median(times), "min_ms": min(times),
                        "stream_syncs_per_batch": delegate.sync_calls / args.iterations,
                        "byte_exact": True, "samples_ms": times}
                finally:
                    transport.close()
                del transport, staging
            results["speedup"] = (results["runs"]["serial"]["median_ms"]
                                  / results["runs"]["pipeline"]["median_ms"])
            output = json.dumps(results, indent=2)
            print(output)
            if args.output:
                args.output.write_text(output + "\n")
        finally:
            region.close()


if __name__ == "__main__":
    main()
