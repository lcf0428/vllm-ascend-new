# SPDX-License-Identifier: Apache-2.0
"""CPU-only pipeline benchmark; does not measure NPU bandwidth or service TPOT.

Compare first submission and CPU completion against whole-layer packing using
the same native code, canonical mmap, destination buffers and fake DMA delay.
Run with --experts 21 --expert-mib 6 --budget 48 --iterations 20.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_ascend.expert_offload.global_store.native_staging import (
    NativeStagingTransport, build_native_staging, load_native_staging, staging_chunk_layout,
)
from vllm_ascend.expert_offload.global_store.shared_store import SharedRegion
from vllm_ascend.expert_offload.global_store.types import ExpertKey, Source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=21)
    parser.add_argument("--expert-mib", type=int, default=6)
    parser.add_argument("--budget", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="native-staging-bench-") as temporary:
        directory = Path(temporary)
        module = load_native_staging(build_native_staging(directory / "build", cpu_test=True))
        size = args.expert_mib * (1 << 20) // 2
        experts = [(ExpertKey("benchmark", "target", "layer", eid), {
            "w13": torch.full((size,), eid, dtype=torch.uint8),
            "w2": torch.full((size,), eid + 1, dtype=torch.uint8),
        }) for eid in range(args.experts)]
        region, bundles = SharedRegion.publish(directory / "owner.bin", 0, experts)
        store = SimpleNamespace(regions={0: region},
                                directory=SimpleNamespace(bundles={b.key: b for b in bundles}))
        tasks, outputs = [], []
        for bundle in bundles:
            for field in bundle.fields:
                output = torch.empty(field.nbytes, dtype=torch.uint8)
                outputs.append(output)
                tasks.append(SimpleNamespace(source=Source(bundle, field), destination=output.untyped_storage(),
                                              nbytes=field.nbytes))
        result = {"scope": "CPU-only, fake asynchronous DMA, excludes real NPU/PCIe and Python manager/planner",
                  "parameters": vars(args), "cases": []}
        cases = [("whole_layer", args.budget - 1, 1)]
        cases += [("pipeline", submit, workers) for submit in (1, 2, 4) for workers in (1, 2, 4)]
        for name, submit, workers in cases:
            if submit >= args.budget:
                continue
            capacities, counts = staging_chunk_layout(experts, args.budget, submit)
            if name == "whole_layer":
                # Same total bytes, but deliberately wait for the whole layer.
                capacities, counts = (sum(capacities),), [args.budget]
            store.config = SimpleNamespace(transport={"staging_pack_threads": workers, "timeout_s": 60})
            transport = NativeStagingTransport(store, torch.empty(sum(capacities), dtype=torch.uint8),
                                               SimpleNamespace(close=lambda: None, synchronize=lambda s: None),
                                               capacities, counts, module, cpu_test=True, delay_us=0)
            samples = []
            try:
                for iteration in range(args.iterations + 3):
                    start = time.perf_counter_ns()
                    transport.copy_batch(tasks)
                    transport.synchronize(None)
                    completion = time.perf_counter_ns() - start
                    if iteration >= 3:
                        samples.append(dict(transport.last_stats, completion_ns=completion))
                for task, output in zip(tasks, outputs):
                    expected = task.source.bundle.key.expert + (task.source.field.name == "w2")
                    assert bool(torch.all(output == expected))
                result["cases"].append({"name": name, "submit_experts": args.budget if name == "whole_layer" else submit,
                    "pack_threads": workers,
                    **{key: statistics.median(s[key] for s in samples) for key in samples[0]}})
            finally:
                transport.close()
        region.close()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
