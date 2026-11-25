# SPDX-License-Identifier: Apache-2.0
"""CPU-only Gloo vs shared-memory statistics+READY, without model execution."""

import argparse
import datetime
import json
import socket
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.expert_offload.global_store.control import SharedMemoryControl, build_control_library

WARMUP = 30


def _worker(rank, port, directory, iterations):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=60))
    library = build_control_library(directory) if rank == 0 else None
    dist.barrier()
    if library is None:
        library = next(Path(directory).glob("control-*.so"))
    control = None
    if rank == 0:
        control = SharedMemoryControl(Path(directory) / "control.bin", library, rank, 2, 256, 5, create=True)
    dist.barrier()
    if control is None:
        control = SharedMemoryControl(Path(directory) / "control.bin", library, rank, 2, 256, 5)
    results = {}
    for with_scores in (False, True):
        for backend in ("gloo", "shared_memory"):
            samples = []
            dist.barrier()
            for step in range(WARMUP + iterations):
                counts = torch.full((256,), rank + step, dtype=torch.int64)
                scores = counts.float() / 8 if with_scores else None
                start = time.perf_counter_ns()
                if backend == "gloo":
                    global_counts = counts.clone()
                    dist.all_reduce(global_counts)
                    if scores is not None:
                        global_scores = scores.clone()
                        dist.all_reduce(global_scores)
                    failures = [None, None]
                    dist.all_gather_object(failures, None)
                else:
                    global_counts, global_scores = control.statistics(step % 43, counts, scores)
                    assert not control.ready(step % 43)
                elapsed = (time.perf_counter_ns() - start) / 1000
                assert torch.equal(global_counts, torch.full((256,), 2 * step + 1, dtype=torch.int64))
                if scores is not None:
                    assert torch.equal(global_scores, global_counts.float() / 8)
                if step >= WARMUP:
                    samples.append(elapsed)
            ordered = sorted(samples)
            results[f"{backend}_{'counts_scores' if with_scores else 'counts'}"] = {
                "median_us": statistics.median(samples), "p90_us": ordered[int(0.9 * len(ordered))],
                "mean_us": statistics.mean(samples), "iterations": iterations,
            }
    dist.barrier()
    control.close()
    (Path(directory) / f"rank{rank}.json").write_text(json.dumps(results))
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="expert-control-bench-") as directory:
        mp.spawn(_worker, args=(port, directory, args.iterations), nprocs=2, join=True)
        result = {f"rank{rank}": json.loads((Path(directory) / f"rank{rank}.json").read_text()) for rank in range(2)}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
