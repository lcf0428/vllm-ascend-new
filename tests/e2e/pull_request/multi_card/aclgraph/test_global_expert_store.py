# SPDX-License-Identifier: Apache-2.0
"""Two dedicated idle NPUs: cross-owner materialization during FULL replay.

This small test validates the callback/staging primitive. Full-model MC2
dispatch/combine accuracy is a separate acceptance test. Do not run on NPUs
allocated to other services.
"""

import datetime
import gc
import os
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from vllm_ascend.expert_offload.global_store.config import GlobalStoreConfig
from vllm_ascend.expert_offload.global_store.executor import TransferExecutor
from vllm_ascend.expert_offload.global_store.staging import StagingTransport
from vllm_ascend.expert_offload.global_store.store import GlobalExpertStore
from vllm_ascend.expert_offload.global_store.types import ExpertKey
from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport


def _worker(rank, port, runtime_dir):
    torch.npu.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60))

    def gather(value):
        values = [None, None]
        dist.all_gather_object(values, value)
        return values

    config = GlobalStoreConfig.parse({
        "expert_offload": True, "enable_multi_card": True,
        "host_storage": {"mode": "global_sharded", "runtime_dir": runtime_dir},
    })
    staging = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    size = 16384
    key = ExpertKey("tiny-model", "target", "layer.0", rank)
    private = {"w13": torch.full((size,), rank + 1, dtype=torch.uint8),
               "w2": torch.full((size,), rank + 3, dtype=torch.uint8)}
    store = GlobalExpertStore(config, rank, 2, gather, "tiny-model", ["layer.0"],
                              [(key, private)], staging)
    private.clear()
    transport = StagingTransport(store, staging, TorchCopyH2DTransport())
    executor = TransferExecutor(rank, 30)
    load_stream = torch.npu.Stream()
    compute_stream = torch.npu.Stream()
    torch_npu.npu._subscribe_report(compute_stream)
    ids = torch.zeros(1, dtype=torch.int32, device="npu")
    ids_h = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    destination = torch.zeros(size, dtype=torch.uint8, device="npu")

    def materialize(eid):
        with torch.npu.stream(load_stream):
            task = SimpleNamespace(source=store.source(0, eid, "w13"),
                                   destination=destination.untyped_storage(), nbytes=size, name="w13")
            transport.copy_batch([task])
            transport.synchronize(load_stream)

    def callback(_):
        failure = None
        try:
            executor.run(materialize, int(ids_h[0]))
        except Exception as exc:
            failure = str(exc)
        failures = gather(failure)
        if any(failures):
            # A C callback cannot propagate exceptions to the engine.
            os._exit(1)

    def forward():
        ids_h.copy_(ids, non_blocking=True)
        torch_npu.npu._launch_host_func(compute_stream, callback, None)
        return destination.float().sum()

    with torch.npu.stream(compute_stream):
        forward()
    compute_stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=compute_stream, capture_error_mode="global"):
        output = forward()
    for step in range(6):
        expert = (rank + step) % 2
        ids.fill_(expert)
        torch.npu.current_stream().synchronize()
        graph.replay()
        compute_stream.synchronize()
        assert output.cpu().item() == size * (expert + 1)
        assert torch.equal(destination.cpu(), torch.full((size,), expert + 1, dtype=torch.uint8))
    torch.npu.synchronize()
    # Stop the callback report thread before destroying its captured graph.
    torch_npu.npu._unsubscribe_report(compute_stream)
    del graph, output
    gc.collect()
    executor.close()
    transport.close()
    store.close()
    dist.destroy_process_group()


def test_global_store_cross_owner_full_graph_replay():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="global-expert-e2e-") as runtime_dir:
        mp.spawn(_worker, args=(port, str(Path(runtime_dir).resolve())), nprocs=2, join=True)
