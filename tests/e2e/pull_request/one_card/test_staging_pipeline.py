# SPDX-License-Identifier: Apache-2.0
"""Run only on an idle, dedicated NPU; stress bank reuse and quant layouts."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.global_store.directory import Directory
from vllm_ascend.expert_offload.global_store.shared_store import SharedRegion
from vllm_ascend.expert_offload.global_store.staging import StagingTransport, staging_buffer_sizes
from vllm_ascend.expert_offload.global_store.types import ExpertKey
from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport


@pytest.mark.parametrize("expert_budget", [2, 3, 4])
def test_staging_pipeline_byte_exact_reuse_and_quant_conversion(tmp_path, expert_budget):
    torch.npu.set_device(0)
    stream = torch.npu.Stream()
    experts = []
    for eid in range(7):
        fields = {"w13": torch.full((256 * 1024,), eid + 1, dtype=torch.uint8),
                  "w2": torch.full((128 * 1024,), eid + 10, dtype=torch.uint8),
                  "scale": (torch.arange(2048, dtype=torch.float32).reshape(64, 32) + eid).t(),
                  "offset": torch.arange(17, dtype=torch.int64) + eid}
        experts.append((ExpertKey("staging-test", "target", "layer", eid), fields))
    region, bundles = SharedRegion.publish(tmp_path / "weights", 0, experts, backing="memfd")
    directory = Directory([{"rank": 0, "size": region.size, "bundles": bundles}],
                          [key for key, _ in experts])

    class Store:
        def read_into(self, source, offset, target, size):
            region.read_into(source.field.offset + offset, target, size)

    sizes = staging_buffer_sizes(experts, expert_budget)
    staging = torch.empty(sum(sizes), dtype=torch.uint8, pin_memory=True)
    transport = StagingTransport(Store(), staging, TorchCopyH2DTransport(), sizes)
    destinations, expected, batches = [], [], []
    for key, fields in experts:
        tasks = []
        for name, tensor in fields.items():
            if name.startswith("w"):
                destination = torch.empty_like(tensor, device="npu")
                target = destination.untyped_storage()
            else:
                # Exercise noncontiguous destinations and logical dtype conversion.
                dtype = torch.float16 if name == "scale" else tensor.dtype
                destination = torch.empty_strided(tensor.shape, tensor.stride(), dtype=dtype, device="npu")
                target = destination
            tasks.append(SimpleNamespace(source=directory.source(key, name), destination=target,
                                         nbytes=tensor.untyped_storage().nbytes(), name=name))
            destinations.append(destination)
            expected.append(tensor.to(destination.dtype))
        batches.append(tasks)
    try:
        for step in range(8):
            with torch.npu.stream(stream):
                if step % 2:
                    for tasks in reversed(batches):
                        transport.copy_batch(tasks)
                else:
                    transport.copy_batch([item for tasks in batches for item in tasks])
                transport.synchronize(stream)
            for destination, reference in zip(destinations, expected):
                assert torch.equal(destination.cpu(), reference)
        # Leave DMA pending and ensure close drains it before releasing sources.
        with torch.npu.stream(stream):
            transport.copy_batch([item for tasks in batches for item in tasks])
        transport.close()
        for destination, reference in zip(destinations, expected):
            assert torch.equal(destination.cpu(), reference)
    finally:
        transport.close()
        region.close()
