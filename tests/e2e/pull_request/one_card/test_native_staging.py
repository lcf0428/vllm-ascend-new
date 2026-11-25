# SPDX-License-Identifier: Apache-2.0
"""Requires one idle NPU: raw NZ bytes plus strided quant copies on a load stream."""

from types import SimpleNamespace

import torch
import torch_npu

from vllm_ascend.expert_offload.global_store.native_staging import (
    NativeStagingTransport, build_native_staging, load_native_staging, staging_chunk_layout,
)
from vllm_ascend.expert_offload.global_store.shared_store import SharedRegion
from vllm_ascend.expert_offload.global_store.staging import StagingTransport, staging_buffer_sizes
from vllm_ascend.expert_offload.global_store.types import ExpertKey, Source
from vllm_ascend.expert_offload.h2d_transfer import H2DCopyTask, TorchCopyH2DTransport


def test_native_staging_nz_and_strided_quant_match_torch(tmp_path):
    torch.npu.set_device(0)
    model = torch_npu.npu_format_cast(torch.arange(2048, device="npu", dtype=torch.float32)
                                     .reshape(32, 64).bfloat16(), 29)
    physical = torch.empty(model.untyped_storage().nbytes(), dtype=torch.uint8, pin_memory=True)
    physical.untyped_storage().copy_(model.untyped_storage())
    torch.npu.synchronize()
    experts = [(ExpertKey("test", "target", "layer", i), {
        "w13": physical.clone(), "w2": physical.clone(),
        "scale": (torch.arange(12, dtype=torch.float32).reshape(3, 4) + i).t(),
    }) for i in range(7)]
    region, bundles = SharedRegion.publish(tmp_path / "owner.bin", 0, experts)
    store = SimpleNamespace(config=SimpleNamespace(transport={"staging_pack_threads": 2, "timeout_s": 60}),
                            regions={0: region}, directory=SimpleNamespace(bundles={b.key: b for b in bundles}))
    store.read_into = lambda source, offset, target, size: region.read_into(source.field.offset + offset, target, size)
    capacities, counts = staging_chunk_layout(experts, 3, 1)
    native = NativeStagingTransport(store, torch.empty(sum(capacities), dtype=torch.uint8, pin_memory=True),
                                    TorchCopyH2DTransport(), capacities, counts,
                                    load_native_staging(build_native_staging(tmp_path / "build")))
    banks = staging_buffer_sizes(experts, 3)
    baseline = StagingTransport(store, torch.empty(sum(banks), dtype=torch.uint8, pin_memory=True),
                                TorchCopyH2DTransport(), banks)
    stream = torch.npu.Stream()
    outputs = []
    try:
        for transport in (baseline, native):
            tasks, values = [], []
            for bundle in bundles:
                fields = {"w13": torch_npu.npu_format_cast(torch.zeros_like(model, memory_format=torch.contiguous_format), 29),
                          "w2": torch_npu.npu_format_cast(torch.zeros_like(model, memory_format=torch.contiguous_format), 29),
                          "scale": torch.empty(6, 2, device="npu", dtype=torch.float16).t()}
                values.append(fields)
                for field in bundle.fields:
                    destination = fields[field.name]
                    tasks.append(H2DCopyTask(Source(bundle, field), destination if field.name == "scale"
                                             else destination.untyped_storage(), field.nbytes))
            # Producer stream initialized destination tensors above.
            torch.npu.synchronize()
            with torch.npu.stream(stream):
                for _ in range(5):
                    transport.copy_batch(tasks)
                    transport.synchronize(stream)
            outputs.append(values)
        for torch_fields, native_fields in zip(*outputs):
            for name in torch_fields:
                torch.testing.assert_close(native_fields[name], torch_fields[name], rtol=0, atol=0)
            torch.testing.assert_close(native_fields["w13"], model, rtol=0, atol=0)
    finally:
        native.close()
        baseline.close()
        region.close()
