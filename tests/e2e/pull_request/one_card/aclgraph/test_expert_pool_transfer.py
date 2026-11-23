"""Run with an idle visible NPU and a supported pinned-memory allocator."""
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.prefill_transfer import (
    PoolConversion,
    PoolCopy,
    PoolTransferPlan,
    PoolTransferScheduler,
)


def test_global_capture_pool_reload_and_cross_graph_slot_reuse():
    torch.npu.set_device(0)
    sources = [torch.empty(4096, dtype=torch.uint8, pin_memory=True) for _ in range(3)]
    pool = [torch.empty(4096, dtype=torch.uint8, device="npu") for _ in range(2)]
    scale = torch.empty(16, dtype=torch.float16, pin_memory=True).fill_(0.5)
    device_scale = torch.empty(16, dtype=torch.float16, device="npu")
    fp32 = torch.empty(16, dtype=torch.float32, device="npu")
    plans = {}
    for layer in range(3):
        sources[layer].fill_(layer + 1)
        copies = [PoolCopy.prepare(sources[layer], pool[layer % 2], 4096, f"L{layer}")]
        conversions = ()
        if layer == 0:
            copies.append(PoolCopy.prepare(scale, device_scale, scale.numel() * scale.element_size(), "scale"))
            conversions = (PoolConversion(device_scale, fp32),)
        plans[(layer, layer % 2)] = PoolTransferPlan(tuple(copies), conversions)
    scheduler = PoolTransferScheduler(SimpleNamespace(prepare=lambda layer, slot: plans[(layer, slot)]))

    def forward(n):
        outputs = []
        for layer in range(3):
            scheduler.submit(layer, layer % 2)
            outputs.append(pool[layer % 2][:n].clone())
        return outputs, fp32.clone()

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        forward(64)
    stream.synchronize()
    graphs, results = {}, {}
    for size in (64, 128):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
            results[size] = forward(size)
        graphs[size] = graph
    for step, size in enumerate((64, 128, 64, 128)):
        for layer, source in enumerate(sources):
            source.fill_(10 * step + layer + 1)
        scale.fill_(step + 0.5)
        graphs[size].replay()
        stream.synchronize()
        values, converted = results[size]
        for layer, value in enumerate(values):
            assert torch.all(value.cpu() == 10 * step + layer + 1)
        assert torch.all(converted.cpu() == step + 0.5)
    graphs.clear()
    del graph
    scheduler.close()


def _adapter_pool_case(dtype):
    count = 2
    cpu_weights = {
        name: [
            [torch.full((size,), 20 * layer + eid + 1, dtype=torch.uint8, pin_memory=True)
             for eid in range(count)]
            for layer in range(3)
        ]
        for name, size in (("w13", 8), ("w2", 4))
    }
    scales = [
        [torch.arange(24, dtype=dtype).reshape(6, 4).add(layer).pin_memory() for _ in range(count)]
        for layer in range(3)
    ]
    scale_pools = [
        torch.empty_strided((count, 3, 2, 4), (24, 4, 12, 1), dtype=dtype, device="npu")
        for _ in range(2)
    ]
    manager = SimpleNamespace(
        native_mxfp4_enabled=False, mc_shard_size=count, ep_rank=1,
        w13_expert_size_bytes=8, w2_expert_size_bytes=4,
        _prefill_w13=[torch.empty(count, 8, dtype=torch.uint8, device="npu") for _ in range(2)],
        _prefill_w2=[torch.empty(count, 4, dtype=torch.uint8, device="npu") for _ in range(2)],
        scale_cpu_buffers={"w13_weight_scale": scales}, offset_cpu_buffers={}, scale_bias_cpu_buffers={},
        _prefill_w13_scale=scale_pools, _prefill_w2_scale=[], _prefill_w13_offset=[],
        _prefill_w2_offset=[], _prefill_w13_scale_bias=[], _prefill_w2_scale_bias=[],
        _prefill_w13_scale_fp32=(
            [torch.empty(count, 3, 2, 4, dtype=torch.float32, device="npu") for _ in range(2)]
            if dtype == torch.float16 else []
        ),
    )
    manager._shard_local = lambda eid: eid - count if count <= eid < 2 * count else None
    manager._expert_src_storage = lambda layer, eid, field: cpu_weights[field][layer][eid-count].untyped_storage()
    from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager

    manager._prefill_initialized = True
    manager.num_device_layers = 2
    manager.moe_layers = [SimpleNamespace() for _ in range(3)]
    manager.offload_config = SimpleNamespace(h2d_backend="torch")

    def forward():
        result = []
        for layer in range(3):
            slot = layer % 2
            ExpertOffloadManager._prefill_load_layer_shard(manager, layer)
            result.append((
                manager._prefill_w13[slot].clone(),
                manager._prefill_w13_scale[slot].clone(),
                manager._prefill_w13_scale_fp32[slot].clone() if dtype == torch.float16 else None,
            ))
        return result

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        eager = forward()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
        captured = forward()
    for _ in range(3):
        # Deliberately destroy the last layer's pool contents. Replay must
        # reload all layers, including L0 and L2 which share slot zero.
        with torch.npu.stream(stream):
            for pool in manager._prefill_w13:
                pool.zero_()
        graph.replay()
        stream.synchronize()
        for layer, (weight, scale, fp32) in enumerate(captured):
            assert torch.equal(weight.cpu(), eager[layer][0].cpu())
            assert torch.equal(scale.cpu(), torch.stack(scales[layer]).reshape(count, 3, 2, 4))
            if fp32 is not None:
                assert torch.equal(fp32.cpu(), scale.cpu().float())
    del graph
    manager._pool_transfer_scheduler.close()


def test_adapter_uint8_strided_quant_pool_global_capture():
    torch.npu.set_device(0)
    _adapter_pool_case(torch.uint8)


def test_adapter_fp16_quant_pool_and_fp32_cache_global_capture():
    torch.npu.set_device(0)
    _adapter_pool_case(torch.float16)
