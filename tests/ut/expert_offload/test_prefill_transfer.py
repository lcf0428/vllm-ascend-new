"""Static transfer scheduling and quantization layout contracts."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.expert_offload.prefill_transfer import (
    ExpertPoolAdapter,
    PoolConversion,
    PoolCopy,
    PoolTransferPlan,
    PoolTransferScheduler,
    byte_view,
)


def test_byte_view_preserves_packed_storage_and_offset():
    owner = torch.arange(12, dtype=torch.int32)
    view = byte_view(owner[3:7], 16)
    assert torch.equal(view, owner.view(torch.uint8)[12:28])
    view.zero_()
    assert owner[3:7].tolist() == [0, 0, 0, 0]
    assert owner[2] == 2 and owner[7] == 7


@pytest.mark.parametrize("size", [0, -1, 100])
def test_byte_view_rejects_out_of_bounds(size):
    with pytest.raises(ValueError):
        byte_view(torch.empty(4, dtype=torch.uint8), size)


def test_pointer_only_source_is_not_silently_captured():
    with pytest.raises(TypeError):
        byte_view(SimpleNamespace(data_ptr=lambda: 1234), 4)


def test_quant_layout_matches_legacy_logical_copy_without_touching_host_source():
    source = torch.arange(24, dtype=torch.uint8).reshape(6, 4)
    destination = torch.empty_strided((3, 2, 4), (4, 12, 1), dtype=torch.uint8)
    expected = torch.empty_strided(destination.shape, destination.stride(), dtype=source.dtype)
    expected.copy_(source.reshape(destination.shape))
    allocate = torch.empty_strided

    def cpu_allocate(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return allocate(*args, **kwargs)

    with patch("torch.empty_strided", side_effect=cpu_allocate):
        staged = ExpertPoolAdapter._quant_source(source, destination)
    assert staged.stride() == destination.stride()
    assert torch.equal(byte_view(staged, 24), byte_view(expected, 24))
    assert torch.equal(source.flatten(), torch.arange(24, dtype=torch.uint8))


def test_quant_adapter_rejects_unimplemented_conversion():
    with pytest.raises(ValueError, match="adapter"):
        ExpertPoolAdapter._quant_source(torch.ones(4), torch.ones(4, dtype=torch.int64))


def test_scheduler_reloads_reused_pool_on_every_submission_and_converts_scale():
    source = torch.tensor([3, 5], dtype=torch.uint8)
    destination = torch.zeros_like(source)
    scale = torch.tensor([0.5, 1.5], dtype=torch.float16)
    fp32 = torch.zeros(2)
    plan = PoolTransferPlan((PoolCopy(source, destination, "weight"),), (PoolConversion(scale, fp32),))
    adapter = SimpleNamespace(prepare=MagicMock(return_value=plan))
    scheduler = PoolTransferScheduler(adapter)
    stream = SimpleNamespace(npu_stream=123, synchronize=MagicMock())
    with patch("torch.npu.is_current_stream_capturing", return_value=False), patch(
        "torch.npu.current_stream", return_value=stream
    ):
        scheduler.submit(0, 0)
        assert torch.equal(destination, source)
        destination.zero_()
        source.fill_(9)
        scheduler.submit(0, 0)
        assert destination.tolist() == [9, 9]
        assert torch.equal(fp32, scale.float())
    adapter.prepare.assert_called_once_with(0, 0)
    scheduler.close()
    stream.synchronize.assert_called_once()
    assert not scheduler.plans


def test_capture_requires_prepared_plan():
    adapter = SimpleNamespace(prepare=MagicMock())
    scheduler = PoolTransferScheduler(adapter)
    with patch("torch.npu.is_current_stream_capturing", return_value=True):
        with pytest.raises(RuntimeError, match="warmup"):
            scheduler.submit(1, 0)
    adapter.prepare.assert_not_called()


def test_adapter_only_reads_rank_one_shard():
    manager = SimpleNamespace(
        native_mxfp4_enabled=False, mc_shard_size=2, ep_rank=1,
        w13_expert_size_bytes=8, w2_expert_size_bytes=4,
        _prefill_w13=[torch.empty(2, 8, dtype=torch.uint8)],
        _prefill_w2=[torch.empty(2, 4, dtype=torch.uint8)],
        scale_cpu_buffers={}, offset_cpu_buffers={}, scale_bias_cpu_buffers={},
        _prefill_w13_scale=[], _prefill_w2_scale=[], _prefill_w13_offset=[],
        _prefill_w2_offset=[], _prefill_w13_scale_bias=[], _prefill_w2_scale_bias=[],
        _prefill_w13_scale_fp32=[],
    )
    source = torch.ones(8, dtype=torch.uint8)
    manager._expert_src_storage = MagicMock(return_value=source)
    with patch.object(PoolCopy, "prepare", return_value=MagicMock()):
        plan = ExpertPoolAdapter(manager).prepare(0, 0)
    assert [call.args[1] for call in manager._expert_src_storage.call_args_list] == [2, 2, 3, 3]
    assert len(plan.copies) == 4
    assert not plan.conversions


def test_quant_pool_with_holes_requires_explicit_adapter():
    source = torch.ones(4)
    destination = torch.empty_strided((4,), (2,))
    with pytest.raises(ValueError, match="dense"):
        ExpertPoolAdapter._quant_source(source, destination)


def test_quant_method_can_supply_custom_fields_and_device_operations():
    operation = SimpleNamespace(submit=MagicMock())
    plan = PoolTransferPlan((), (operation,))
    factory = MagicMock(return_value=plan)
    manager = SimpleNamespace(
        native_mxfp4_enabled=False,
        moe_layers=[SimpleNamespace(quant_method=SimpleNamespace(build_pool_transfer_plan=factory))],
    )
    adapter = ExpertPoolAdapter(manager)
    stream = SimpleNamespace(npu_stream=1)
    with patch("torch.npu.is_current_stream_capturing", return_value=False), patch(
        "torch.npu.current_stream", return_value=stream
    ):
        scheduler = PoolTransferScheduler(adapter)
        scheduler.submit(0, 0)
        scheduler.submit(0, 0)
    factory.assert_called_once_with(manager, 0, 0)
    assert operation.submit.call_count == 2
