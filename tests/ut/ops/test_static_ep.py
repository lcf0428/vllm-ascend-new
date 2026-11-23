# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.static_ep import dispatch_static_ep


@pytest.mark.parametrize("shape,dtype,local_experts", [
    ((2, 3, 4), torch.bfloat16, 128),
    ((4, 8), torch.int8, 128),
    ((4, 8), torch.bfloat16, 127),
])
def test_invalid_static_ep_contract_fails_before_collectives(shape, dtype, local_experts):
    token_input = SimpleNamespace(
        hidden_states=torch.zeros(shape, dtype=dtype),
        routing=SimpleNamespace(pertoken_scale=None),
    )
    with patch("torch.distributed.all_gather_into_tensor") as gather:
        with pytest.raises(ValueError):
            dispatch_static_ep(token_input, None, 0, 2, 256, local_experts)
        gather.assert_not_called()


def test_full_graph_dispatch_bypasses_host_split_preprocessing():
    from vllm.config import CUDAGraphMode
    import vllm_ascend.ops.fused_moe.token_dispatcher as mod
    dispatcher = mod.TokenDispatcherWithAll2AllV.__new__(mod.TokenDispatcherWithAll2AllV)
    dispatcher.num_experts = 256
    dispatcher.num_local_experts = 128
    dispatcher.lora_context = None
    group = SimpleNamespace(device_group="test-group", rank_in_group=1, world_size=2)
    config = SimpleNamespace(expert_offload_config=SimpleNamespace(
        expert_offload=True, enable_multi_card=True, h2d_backend="torch"
    ))
    token_input = object()
    result = object()
    with (
        patch.object(mod, "is_forward_context_available", return_value=True),
        patch.object(mod, "get_forward_context", return_value=SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.FULL
        )),
        patch.object(mod, "get_ascend_config", return_value=config),
        patch.object(mod, "get_ep_group", return_value=group),
        patch.object(mod, "dispatch_static_ep", return_value=result) as dispatch,
        patch.object(dispatcher, "_dispatch_preprocess", side_effect=AssertionError("Host splits")),
    ):
        assert dispatcher.token_dispatch(token_input) is result
        dispatch.assert_called_once_with(token_input, "test-group", 1, 2, 256, 128)
