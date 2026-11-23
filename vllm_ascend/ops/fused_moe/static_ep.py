# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape EP dispatch for graphs consuming full local expert shards.

Every rank gathers the padded token inputs, computes only its expert shard,
then reduce-scatters weighted contributions back to the token owners. No
routing-dependent output allocation, Host split counts, or Host waits occur.
"""
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchOutput


@dataclass(frozen=True)
class StaticEPCombineMetadata:
    local_shape: torch.Size
    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor


def _gather(value: torch.Tensor, group, world_size: int) -> torch.Tensor:
    output = torch.empty(
        (value.shape[0] * world_size, *value.shape[1:]),
        dtype=value.dtype, device=value.device,
    )
    dist.all_gather_into_tensor(output, value.contiguous(), group=group)
    return output


def dispatch_static_ep(token_input, group, rank: int, world_size: int,
                       num_experts: int, num_local_experts: int):
    """Route unquantized activations; the existing MLP performs quantization."""
    local = token_input.hidden_states
    if local.ndim != 2 or local.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Static full-shard EP requires a 2D floating-point activation")
    if token_input.routing.pertoken_scale is not None:
        raise ValueError("Static full-shard EP does not accept prequantized activations")
    if num_experts != world_size * num_local_experts:
        raise ValueError("Static full-shard EP requires contiguous, equal expert shards")
    x = _gather(local, group, world_size)
    ids = _gather(token_input.topk_ids, group, world_size)
    weights = _gather(token_input.topk_weights, group, world_size)
    first = rank * num_local_experts
    last = first + num_local_experts
    weights = weights * ((ids >= first) & (ids < last))
    sorted_x, row_idx, counts, _ = DeviceOperator.npu_moe_init_routing(
        x, ids, scale=None,
        active_num=ids.numel(), expert_num=num_experts,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[first, last], quant_mode=-1,
    )
    return MoETokenDispatchOutput(
        hidden_states=sorted_x, group_list=counts.to(torch.int64),
        group_list_type=1, dynamic_scale=None,
        combine_metadata=StaticEPCombineMetadata(local.shape, weights, row_idx),
    )


def combine_static_ep(hidden_states, metadata: StaticEPCombineMetadata, group):
    partial = DeviceOperator.npu_moe_token_unpermute(
        permuted_tokens=hidden_states,
        sorted_indices=metadata.expanded_row_idx,
        probs=metadata.topk_weights.to(hidden_states.dtype),
    )
    output = torch.empty(metadata.local_shape, dtype=partial.dtype, device=partial.device)
    dist.reduce_scatter_tensor(output, partial.contiguous(), group=group)
    return output
