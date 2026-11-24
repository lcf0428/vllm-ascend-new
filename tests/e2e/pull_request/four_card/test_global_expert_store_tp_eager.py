# SPDX-License-Identifier: Apache-2.0
"""Four idle NPUs: real TP/EP groups, paging and eager MC2/ALLTOALL accuracy.

Uses a synthetic one-layer, 64-expert MoE with BF16 weights and scale fields.
The expert arithmetic is ordinary Torch SwiGLU, so this does not establish
DeepSeek's FP4 grouped-matmul accuracy. FULL tests use static masked Torch expert arithmetic.
"""

import json
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _weights(expert, hidden=1024, intermediate=128):
    generator = torch.Generator().manual_seed(1700 + expert)
    return {
        "w13": (torch.randn(hidden, 2 * intermediate, generator=generator) * 0.03).bfloat16(),
        "w2": (torch.randn(intermediate, hidden, generator=generator) * 0.03).bfloat16(),
        "w13_weight_scale": torch.linspace(0.8, 1.2, 2 * intermediate) + expert / 100,
        "w2_weight_scale": torch.linspace(0.9, 1.1, hidden) + expert / 100,
    }


def _expert(x, w13, w2, scale13, scale2):
    gate, up = (x @ w13).float().mul(scale13).chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up).to(w2.dtype).matmul(w2).float() * scale2


def _worker(rank, port, directory, full_graph=False, world_size=4):
    # Lazy worker imports keep NPU initialization out of the spawning process.
    import torch_npu  # noqa: F401
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel,
        get_ep_group, get_tp_group, init_distributed_environment, initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ascend_forward_context import (
        MoECommType, set_ascend_forward_context, set_mc2_mask, set_mc2_tokens_capacity,
    )
    from vllm_ascend.distributed.parallel_state import (
        destroy_ascend_model_parallel, init_ascend_model_parallel,
    )
    from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
    from vllm_ascend.expert_offload.routed_experts import apply_offloaded_experts
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import build_token_dispatch_input
    from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
    from vllm_ascend.quantization.quant_type import QuantType

    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    hidden, intermediate, experts, slots, topk = 1024, 128, 64, 48, 6
    offload = {
        "expert_offload": True, "enable_multi_card": True,
        "h2d_backend": "torch", "num_device_experts": [slots], "num_device_layers": 1,
        "cache_policy_enabled": True,
        "host_storage": {"mode": "global_sharded", "runtime_dir": str(Path(directory) / "store")},
        "transport": {"staging_buffer_mb": 1, "timeout_s": 60},
        "graph": {"mc2_decode": full_graph},
    }
    cfg = EngineArgs(model=directory, tensor_parallel_size=world_size, enable_expert_parallel=True,
                     enforce_eager=not full_graph, dtype="bfloat16", max_model_len=64, max_num_seqs=8,
                     compilation_config=({"cudagraph_mode": "FULL_DECODE_ONLY",
                                          "cudagraph_capture_sizes": [4, 8]} if full_graph else {}),
                     max_num_batched_tokens=64, additional_config={"expert_offload_config": offload,
                                                                 "enable_cpu_binding": False}).create_engine_config()
    with set_current_vllm_config(cfg):
        init_ascend_config(cfg)
        init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="hccl")
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        init_ascend_model_parallel(cfg.parallel_config)
        set_mc2_tokens_capacity(cfg, 16, 1)
        set_mc2_mask(cfg, torch.device("npu", rank))
        manager = ExpertOffloadManager(cfg)
        layer = torch.nn.Module()
        layer.layer_name = "model.layers.0.mlp.experts"
        layer.global_num_experts = experts
        layer.gate = None
        layer.moe_config = SimpleNamespace(
            tp_size=1, ep_size=world_size, dp_size=1, pcp_size=1, hidden_dim=hidden,
            experts_per_token=topk, num_experts=experts, num_logical_experts=experts,
            num_local_experts=slots // world_size, intermediate_size=intermediate,
            swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None,
            intermediate_size_per_partition=intermediate, is_sequence_parallel=False,
            moe_parallel_config=SimpleNamespace(tp_rank=0, ep_rank=rank),
            tp_group=get_tp_group(), ep_group=get_ep_group())
        layer.w13_weight = torch.zeros(slots // world_size, 2 * intermediate, hidden,
                                       dtype=torch.bfloat16, device="npu")
        layer.w2_weight = torch.zeros(slots // world_size, hidden, intermediate,
                                      dtype=torch.bfloat16, device="npu")
        layer.w13_weight_scale = torch.zeros(slots // world_size, 2 * intermediate, device="npu")
        layer.w2_weight_scale = torch.zeros(slots // world_size, hidden, device="npu")
        manager.init_layer_cpu_buffers(layer, 0)
        for eid in range(experts):
            weights = _weights(eid)
            w1, w3 = weights["w13"].chunk(2, dim=1)
            manager.load_w13(0, eid, w1.t(), "w1")
            manager.load_w13(0, eid, w3.t(), "w3")
            manager.load_w2(0, eid, weights["w2"].t())
            s1, s3 = weights["w13_weight_scale"].chunk(2)
            manager._load_scale_shard(0, eid, "w13_weight_scale", "w1", s1)
            manager._load_scale_shard(0, eid, "w13_weight_scale", "w3", s3)
            manager._load_scale_shard(0, eid, "w2_weight_scale", "w2", weights["w2_weight_scale"])
        manager.drain_load_pool()
        layer.w13_weight = layer.w13_weight.transpose(1, 2).contiguous()
        layer.w2_weight = layer.w2_weight.transpose(1, 2).contiguous()
        manager._finalize_offload(None)
        remote_reads = [0]
        read_into = manager._global_store.read_into

        def count_read(source, offset, target, size):
            if source.bundle.owner != rank:
                remote_reads[0] += 1
            return read_into(source, offset, target, size)

        manager._global_store.read_into = count_read
        layer.log2phy = torch.full((experts,), -1, dtype=torch.int32, device="npu")
        layer.ascend_expert_map = torch.arange(slots, dtype=torch.int32, device="npu")
        layer.enable_multi_card = True
        layer.tid2eid = None
        layer.router = SimpleNamespace(renormalize=True, scoring_func="softmax")

        class TinyExperts:
            def apply(self, layer, x, topk_weights, topk_ids, **kwargs):
                from vllm_ascend.ascend_forward_context import _EXTRA_CTX

                dispatcher = _EXTRA_CTX.moe_comm_method.token_dispatcher
                dispatch_input = build_fused_experts_input(
                    hidden_states=x, topk_weights=topk_weights, topk_ids=topk_ids,
                    layer=layer, quant_type=QuantType.NONE, dynamic_eplb=False,
                    expert_map=layer.ascend_expert_map, mc2_mask=layer.ascend_mc2_mask)
                dispatched = dispatcher.token_dispatch(
                    build_token_dispatch_input(fused_experts_input=dispatch_input))
                if full_graph:
                    # Fixed-shape arithmetic: no Host reads of captured routing.
                    ends = dispatched.group_list
                    if dispatched.group_list_type == 1:
                        ends = ends.cumsum(0)
                    positions = torch.arange(dispatched.hidden_states.shape[0], device="npu")
                    expert_output = torch.zeros_like(dispatched.hidden_states)
                    for slot in range(layer.w13_weight.shape[0]):
                        start = 0 if slot == 0 else ends[slot - 1]
                        active = (positions >= start) & (positions < ends[slot])
                        value = _expert(dispatched.hidden_states, layer.w13_weight[slot], layer.w2_weight[slot],
                                        layer.w13_weight_scale[slot], layer.w2_weight_scale[slot])
                        expert_output.add_(torch.where(active[:, None], value.bfloat16(), 0))
                else:
                    ends = dispatched.group_list.cpu().tolist()
                    if dispatched.group_list_type == 1:
                        ends = torch.tensor(ends).cumsum(0).tolist()
                    expert_output = torch.zeros_like(dispatched.hidden_states)
                    start = 0
                    for slot, end in enumerate(ends):
                        end = int(end)
                        if end > start:
                            expert_output[start:end] = _expert(
                                dispatched.hidden_states[start:end], layer.w13_weight[slot], layer.w2_weight[slot],
                                layer.w13_weight_scale[slot], layer.w2_weight_scale[slot]).bfloat16()
                        start = end
                return SimpleNamespace(routed_out=dispatcher.token_combine(expert_output, dispatched.combine_metadata))

        layer.quant_method = TinyExperts()
        setup_moe_comm_method(layer.moe_config)
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        runner = SimpleNamespace(dp_size=1, ascend_config=SimpleNamespace(
            expert_offload_config=SimpleNamespace(global_store_enabled=True)))
        calls = [0]
        callback_calls = [0]
        callback = manager._update_weights_multi_card_graph_guarded

        def count_callback(args):
            callback_calls[0] += 1
            return callback(args)

        manager._update_weights_multi_card_graph_guarded = count_callback

        def forward(x, logits):
            calls[0] += 1
            comm = _EXTRA_CTX.moe_comm_method
            prepared = comm.prepare(x, logits)
            route_scores, ids = prepared.router_logits.topk(topk, dim=-1)
            route_weights = route_scores.softmax(-1).to(torch.bfloat16)
            layer.ascend_mc2_mask = prepared.mc2_mask
            result = apply_offloaded_experts(layer, prepared.hidden_states, route_weights,
                                            ids.int(), prepared.router_logits)
            return comm.finalize(result.routed_out, True, prepared.padded_hidden_states_shape)

        wrapped = ACLGraphWrapper(forward, cfg, CUDAGraphMode.FULL) if full_graph else forward
        buffers = {n: (torch.empty(n, hidden, dtype=torch.bfloat16, device="npu"),
                       torch.empty(n, experts, device="npu")) for n in (4, 8, 9, 22)}
        # Reuse each MC2 graph with different valid rows/routes, then cross an
        # ALLTOALL boundary and return to the already captured MC2 graph.
        cases = ([(8, 8, 7), (8, 1, 52), (8, 3, 31), (8, 0, 4),
                  (4, 3, 11), (4, 1, 48), (8, 7, 42), (9, 9, 5),
                  (8, 5, 9), (22, 22, 0), (8, 8, 55), (4, 2, 29), (8, 1, 7)]
                 if full_graph else [(n, n, shift) for n, shift in
                  [(1, 7), (3, 11), (5, 9), (8, 5), (9, 5), (22, 0), (1, 52), (8, 5)]])
        graph_ids = {}
        for rows, actual, shift in cases:
            generator = torch.Generator().manual_seed(1800 + actual)
            x_cpu = (torch.randn(rows, hidden, generator=generator) * 0.2).bfloat16()
            if rows not in buffers:
                buffers[rows] = (torch.empty_like(x_cpu, device="npu"), torch.empty(rows, experts, device="npu"))
            x, logits = buffers[rows]
            x.copy_(x_cpu.to("npu") / world_size)
            dist.all_reduce(x, group=get_tp_group().device_group)
            logits.fill_(-20.0)
            ids_cpu = (torch.arange(rows).unsqueeze(1) + shift + 5 * torch.arange(topk)) % experts
            scores_cpu = torch.linspace(2.0, 1.0, topk)
            logits.scatter_(1, ids_cpu.to("npu"), scores_cpu.to("npu").expand(rows, topk))
            requested = CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE
            _, metadata, _ = NPUModelRunner._sync_metadata_across_dp(
                runner, rows, cudagraph_mode=requested, num_actual_tokens=actual)
            # Warmup allocates operator workspaces without changing input addresses.
            if full_graph and rows in (4, 8) and rows not in graph_ids:
                with set_ascend_forward_context(None, cfg, num_tokens=rows, num_actual_tokens=actual,
                                                num_tokens_across_dp=metadata):
                    forward(x, logits)
                torch.npu.synchronize()
            before = calls[0]
            with set_ascend_forward_context(None, cfg, num_tokens=rows, num_actual_tokens=actual,
                                            num_tokens_across_dp=metadata, aclgraph_runtime_mode=requested):
                mc2 = _EXTRA_CTX.moe_comm_type == MoECommType.MC2
                mode = get_forward_context().cudagraph_runtime_mode
                assert (mode == CUDAGraphMode.FULL) == (full_graph and mc2)
                output_tensor = wrapped(x, logits)
                if full_graph and mc2 and rows not in graph_ids:
                    # ACL capture records work; its output is only valid after
                    # replay (startup capture outputs are normally discarded).
                    _capture_output = output_tensor
                    output_tensor = wrapped(x, logits)
                output = output_tensor.float().cpu()[:actual]
            expected = torch.zeros(actual, hidden)
            weights_cpu = scores_cpu.softmax(0).bfloat16().float()
            for token in range(actual):
                for k, eid in enumerate(ids_cpu[token].tolist()):
                    fields = _weights(eid)
                    value = _expert(x_cpu[token:token + 1].float(), fields["w13"].float(),
                                    fields["w2"].float(), fields["w13_weight_scale"], fields["w2_weight_scale"])
                    expected[token] += value[0] * weights_cpu[k]
            torch.testing.assert_close(output, expected, rtol=0.04, atol=0.001)
            if full_graph and mc2:
                entries = wrapped.concrete_aclgraph_entries
                key = next(k for k in entries if k[0].num_tokens == rows)
                graph = entries[key].aclgraph
                if rows in graph_ids:
                    assert graph_ids[rows] is graph and calls[0] == before
                else:
                    graph_ids[rows] = graph
            error = (output - expected).abs().max().item() if actual else 0.0
            print(f"TP_{'FULL' if full_graph else 'EAGER'}_CASE rank={rank} rows={rows} actual={actual} "
                  f"comm={'MC2' if mc2 else 'ALLTOALL'} mode={mode.name} max_abs_error={error:.6g}", flush=True)
        if full_graph:
            assert len(wrapped.concrete_aclgraph_entries) == 2
            assert callback_calls[0] == sum(rows <= 8 for rows, _, _ in cases)
            print(f"TP_FULL_VERIFIED rank={rank} captures=2 callbacks={callback_calls[0]} "
                  f"remote_reads={remote_reads[0]}", flush=True)
        remote_counts = torch.tensor([remote_reads[0]])
        dist.all_reduce(remote_counts, group=get_ep_group().cpu_group)
        assert remote_counts.item() > 0
        manager.close()
        torch.npu.synchronize()
        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()


def _run(full_graph=False, world_size=4):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="tp4-expert-eager-") as directory:
        config = {
            "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
            "hidden_size": 1024, "intermediate_size": 128, "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 128,
            "num_experts": 64, "num_experts_per_tok": 6, "max_position_embeddings": 64,
            "torch_dtype": "bfloat16",
        }
        (Path(directory) / "config.json").write_text(json.dumps(config))
        mp.spawn(_worker, args=(port, directory, full_graph, world_size), nprocs=world_size, join=True)



def test_global_store_tp4_ep4_eager():
    _run()


def test_global_store_tp4_ep4_full_decode_graph():
    _run(full_graph=True)



def test_global_store_tp2_ep2_full_decode_graph():
    _run(full_graph=True, world_size=2)
