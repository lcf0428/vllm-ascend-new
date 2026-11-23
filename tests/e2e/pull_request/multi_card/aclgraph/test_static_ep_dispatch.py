# SPDX-License-Identifier: Apache-2.0
"""Two idle visible NPUs; validates real GLOBAL graphs with changing EP routes."""
import datetime
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl", rank=rank, world_size=2, timeout=datetime.timedelta(seconds=60)
    )
    dist.all_reduce(torch.ones(1, device="npu"))
    torch.npu.synchronize()
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import build_token_dispatch_input
    from vllm_ascend.ops.fused_moe.static_ep import combine_static_ep, dispatch_static_ep
    from vllm_ascend.quantization.quant_type import QuantType

    stream = torch.npu.Stream()
    cases = {}
    for n in (8, 32, 48):
        x = torch.full((n, 512), rank + 1, dtype=torch.bfloat16, device="npu")
        ids = torch.arange(n * 6, device="npu", dtype=torch.int32).reshape(n, 6) % 256
        weights = torch.full((n, 6), 1 / 8, dtype=torch.bfloat16, device="npu")
        inp = build_token_dispatch_input(fused_experts_input=build_fused_experts_input(
            hidden_states=x, topk_ids=ids, topk_weights=weights,
            num_local_experts=128, quant_type=QuantType.NONE, dynamic_eplb=False,
        ))

        def forward():
            routed = dispatch_static_ep(inp, dist.group.WORLD, rank, 2, 256, 128)
            # Every local expert scales by its owning rank. This makes
            # cross-rank route ownership observable in the combined result.
            return combine_static_ep(
                routed.hidden_states * (rank + 1), routed.combine_metadata, dist.group.WORLD
            )

        with torch.npu.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
            output = forward()
        cases[n] = (graph, output, x, ids, weights)
    for step, n in enumerate((8, 48, 32, 8, 32, 48)):
        graph, output, x, ids, weights = cases[n]
        new_ids = (torch.arange(n * 6, dtype=torch.int32).reshape(n, 6) + step * 103 + rank * 19) % 256
        ids.copy_(new_ids.to("npu"))
        x.fill_(rank + 1 + step)
        graph.replay()
        stream.synchronize()
        coeff = (((new_ids >= 128).to(torch.float32) + 1) * (1 / 8)).sum(-1)
        expected = x.cpu().float() * coeff[:, None]
        torch.testing.assert_close(output.cpu().float(), expected, rtol=0, atol=0)
        print("STATIC_EP_GLOBAL_GRAPH_PASS", rank, n, step, flush=True)
    del graph
    cases.clear()
    dist.destroy_process_group()


def test_full_shard_ep_global_graph_changes_routes_and_shapes():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    test_full_shard_ep_global_graph_changes_routes_and_shapes()
