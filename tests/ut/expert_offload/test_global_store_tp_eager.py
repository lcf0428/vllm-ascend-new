"""TP eager contracts, tested against production methods without NPU imports."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.global_store.config import GlobalStoreConfig

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"


def method(file, cls, name, namespace):
    tree = ast.parse((ROOT / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    function = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / file), "exec"), namespace)
    return namespace[name]


def settings(tmp_path, graph=False):
    return GlobalStoreConfig.parse({
        "expert_offload": True, "enable_multi_card": True,
        "host_storage": {"mode": "global_sharded", "runtime_dir": str(tmp_path)},
        "graph": {"mc2_decode": graph},
    })


def parallel(**updates):
    return SimpleNamespace(**dict(dict(tensor_parallel_size=4, data_parallel_size=1,
                                      pipeline_parallel_size=1, prefill_context_parallel_size=1,
                                      decode_context_parallel_size=1, enable_expert_parallel=True,
                                      enable_eplb=False, enable_elastic_ep=False), **updates))


def test_tp_eager_and_existing_dp_graph_contracts(tmp_path):
    settings(tmp_path).validate_parallel(parallel(), enforce_eager=True)
    settings(tmp_path, graph=True).validate_parallel(parallel(), enforce_eager=False)
    settings(tmp_path, graph=True).validate_parallel(
        parallel(tensor_parallel_size=1, data_parallel_size=4), enforce_eager=False)


@pytest.mark.parametrize("changes, eager, graph, error", [
    ({"data_parallel_size": 4}, True, False, "DP=1"),
    ({"pipeline_parallel_size": 2}, True, False, "PP=PCP=DCP=1"),
    ({"prefill_context_parallel_size": 2}, True, False, "PP=PCP=DCP=1"),
    ({"decode_context_parallel_size": 2}, True, False, "PP=PCP=DCP=1"),
    ({"enable_expert_parallel": False}, True, False, "enable_expert_parallel"),
    ({"enable_eplb": True}, True, False, "static EP"),
    ({"enable_elastic_ep": True}, True, False, "static EP"),
])
def test_unsupported_parallel_layout_fails_before_allocation(tmp_path, changes, eager, graph, error):
    with pytest.raises(ValueError, match=error):
        settings(tmp_path, graph).validate_parallel(parallel(**changes), enforce_eager=eager)


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 8, 9, 21, 22, 513])
@pytest.mark.parametrize("mc2", [False, True])
def test_prepare_ids_mask_and_finalize_preserve_token_order(rows, mc2):
    tp = 4
    padded = ((rows + tp - 1) // tp) * tp
    context = SimpleNamespace(mc2_mask=torch.arange(padded) < rows, padded_num_tokens=padded)
    namespace = {"torch": torch, "nn": torch.nn, "QuantType": SimpleNamespace(NONE=0),
                 "_EXTRA_CTX": context, "MoEPrepareOutput": lambda **kw: SimpleNamespace(**kw),
                 "_pad_tokens_with_cat": lambda x, size: torch.cat([x, x.new_zeros(size - len(x), x.shape[1])])}
    base = "PrepareAndFinalizeWithAll2All"
    cls = "PrepareAndFinalizeWithMC2" if mc2 else base
    prepare = method("ops/fused_moe/prepare_finalize.py", cls, "prepare", namespace)
    ids_method = method("ops/fused_moe/prepare_finalize.py", cls, "pad_and_split_input_ids", namespace)
    finalize = method("ops/fused_moe/prepare_finalize.py", base, "finalize", namespace)
    x = torch.arange(rows * 2).reshape(rows, 2).float()
    shards, managers, masks = [], [], []
    for rank in range(tp):
        manager = SimpleNamespace(tp_size=tp, tp_rank=rank, lora_context=None,
                                  moe_config=SimpleNamespace(tp_group=SimpleNamespace(device_group=None)))
        out = prepare(manager, x, torch.zeros(rows, 16))
        ids = ids_method(manager, torch.arange(rows))
        assert len(out.hidden_states) == padded // tp
        assert len(ids) == len(out.hidden_states)
        start = rank * (padded // tp)
        count = min(padded // tp, max(0, rows - start))
        assert ids[:count].tolist() == list(range(start, start + count))
        if mc2:
            assert out.mc2_mask.sum().item() == count
            masks.append(out.mc2_mask)
        shards.append(out.hidden_states)
        managers.append((manager, out.padded_hidden_states_shape))

    def gather(outputs, local, group):
        for dst, src in zip(outputs, shards):
            dst.copy_(src)

    namespace["dist"] = SimpleNamespace(all_gather=gather)
    for manager, shape in managers:
        actual = finalize(manager, shards[manager.tp_rank], True, shape)
        torch.testing.assert_close(actual, x)
    if mc2:
        assert torch.cat(masks).sum().item() == rows


@pytest.mark.parametrize("failure", [None, "expert_tp", "partition", "ep_rank", "mc2_order", "tp_members"])
def test_layer_validates_full_expert_layout_and_rank_order(monkeypatch, failure):
    ep = SimpleNamespace(world_size=4, rank_in_group=2, ranks=[0, 1, 2, 3])
    mc2 = SimpleNamespace(rank_in_group=2, ranks=[0, 1, 2, 3])
    tp = SimpleNamespace(ranks=[0, 1, 2, 3])
    physical = ModuleType("vllm.distributed.parallel_state")
    physical.get_ep_group = lambda: ep
    physical.get_tp_group = lambda: tp
    ascend = ModuleType("vllm_ascend.distributed.parallel_state")
    ascend.get_mc2_group = lambda: mc2
    monkeypatch.setitem(sys.modules, physical.__name__, physical)
    monkeypatch.setitem(sys.modules, ascend.__name__, ascend)
    config = SimpleNamespace(tp_size=1, intermediate_size=128, intermediate_size_per_partition=128,
                             ep_size=4, moe_parallel_config=SimpleNamespace(tp_rank=0, ep_rank=2))
    if failure == "expert_tp":
        config.tp_size = 4
    elif failure == "partition":
        config.intermediate_size_per_partition = 32
    elif failure == "ep_rank":
        config.moe_parallel_config.ep_rank = 1
    elif failure == "mc2_order":
        mc2.ranks = [0, 2, 1, 3]
    elif failure == "tp_members":
        tp.ranks = [0, 4]
    validate = method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                      "_validate_global_layer_layout", {})
    manager = SimpleNamespace(_global_expected_ep_size=4)
    layer = SimpleNamespace(moe_config=config)
    if failure:
        with pytest.raises(ValueError):
            validate(manager, layer)
    else:
        validate(manager, layer)


def test_global_store_prefetch_state_needs_no_external_diagnostics(monkeypatch):
    from unittest.mock import Mock

    from vllm_ascend.expert_offload.cpu_prefetch_diagnostics import NullCpuPrefetchTrace

    native = Mock(side_effect=AssertionError("must not open a diagnostic library"))
    namespace = {"torch_npu": SimpleNamespace(npu=SimpleNamespace(Stream=object)),
                 "threading": __import__("threading"), "CpuPrefetchTrace": native,
                 "NullCpuPrefetchTrace": NullCpuPrefetchTrace, "maybe_create_driver": lambda _: None}
    initialize = method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                        "_init_prefetch_state", namespace)
    manager = SimpleNamespace(offload_config=SimpleNamespace(global_store_enabled=True,
                                                             expert_prefetch_wait_timing=False))
    initialize(manager)
    native.assert_not_called()
    trace = manager._prefetch_trace
    assert trace.path is None
    assert trace.next_static_id() == 1
    assert trace.begin_callback(0, 1) is None
    trace.mark(10)
    trace.mark_static(10, 0, 1, False)
    trace.mark_operation(10, 0)
    trace.end_callback(None)
    trace.close()


@pytest.mark.parametrize("rows", [0, 1, 3, 5, 7, 8, None])
def test_dp1_metadata_keeps_actual_rows_without_collective(rows):
    sync = method("worker/model_runner_v1.py", "NPUModelRunner", "_sync_metadata_across_dp", {"torch": torch, "CUDAGraphMode": SimpleNamespace(NONE=0)})
    runner = SimpleNamespace(dp_size=1, ascend_config=SimpleNamespace(
        expert_offload_config=SimpleNamespace(global_store_enabled=True)))
    count, sizes, mode = sync(runner, 8, cudagraph_mode="FULL", num_actual_tokens=rows)
    assert count == 8 and mode == "FULL"
    assert sizes.tolist() == [8]
    assert sizes.expert_offload_actual_rows == [8 if rows is None else rows]


@pytest.mark.parametrize("rows", [-1, 9])
def test_dp1_metadata_rejects_invalid_rows(rows):
    sync = method("worker/model_runner_v1.py", "NPUModelRunner", "_sync_metadata_across_dp", {"torch": torch, "CUDAGraphMode": SimpleNamespace(NONE=0)})
    runner = SimpleNamespace(dp_size=1, ascend_config=SimpleNamespace(
        expert_offload_config=SimpleNamespace(global_store_enabled=True)))
    with pytest.raises(ValueError, match="unpadded DP"):
        sync(runner, 8, cudagraph_mode="FULL", num_actual_tokens=rows)


def test_legacy_dp1_metadata_stays_unset():
    sync = method("worker/model_runner_v1.py", "NPUModelRunner", "_sync_metadata_across_dp", {"CUDAGraphMode": SimpleNamespace(NONE=0)})
    runner = SimpleNamespace(dp_size=1, ascend_config=SimpleNamespace(
        expert_offload_config=SimpleNamespace(global_store_enabled=False)))
    assert sync(runner, 8, cudagraph_mode="NONE") == (8, None, "NONE")


@pytest.mark.parametrize("rows", [0, 1, 3, 5, 8])
@pytest.mark.parametrize("metadata_rows", [None, 2])
def test_dp1_forward_uses_current_valid_rows_and_stable_graph_key(monkeypatch, rows, metadata_rows):
    import math
    from contextlib import contextmanager
    from enum import Enum

    class Mode(Enum):
        NONE = 0
        FULL = 1

    class Comm(Enum):
        MC2 = 1
        ALLTOALL = 2

    @contextmanager
    def noop(*args, **kwargs):
        yield

    offload = SimpleNamespace(global_store_enabled=True, global_store_config=SimpleNamespace(
        graph={"mc2_decode": True}))
    ctx = SimpleNamespace(dp_metadata=None, cudagraph_runtime_mode=Mode.FULL)
    mask = torch.empty(8, dtype=torch.bool)
    seen = []

    def select(*args, **kwargs):
        assert args[0] == 8
        seen.append(kwargs["actual_tokens_across_dp"])
        return Comm.MC2

    namespace = dict(math=math, torch=torch, contextmanager=contextmanager, CUDAGraphMode=Mode,
                     MoECommType=Comm, QuantType=SimpleNamespace(NONE=0),
                     sync_v2_extra_kwargs=lambda _: None, set_current_vllm_config=noop,
                     set_forward_context=noop, get_forward_context=lambda: ctx,
                     get_ascend_config=lambda: SimpleNamespace(expert_offload_config=offload),
                     select_moe_comm_method=select, _should_skip_compiled_for_multi_card_offload=lambda _: False,
                     _is_decode_only_node=lambda _: False, use_cann_megamoe=lambda _: False,
                     get_tensor_model_parallel_world_size=lambda: 4, get_dp_group=lambda: SimpleNamespace(world_size=1),
                     has_layer_idx=lambda _: False, get_mc2_mask=lambda: mask)
    tree = ast.parse((ROOT / "ascend_forward_context.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "set_ascend_forward_context")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "ascend_forward_context.py", "exec"), namespace)
    mod = ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    mod.get_moe_comm_method = lambda _: object()
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    metadata = None
    if metadata_rows is not None:
        metadata = torch.tensor([16])
        metadata.expert_offload_actual_rows = [metadata_rows]
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    with namespace["set_ascend_forward_context"](None, cfg, num_tokens=8, num_actual_tokens=rows,
                                                 num_tokens_across_dp=metadata, aclgraph_runtime_mode=Mode.FULL):
        assert seen == [[rows]]
        assert mask.tolist() == [i < rows for i in range(8)]
        assert ctx.expert_offload_graph_key == (Comm.MC2, False, 1)
        assert ctx.cudagraph_runtime_mode is Mode.FULL
