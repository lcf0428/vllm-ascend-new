"""Exercise real manager hooks and admission without constructing an NPU model.

The AST loader isolates existing methods from optional model/plugin imports;
the methods under test are the production methods, not copies of their logic.
"""

import ast
import logging
import threading
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"


def production_method(file, class_name, method, namespace):
    tree = ast.parse((ROOT / file).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(ROOT / file), "exec"), namespace)
    return namespace[method]


def test_eager_error_propagates_and_graph_error_terminates():
    namespace = {"threading": threading, "logger": logging.getLogger(__name__), "os": Mock()}
    eager = production_method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                              "_update_weights_multi_card_guarded", namespace)
    graph = production_method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                              "_update_weights_multi_card_graph_guarded", namespace)
    manager = SimpleNamespace(_global_store=object(),
                              offload_config=SimpleNamespace(global_store_config=SimpleNamespace(
                                  transport={"timeout_s": 60})),
                              _global_timeout_exit=Mock(), _note_cb_failure=Mock(),
                              _update_weights_multi_card=Mock(side_effect=RuntimeError("failed read")))
    manager._update_weights_multi_card_guarded = lambda args: eager(manager, args)
    with pytest.raises(RuntimeError, match="failed read"):
        eager(manager, ())
    manager._note_cb_failure.assert_not_called()
    graph(manager, ())
    namespace["os"]._exit.assert_called_once_with(1)


def test_prefill_uses_unified_executor_instead_of_private_shard_copy():
    namespace = {"torch": SimpleNamespace(npu=SimpleNamespace(is_current_stream_capturing=lambda: False))}
    method = production_method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                               "_prefill_load_layer_shard", namespace)
    executor = Mock()
    manager = SimpleNamespace(_prefill_initialized=True, moe_layers=[object()], num_device_layers=2,
                              mc_shard_size=64, ep_rank=2, _global_store=object(),
                              _global_executor=executor, _global_prefill_load=Mock())
    method(manager, 3)
    executor.run.assert_called_once_with(manager._global_prefill_load, 3, 1, 128, 64)


def test_raw_and_quant_tasks_use_global_sources_even_for_locally_owned_experts():
    namespace = {}
    source_method = production_method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                                      "_expert_src_storage", namespace)
    store = SimpleNamespace(source=Mock(return_value=object()))
    manager = SimpleNamespace(_global_store=store)
    source = source_method(manager, 3, 100, "w2")
    assert source is store.source.return_value
    store.source.assert_called_once_with(3, 100, "w2")


def test_real_config_accepts_global_store_and_rejects_cpu_moe():
    # Configuration class itself is independent of the model runtime; extract
    # its real definition and provide the CPU MoE parser dependency.
    tree = ast.parse((ROOT / "ascend_config.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExpertOffloadConfig")
    namespace = {"CpuMoeConfig": lambda c: SimpleNamespace(enabled=bool((c or {}).get("enabled")))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "ascend_config.py", "exec"),
         namespace)
    # _validate_config uses only scalar/container types and logging.
    namespace["logger"] = logging.getLogger(__name__)
    config_class = namespace["ExpertOffloadConfig"]
    settings = {"expert_offload": True, "enable_multi_card": True, "num_device_experts": [128],
                "host_storage": {"mode": "global_sharded", "runtime_dir": "/tmp/own-expert-test"},
                "graph": {"mc2_decode": True}}
    cfg = config_class(settings)
    assert cfg.global_store_enabled
    assert cfg.num_device_experts_for_rank(0, 4) == 32
    with pytest.raises(ValueError, match="cpu_moe"):
        config_class(dict(settings, cpu_moe={"enabled": True}))


def test_real_comm_selector_admits_48_routes_and_rejects_192():
    class GraphMode(IntEnum):
        NONE = 0
        FULL = 2

        def decode_mode(self):
            return self

    class Comm(Enum):
        ALLGATHER = 1
        ALLTOALL = 2
        MC2 = 3

    config = SimpleNamespace(expert_offload=True, enable_multi_card=True, global_store_enabled=True,
                             h2d_backend="torch", shard_per_rank=True, num_device_experts_list=[128],
                             moe_offload_debug=False)
    namespace = {"QuantType": SimpleNamespace(NONE=0), "MoECommType": Comm,
                 "VllmConfig": object, "CUDAGraphMode": GraphMode,
                 "is_moe_model": lambda _: True, "get_mc2_tokens_capacity": lambda: 512,
                 "get_current_hardware_profile": lambda: SimpleNamespace(moe_comm_policy=None),
                 "get_ep_group": lambda: SimpleNamespace(world_size=4),
                 "get_ascend_config": lambda: SimpleNamespace(expert_offload_config=config),
                 "logger": Mock()}
    tree = ast.parse((ROOT / "ascend_forward_context.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_moe_comm_method")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "ascend_forward_context.py", "exec"), namespace)
    vllm = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(num_experts_per_tok=6)),
                           parallel_config=SimpleNamespace(enable_expert_parallel=True, data_parallel_size=4),
                           compilation_config=SimpleNamespace(cudagraph_mode=GraphMode.FULL))
    select = namespace["select_moe_comm_method"]
    assert select(8, vllm, actual_tokens_across_dp=[8, 0, 0, 0]) == Comm.MC2
    assert select(8, vllm, actual_tokens_across_dp=[8, 8, 8, 8]) == Comm.ALLTOALL
    assert select(8, vllm) == Comm.ALLTOALL  # no metadata must remain conservative
    config.global_store_enabled = False
    assert select(8, vllm, actual_tokens_across_dp=[8, 0, 0, 0]) == Comm.ALLTOALL


def test_real_dp_sync_preserves_unpadded_counts_and_disables_skip_optimization():
    class GraphMode(IntEnum):
        NONE = 0
        FULL = 2

    class Comm(Enum):
        ALLGATHER = 1
        ALLTOALL = 2
        MC2 = 3
        FUSED_MC2 = 4

    def all_reduce(tensor, **kwargs):
        tensor[0] = torch.tensor([8, 8, 8, 8])
        tensor[1] = GraphMode.FULL
        tensor[2] = torch.tensor([8, 0, 0, 0])

    selector = Mock(return_value=Comm.MC2)
    namespace = {"torch": torch, "CUDAGraphMode": GraphMode, "MoECommType": Comm,
                 "should_skip_allreduce_across_dp_group": lambda *a: True,
                 "dist": SimpleNamespace(all_reduce=all_reduce),
                 "get_dp_group": lambda: SimpleNamespace(cpu_group=None),
                 "_post_process_cudagraph_mode": lambda t: int(t[1].min()),
                 "select_moe_comm_method": selector, "use_cann_megamoe": lambda c: False}
    method = production_method("worker/model_runner_v1.py", "NPUModelRunner",
                               "_sync_metadata_across_dp", namespace)
    manager = SimpleNamespace(dp_size=4, dp_rank=0,
                              vllm_config=SimpleNamespace(model_config=SimpleNamespace(enable_return_routed_experts=False)),
                              ascend_config=SimpleNamespace(expert_offload_config=SimpleNamespace(global_store_enabled=True),
                                  finegrained_tp_config=SimpleNamespace(oproj_tensor_parallel_size=1,
                                      lmhead_tensor_parallel_size=1, embedding_tensor_parallel_size=1,
                                      mlp_tensor_parallel_size=1)))
    size, metadata, mode = method(manager, 8, cudagraph_mode=GraphMode.FULL, num_actual_tokens=8)
    assert (size, mode) == (8, GraphMode.FULL)
    assert metadata.tolist() == [8, 8, 8, 8]
    assert metadata.expert_offload_actual_rows == [8, 0, 0, 0]
    assert selector.call_args.kwargs["actual_tokens_across_dp"] == [8, 0, 0, 0]


def test_published_store_cannot_be_reloaded_or_converted_twice():
    namespace = {"torch": torch}
    manager = SimpleNamespace(_global_store=object())
    finalize = production_method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                                 "_finalize_offload", namespace)
    # An accidental repeated finalize must not NZ-convert shared bytes again.
    finalize(manager, None)
    for name, args in [("load_w13", (0, 0, None, "w1")), ("load_w2", (0, 0, None))]:
        method = production_method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                                   name, namespace)
        with pytest.raises(RuntimeError, match="immutable"):
            method(manager, *args)


@pytest.mark.parametrize("fail", [True, False])
def test_decode_commits_residency_and_mapping_only_after_global_ready(fail):
    namespace = {"torch": torch, "time": __import__("time")}
    method = production_method("expert_offload/expert_offload_manager.py", "ExpertOffloadManager",
                               "_update_weights_multi_card_impl", namespace)
    resident = {0: 0, 1: 1}
    previous = torch.tensor([0, 1], dtype=torch.int32)
    output_map = torch.full((2,), -1, dtype=torch.int32)
    ready_observations = []

    def gather(failure):
        # Device slots may have been filled, but public residency/map cannot
        # change until every rank's READY/error exchange completes.
        assert resident == {0: 0, 1: 1}
        assert output_map.tolist() == [-1, -1]
        ready_observations.append(failure)
        return [failure]

    def copy(layer, index, misses, pending, debug_context):
        pending[0] = 1
        pending[1] = 0
        if fail:
            raise RuntimeError("injected partial bundle failure")

    manager = SimpleNamespace(
        _debug=False, ep_size=1, ep_rank=0, _global_store=object(),
        offload_config=SimpleNamespace(shard_per_rank=True),
        _shared_h2d_layer_ready=lambda _: True,
        _gather_global_counts_and_hotness=lambda *a: (torch.tensor([1, 1]), True, None, previous),
        _compute_resident_hits=lambda *a: (resident, [], [(0, 1), (1, 0)]),
        _mc_prev_log2phy={0: previous}, _global_all_gather=gather,
        _h2d_load_mc_misses=copy, _global_executor=SimpleNamespace(run=lambda fn, *args: fn(*args)),
    )
    args = (torch.tensor([[0]], dtype=torch.int32), output_map, object(), 0, 2, False, None)
    if fail:
        with pytest.raises(RuntimeError, match="materialization failed"):
            method(manager, args)
        assert resident == {0: 0, 1: 1}
        assert manager._mc_prev_log2phy[0] is previous
        assert output_map.tolist() == [-1, -1]
    else:
        method(manager, args)
        assert resident == {0: 1, 1: 0}
        assert (output_map >= 0).all()
        assert manager._mc_prev_log2phy[0] is not previous
    assert len(ready_observations) == 1
