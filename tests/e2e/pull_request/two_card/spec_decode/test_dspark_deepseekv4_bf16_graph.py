# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local DeepSeek-V4 BF16 DSpark regression with TP2/EP2.

Run with the CANN/custom-op environment used by run_tp2_ep2_graph.sh and two
explicitly assigned, idle NPUs. This test loads the local checkpoint twice:
first eager, then FULL_DECODE_ONLY, and compares greedy token IDs across
successive long/short requests. It skips when the checkpoint is unavailable.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.inputs import TokensPrompt

from tests.e2e.conftest import VllmRunner
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

MODEL_PATH = Path("/data/weights/DeepSeek-V4-Flash-0731")
REQUEST_SIZES = [(1107, 1024), (31, 64), (263, 64), (1107, 128)]


def _offload_config(runtime_dir: Path) -> dict:
    return {
        "enable_cpu_binding": False,
        "expert_offload_config": {
            "expert_offload": True,
            "enable_multi_card": True,
            "exclusive_expert_storage": False,
            "h2d_backend": "torch",
            "host_storage": {
                "mode": "global_sharded",
                "replicas": 1,
                "backing": "memfd",
                "runtime_dir": str(runtime_dir),
                "startup_timeout_s": 600,
            },
            "transport": {
                "intra_node": "shared_memory",
                "inter_node": "disabled",
                "staging_buffer_experts": 4,
                "timeout_s": 60,
            },
            "placement": {"scope": "ep_group", "policy": "stable"},
            "graph": {"mc2_decode": True},
            "num_device_experts": [84],
            "num_device_layers": 2,
            "cache_policy_enabled": True,
            "expert_prefetch_enabled": False,
            "prefill_delta_load": False,
            "prefill_split_gmm": False,
            "expert_substitution_enabled": False,
            "hot_expert_preload": False,
            "cpu_moe": {"enabled": False},
        },
    }


@pytest.mark.skipif(not (MODEL_PATH / "config.json").is_file(), reason="Local DeepSeek-V4 DSpark checkpoint required")
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
def test_dspark_deepseekv4_bf16_tp2_ep2_graph_matches_eager(tmp_path: Path):
    if not get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
        pytest.skip("BF16 SparseFlashMla requires Ascend A5")
    outputs_by_mode = []
    for graph_mode in ("NONE", "FULL_DECODE_ONLY"):
        with VllmRunner(
            str(MODEL_PATH),
            tokenizer_mode="deepseek_v4",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            block_size=128,
            max_model_len=4096,
            max_num_batched_tokens=4096,
            max_num_seqs=1,
            gpu_memory_utilization=0.90,
            kv_cache_dtype="bfloat16",
            seed=1024,
            enable_prefix_caching=False,
            disable_log_stats=False,
            enforce_eager=graph_mode == "NONE",
            speculative_config={"method": "dspark", "num_speculative_tokens": 7},
            compilation_config=CompilationConfig(
                cudagraph_mode=graph_mode,
                cudagraph_capture_sizes=[1, 2, 3, 4, 8, 16],
            ),
            additional_config=_offload_config(tmp_path / graph_mode),
        ) as runner:
            tokenizer = runner.model.get_tokenizer()
            seed_ids = tokenizer.encode(
                "Explain how paged attention handles a long context followed by a shorter request. ",
                add_special_tokens=False,
            )
            assert seed_ids
            generated = []
            for prompt_len, output_len in REQUEST_SIZES:
                token_ids = (seed_ids * ((prompt_len + len(seed_ids) - 1) // len(seed_ids)))[:prompt_len]
                response = runner.model.generate(
                    [TokensPrompt(prompt_token_ids=token_ids)],
                    SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True),
                    use_tqdm=False,
                )
                assert len(response) == 1 and response[0].finished
                actual_ids = list(response[0].outputs[0].token_ids)
                assert len(actual_ids) == output_len
                generated.append(actual_ids)
            drafts = sum(m.value for m in runner.model.get_metrics() if m.name == "vllm:spec_decode_num_drafts")
            assert drafts > 0, "The regression must exercise DSpark drafting"
            outputs_by_mode.append(generated)
    assert outputs_by_mode[1] == outputs_by_mode[0]
