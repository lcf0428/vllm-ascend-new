# SPDX-License-Identifier: Apache-2.0
"""Native pack threads, delayed DMA, physical bytes and quantization semantics."""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.global_store.config import GlobalStoreConfig
from vllm_ascend.expert_offload.global_store.directory import Directory
from vllm_ascend.expert_offload.global_store.native_staging import (
    NativeStagingTransport, build_native_staging, load_native_staging, staging_chunk_layout,
)
from vllm_ascend.expert_offload.global_store.shared_store import SharedRegion
from vllm_ascend.expert_offload.global_store.types import ExpertKey, Source


@pytest.fixture(scope="module")
def native_module(tmp_path_factory):
    return load_native_staging(build_native_staging(tmp_path_factory.mktemp("native-staging"), cpu_test=True))


def make_transport(tmp_path, native_module, budget=5, submit=2, threads=2, delay_us=2000):
    experts = []
    for eid in range(13):
        # Raw bytes stand for physical/NZ padding. A transposed quant field must
        # be reshaped and converted, rather than treated as a raw byte copy.
        fields = {"w13": torch.full((1024,), eid + 1, dtype=torch.uint8),
                  "w2": torch.full((512,), eid + 31, dtype=torch.uint8),
                  "scale": (torch.arange(12, dtype=torch.float32).reshape(3, 4) + eid).t()}
        experts.append((ExpertKey("test", "target", "layer", eid), fields))
    capacities, counts = staging_chunk_layout(experts, budget, submit)
    region, bundles = SharedRegion.publish(tmp_path / "owner.bin", 0, experts)
    config = SimpleNamespace(transport={"staging_pack_threads": threads, "timeout_s": 2})
    store = SimpleNamespace(config=config, regions={0: region},
                            directory=SimpleNamespace(bundles={b.key: b for b in bundles}))
    delegate = SimpleNamespace(close=lambda: None, synchronize=lambda stream: None)
    transport = NativeStagingTransport(store, torch.empty(sum(capacities), dtype=torch.uint8),
                                       delegate, capacities, counts, native_module,
                                       cpu_test=True, delay_us=delay_us)
    tasks, outputs = [], []
    for bundle, (_, fields) in zip(bundles, experts):
        output = {}
        for field in bundle.fields:
            dst = (torch.empty((2, 6), dtype=torch.float64) if field.name == "scale"
                   else torch.empty(field.nbytes, dtype=torch.uint8))
            output[field.name] = dst
            tasks.append(SimpleNamespace(source=Source(bundle, field), nbytes=field.nbytes,
                                         destination=dst if field.name == "scale" else dst.untyped_storage()))
        outputs.append((output, fields))
    return transport, region, tasks, outputs


@pytest.mark.parametrize("budget,submit,threads", [(5, 2, 2), (2, 1, 1), (7, 3, 4), (48, 2, 2)])
def test_delayed_dma_reuse_quant_and_repeated_batches(tmp_path, native_module, budget, submit, threads):
    transport, region, tasks, outputs = make_transport(tmp_path, native_module, budget, submit, threads)
    try:
        for _ in range(5):
            transport.copy_batch(tasks)
            transport.synchronize(None)
            for actual, expected in outputs:
                for name, value in expected.items():
                    torch.testing.assert_close(actual[name], value.reshape(actual[name].shape).to(actual[name].dtype))
        transport.copy_batch([])
        count = len(transport._destinations)
        # Different Python storage wrappers must reuse a bounded descriptor.
        for task in tasks:
            if not isinstance(task.destination, torch.Tensor):
                task.destination = task.destination[:]
        transport.copy_batch(tasks)
        transport.close()  # Drains pending fake DMA even without synchronize.
        assert len(transport._destinations) == 0 and count == len(tasks)
        for actual, expected in outputs:
            for name, value in expected.items():
                torch.testing.assert_close(actual[name], value.reshape(actual[name].shape).to(actual[name].dtype))
        transport.close()
        transport.synchronize(None)
        with pytest.raises(RuntimeError, match="closed"):
            transport.copy_batch(tasks)
    finally:
        transport.close()
        region.close()


def test_native_wait_releases_gil_and_first_submit_precedes_completion(tmp_path, native_module):
    transport, region, tasks, _ = make_transport(tmp_path, native_module, 2, 1, 2, delay_us=10000)
    progress = threading.Event()
    runner = threading.Thread(target=lambda: (time.sleep(0.02), progress.set()))
    runner.start()
    try:
        transport.copy_batch(tasks)
        stats = transport.last_stats
        assert progress.is_set(), "native waits retained Python GIL"
        assert stats["chunks"] == 13
        assert stats["first_submit_ns"] < stats["total_ns"] / 2
    finally:
        runner.join()
        transport.close()
        region.close()


def test_native_validation_poison_and_concurrent_call(tmp_path, native_module):
    transport, region, tasks, _ = make_transport(tmp_path, native_module)
    try:
        transport._lock.acquire()
        with pytest.raises(RuntimeError, match="Overlapping"):
            transport.copy_batch(tasks)
        transport._lock.release()
        tasks[0].nbytes += 1
        with pytest.raises(ValueError, match="mismatch"):
            transport.copy_batch(tasks)
        with pytest.raises(RuntimeError, match="poisoned"):
            transport.copy_batch([])
    finally:
        transport.close()
        region.close()


def test_native_submissions_preserve_stream_order_for_same_slot(tmp_path, native_module):
    transport, region, tasks, _ = make_transport(tmp_path, native_module, delay_us=1000)
    output = torch.empty(1024, dtype=torch.uint8)
    same_slot = [SimpleNamespace(source=t.source, destination=output.untyped_storage(), nbytes=t.nbytes)
                 for t in tasks if t.source.field.name == "w13"]
    try:
        for _ in range(10):
            transport.copy_batch(same_slot)
            transport.synchronize(None)
            assert bool(torch.all(output == 13))
    finally:
        transport.close()
        region.close()


def test_native_partial_submission_failure_drains_and_poison(native_module):
    staging = torch.empty(256, dtype=torch.uint8)
    engine = native_module.Pipeline(staging, [128, 128], [1, 1], 2, 2.0, 1000)
    source = torch.arange(16, dtype=torch.float32)
    raw = torch.zeros(64, dtype=torch.uint8)
    sid = engine.add_source(source.data_ptr(), 64, torch.empty(0), [16], [1])
    raw_id = engine.add_destination(torch.empty(0), raw.data_ptr(), 64)
    bad_id = engine.add_destination(torch.zeros(3), 0, 64)  # Impossible reshape.
    engine.run([sid, sid], [raw_id, bad_id], [2], 0)
    with pytest.raises(RuntimeError, match="shape"):
        engine.synchronize()
    # Native CPU simulator errors are reported after joining the DMA thread.
    assert bytes(raw.untyped_storage()) == bytes(source.untyped_storage())
    with pytest.raises(RuntimeError, match="poisoned"):
        engine.run([], [], [], 0)
    engine.close()


def test_directory_sources_are_stable_and_source_bounds_checked(tmp_path, native_module):
    key = ExpertKey("test", "target", "layer", 0)
    fields = {"w13": torch.zeros(64, dtype=torch.uint8), "w2": torch.zeros(64, dtype=torch.uint8)}
    region, bundles = SharedRegion.publish(tmp_path / "stable.bin", 0, [(key, fields)])
    directory = Directory([{"rank": 0, "size": region.size, "bundles": bundles}], [key])
    assert directory.source(key, "w13") is directory.source(key, "w13")
    engine = native_module.Pipeline(torch.empty(128, dtype=torch.uint8), [64, 64], [1, 1], 1, 1.0, 0)
    try:
        with pytest.raises(RuntimeError, match="exceeds field"):
            engine.add_source(region.address, 64, torch.empty(0), [1024], [1])
    finally:
        engine.close()
        region.close()


def test_native_config_and_odd_budget():
    base = {"expert_offload": True, "enable_multi_card": True,
            "host_storage": {"mode": "global_sharded", "runtime_dir": "/tmp/test"}}
    assert GlobalStoreConfig.parse(base).transport["staging_backend"] == "torch"
    for options in ({"staging_backend": "native", "staging_buffer_experts": 2},
                    {"staging_backend": "bad"}, {"staging_pack_threads": True},
                    {"staging_pack_threads": 65}, {"staging_submit_experts": 0},
                    {"staging_backend": "native", "inter_node": "tcp"}):
        with pytest.raises(ValueError):
            GlobalStoreConfig.parse(dict(base, transport=options))
    cfg = GlobalStoreConfig.parse(dict(base, transport={"staging_backend": "native",
                                                       "staging_buffer_experts": 5}))
    assert cfg.transport["staging_submit_experts"] == 2
