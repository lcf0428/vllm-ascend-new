# SPDX-License-Identifier: Apache-2.0
"""Real process-shared native atomics; CPU-only and no NPU initialization."""

import ctypes
import multiprocessing
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.global_store.config import GlobalStoreConfig
from vllm_ascend.expert_offload.global_store.control import (
    CallbackWatchdog, SharedMemoryControl, build_control_library,
)

EXPERTS = 257
ROUNDS = 150


def _worker(rank, world, path, library, results, mode):
    torch.set_num_threads(1)
    control = SharedMemoryControl(path, library, rank, world, EXPERTS, 2)
    try:
        if mode == "mismatch":
            control.statistics(rank, torch.ones(EXPERTS, dtype=torch.int64))
        elif mode == "kind":
            control.ready(0) if rank else control.statistics(0, torch.ones(EXPERTS, dtype=torch.int64))
        elif mode == "scores":
            scores = torch.ones(EXPERTS) if rank else None
            control.statistics(0, torch.ones(EXPERTS, dtype=torch.int64), scores)
        else:
            for step in range(ROUNDS):
                # A deliberately slow reader/producer exercises payload reuse.
                if rank == 1 and step % 13 == 0:
                    time.sleep(0.002)
                counts = torch.arange(EXPERTS, dtype=torch.int64) + step + rank
                scores = (counts.float() / 8) if step % 2 else None
                actual, actual_scores = control.statistics(step % 43, counts, scores)
                expected = world * (torch.arange(EXPERTS) + step) + world * (world - 1) // 2
                assert torch.equal(actual, expected)
                if scores is not None:
                    assert torch.equal(actual_scores, expected.float() / 8)
                failures = control.ready(step % 43, "injected DMA failure" if step == 7 and rank == 1 else None)
                assert bool(failures) == (step == 7)
                if failures:
                    assert "rank 1: injected DMA failure" in failures[0]
            results.put((rank, "ok", control.sequence))
            return
        results.put((rank, "unexpected success"))
    except Exception as exc:
        results.put((rank, str(exc)))
    finally:
        control.close()


@pytest.mark.parametrize("world,mode", [(2, "normal"), (4, "normal"), (2, "mismatch"),
                                        (2, "kind"), (2, "scores")])
def test_real_process_collectives(tmp_path, world, mode):
    library = build_control_library(tmp_path)
    owner = SharedMemoryControl(tmp_path / "control.bin", library, 0, world, EXPERTS, 3, create=True)
    # Owner only creates storage; each child owns its own rank endpoint.
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    workers = [context.Process(target=_worker, args=(rank, world, str(owner.path), str(library), results, mode))
               for rank in range(world)]
    try:
        for process in workers:
            process.start()
        output = [results.get(timeout=45) for _ in workers]
        for process in workers:
            process.join(timeout=10)
            assert process.exitcode == 0
        if mode == "normal":
            assert all(item[1:] == ("ok", 2 * ROUNDS) for item in output), output
        else:
            assert all("operation mismatch" in item[1] for item in output), output
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join()
        owner.close()


def test_timeout_poison_is_visible_to_late_peer(tmp_path):
    library = build_control_library(tmp_path)
    first = SharedMemoryControl(tmp_path / "control.bin", library, 0, 2, EXPERTS, 0.03, create=True)
    second = SharedMemoryControl(first.path, library, 1, 2, EXPERTS, 1)
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            first.ready(0)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="aborted by rank 0"):
            second.ready(0)
        assert time.monotonic() - start < 0.2
    finally:
        second.close()
        first.close()


def _gil_peer(path, library, started, results):
    control = SharedMemoryControl(path, library, 1, 2, EXPERTS, 3)
    started.set()
    time.sleep(0.1)
    control.ready(0)
    results.put(time.monotonic())
    control.close()


def test_native_wait_releases_gil(tmp_path):
    library = build_control_library(tmp_path)
    first = SharedMemoryControl(tmp_path / "control.bin", library, 0, 2, EXPERTS, 3, create=True)
    context = multiprocessing.get_context("spawn")
    started, results = context.Event(), context.Queue()
    peer = context.Process(target=_gil_peer, args=(str(first.path), str(library), started, results))
    failures = []

    def wait():
        try:
            first.ready(0)
        except Exception as exc:
            failures.append(exc)

    peer.start()
    assert started.wait(30)
    waiter = threading.Thread(target=wait)
    waiter.start()
    time.sleep(0.01)
    # PyDLL deliberately retains the GIL; an ordinary Python poll cannot make
    # progress during this interval. The independent peer must still finish.
    hold = ctypes.PyDLL(None).usleep
    hold.argtypes = [ctypes.c_uint]
    hold(400_000)
    released = time.monotonic()
    try:
        assert results.get(timeout=5) < released - 0.1
        waiter.join(5)
        peer.join(5)
        assert not waiter.is_alive() and peer.exitcode == 0 and not failures
    finally:
        if peer.is_alive():
            peer.terminate()
            peer.join()
        first.close()


def test_invalid_input_and_close(tmp_path):
    library = build_control_library(tmp_path)
    control = SharedMemoryControl(tmp_path / "control.bin", library, 0, 1, EXPERTS, 1, create=True)
    with pytest.raises(ValueError, match="contiguous CPU"):
        control.statistics(0, torch.ones(EXPERTS, dtype=torch.int32))
    control.close()
    control.close()
    assert not control.path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        control.ready(0)


def test_bootstrap_rejects_invisible_peer_and_disagreement(tmp_path):
    config = GlobalStoreConfig.parse({
        "expert_offload": True, "enable_multi_card": True,
        "host_storage": {"mode": "global_sharded", "runtime_dir": str(tmp_path)},
        "transport": {"control_backend": "shared_memory"},
    })
    store = SimpleNamespace(config=config, regions={0: object()}, directory=SimpleNamespace(manifests={0: {}, 1: {}}))
    with pytest.raises(RuntimeError, match="same visible host"):
        SharedMemoryControl.bootstrap(store, EXPERTS, lambda item: [item, item])
    with pytest.raises(ValueError, match="disagree"):
        SharedMemoryControl.bootstrap(store, EXPERTS, lambda item: [item, ("gloo", EXPERTS, 60)])


def test_persistent_watchdog_rearms_and_closes():
    fired = threading.Event()
    watchdog = CallbackWatchdog(0.04, fired.set)
    for _ in range(4):
        watchdog.arm()
        watchdog.disarm()
    assert not fired.wait(0.06)
    watchdog.arm()
    assert fired.wait(1)
    watchdog.close()
    assert not watchdog._thread.is_alive()


def test_control_config_is_opt_in(tmp_path):
    assert GlobalStoreConfig.parse({}).transport["control_backend"] == "gloo"
    with pytest.raises(ValueError, match="control_backend"):
        GlobalStoreConfig.parse({"transport": {"control_backend": "auto"}})
    with pytest.raises(ValueError, match="global_sharded"):
        GlobalStoreConfig.parse({"transport": {"control_backend": "shared_memory"}})


def test_bootstrap_attachment_and_teardown(tmp_path):
    config = GlobalStoreConfig.parse({
        "expert_offload": True, "enable_multi_card": True,
        "host_storage": {"mode": "global_sharded", "runtime_dir": str(tmp_path)},
        "transport": {"control_backend": "shared_memory"},
    })
    # Real bootstrap barriers run on two Python threads here. Collectives are
    # native/GIL-free; process-shared visibility is covered separately above.
    barrier = threading.Barrier(2, timeout=10)
    values = [None, None]
    controls = [None, None]
    failures = []

    def start(rank):
        def gather(item):
            values[rank] = item
            barrier.wait()
            result = list(values)
            barrier.wait()
            return result

        store = SimpleNamespace(config=config, rank=rank, job_dir=tmp_path,
                                regions={0: object(), 1: object()},
                                directory=SimpleNamespace(manifests={0: {}, 1: {}}))
        try:
            controls[rank] = SharedMemoryControl.bootstrap(store, EXPERTS, gather)
            assert not controls[rank].ready(0)
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=start, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
    assert not failures and all(not thread.is_alive() for thread in threads), failures
    assert controls[0].path.stat().st_mode & 0o777 == 0o600
    for control in reversed(controls):
        control.close()
    assert not list(tmp_path.glob("control*"))
