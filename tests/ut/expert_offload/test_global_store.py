"""CPU-only coverage; safe to run with pytest --noconftest on a busy NPU host."""

import gc
import multiprocessing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.expert_offload.global_store.admission import route_upper_bound
from vllm_ascend.expert_offload.global_store.config import GlobalStoreConfig
from vllm_ascend.expert_offload.global_store.directory import Directory
from vllm_ascend.expert_offload.global_store.shared_store import SharedRegion
from vllm_ascend.expert_offload.global_store.staging import StagingTransport
from vllm_ascend.expert_offload.global_store.store import GlobalExpertStore
from vllm_ascend.expert_offload.global_store.transport_tcp import TCPReader, TCPServer
from vllm_ascend.expert_offload.global_store.transport_rdma import Endpoint, Verbs
from vllm_ascend.expert_offload.global_store.types import ExpertKey


def config(path, protocol="disabled"):
    return GlobalStoreConfig.parse({
        "expert_offload": True, "enable_multi_card": True,
        "host_storage": {"mode": "global_sharded", "runtime_dir": str(path)},
        "transport": {"inter_node": protocol, "bind_address": "127.0.0.1",
                      "advertise_address": "127.0.0.1", "staging_buffer_mb": 1,
                      "timeout_s": 5},
        "graph": {"mc2_decode": True},
    })


def fields(eid):
    return {"w13": torch.arange(128, dtype=torch.uint8) + eid,
            "w2": torch.arange(64, dtype=torch.uint8) + 2 * eid,
            "w13_weight_scale": torch.arange(12, dtype=torch.float32).reshape(3, 4).t() + eid,
            "w2_scale_bias": torch.arange(4, dtype=torch.int64) + eid}


def test_config_is_opt_in_and_strict(tmp_path):
    assert not GlobalStoreConfig.parse({}).enabled
    assert config(tmp_path).enabled
    for section, value in [("transport", {"inter_node": "auto"}),
                           ("placement", {"scope": "node"}),
                           ("graph", {"mc2_decode": 1}),
                           ("host_storage", {"mode": "global_sharded", "runtime_dir": "relative"}),
                           ("transport", {"staging_buffer_mb": 0}),
                           ("transport", {"typo": 1})]:
        with pytest.raises((TypeError, ValueError)):
            GlobalStoreConfig.parse({section: value})
    with pytest.raises(ValueError, match="shard_per_rank"):
        GlobalStoreConfig.parse({"expert_offload": True, "enable_multi_card": True,
                                "shard_per_rank": False,
                                "host_storage": {"mode": "global_sharded", "runtime_dir": str(tmp_path)}})


def test_admission_sums_independent_dp_rows():
    assert route_upper_bound(8, 4, 6, [8, 0, 0, 0]) == 48
    assert route_upper_bound(8, 4, 6, [8, 8, 8, 8]) == 192
    assert route_upper_bound(8, 4, 6) == 192
    with pytest.raises(ValueError):
        route_upper_bound(8, 4, 6, [9, 0, 0, 0])


def test_rdma_requires_explicit_library_and_endpoint_abi_roundtrips():
    with pytest.raises(ValueError, match="rdma_library"):
        Verbs({"rdma_library": None})
    endpoint = Endpoint(address=123, size=1024, rkey=42, qpn=17, psn=0, lid=2, mtu=3)
    restored = Endpoint.deserialize(endpoint.serialize())
    assert bytes(restored) == bytes(endpoint)
    with pytest.raises(ValueError, match="ABI"):
        Endpoint.deserialize("00")


def test_directory_rejects_missing_duplicate_and_overlap(tmp_path):
    key = ExpertKey("model", "target", "layer.0", 0)
    region, bundles = SharedRegion.publish(tmp_path / "weights", 0, [(key, fields(0))])
    manifest = {"rank": 0, "size": region.size, "bundles": bundles}
    assert Directory([manifest], [key]).source(key, "w13").nbytes == 128
    with pytest.raises(ValueError, match="Incomplete"):
        Directory([manifest], [key, replace(key, expert=1)])
    with pytest.raises(ValueError, match="Duplicate"):
        Directory([manifest, manifest], [key])
    overlapping = replace(bundles[0], fields=tuple(replace(f, offset=0) for f in bundles[0].fields))
    with pytest.raises(ValueError, match="Overlapping"):
        Directory([dict(manifest, bundles=[overlapping])], [key])
    region.close()


def test_shared_mapping_and_tcp_are_byte_exact(tmp_path):
    key = ExpertKey("model", "target", "layer.0", 0)
    region, bundles = SharedRegion.publish(tmp_path / "weights", 0, [(key, fields(0))])
    peer = SharedRegion(region.path, region.size)
    assert peer.address != region.address
    data = bytearray(region.size)
    peer.read_into(0, memoryview(data), len(data))
    assert data == region.mapping[:]
    with pytest.raises(ValueError):
        peer.read_into(region.size, memoryview(data), 1)
    server = TCPServer(region, "127.0.0.1", 2, 17)
    try:
        reader = TCPReader(2)
        endpoint = {"address": "127.0.0.1", "port": server.port, "token": server.token}
        for start in range(0, region.size, 17):
            size = min(17, region.size - start)
            target = bytearray(size)
            reader.read_into(endpoint, start, memoryview(target), size)
            assert target == data[start:start + size]
        with pytest.raises(ConnectionError):
            reader.read_into(dict(endpoint, token="invalid"), 0, memoryview(bytearray(1)), 1)
    finally:
        server.close()
        peer.close()
        region.close()


def _store_worker(rank, connection, root, protocol, force_remote):
    try:
        import vllm_ascend.expert_offload.global_store.store as store_module

        if force_remote:
            store_module.can_map = lambda _: False

        def gather(value):
            connection.send(("gather", value))
            return connection.recv()

        key = ExpertKey("model", "target", "layer.0", rank)
        private = fields(rank)
        staging = torch.empty(32, dtype=torch.uint8)
        store = GlobalExpertStore(config(Path(root), protocol), rank, 2, gather,
                                  "model", ["layer.0"], [(key, private)], staging)
        # Owner metadata must point into the same canonical physical region.
        owner = store.source(0, rank, "w13")
        owner_view = store.local_tensor(owner)
        assert owner_view.data_ptr() == store.regions[rank].address + owner.field.offset
        private.clear()
        for eid in range(2):
            for name, tensor in fields(eid).items():
                source = store.source(0, eid, name)
                target = bytearray(source.nbytes)
                for start in range(0, source.nbytes, staging.numel()):
                    size = min(staging.numel(), source.nbytes - start)
                    chunk = bytearray(size)
                    store.read_into(source, start, memoryview(chunk), size)
                    target[start:start + size] = chunk
                assert target == bytes(tensor.untyped_storage())
        del owner_view
        gc.collect()
        store.close()
        connection.send(("done", None))
    except BaseException:
        import traceback

        connection.send(("error", traceback.format_exc()))


@pytest.mark.parametrize("protocol,force_remote", [("disabled", False), ("tcp", True)])
def test_two_process_single_copy_and_cross_owner_read(tmp_path, protocol, force_remote):
    context = multiprocessing.get_context("spawn")
    parents, processes = [], []
    for rank in range(2):
        parent, child = context.Pipe()
        process = context.Process(target=_store_worker,
                                  args=(rank, child, str(tmp_path), protocol, force_remote))
        process.start()
        parents.append(parent)
        processes.append(process)
        child.close()
    try:
        while True:
            reports = []
            for parent in parents:
                assert parent.poll(30), "Expert worker timed out"
                reports.append(parent.recv())
            errors = [value for kind, value in reports if kind == "error"]
            assert not errors, "\n".join(errors)
            if all(kind == "done" for kind, _ in reports):
                break
            assert all(kind == "gather" for kind, _ in reports)
            for parent in parents:
                parent.send([value for _, value in reports])
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert not list(tmp_path.rglob("owner-*.bin"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        for parent in parents:
            parent.close()


def test_staging_chunks_weights_and_copies_strided_quant_values(tmp_path):
    staging = torch.empty(32, dtype=torch.uint8)

    class Delegate:
        def synchronize(self, stream):
            pass

    key = ExpertKey("model", "target", "layer.0", 0)
    region, bundles = SharedRegion.publish(tmp_path / "weights", 0, [(key, fields(0))])
    directory = Directory([{"rank": 0, "size": region.size, "bundles": bundles}], [key])

    class Store:
        def read_into(self, source, offset, destination, size):
            region.read_into(source.field.offset + offset, destination, size)

    transport = StagingTransport(Store(), staging, Delegate())
    weight = torch.zeros(128, dtype=torch.uint8)
    task = SimpleNamespace(source=directory.source(key, "w13"), destination=weight.untyped_storage(),
                           nbytes=128, name="w13")
    with patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: None), create=True):
        transport.copy_batch([task])
    assert torch.equal(weight, fields(0)["w13"])
    # A field larger than staging is rejected before any quant copy.
    task = SimpleNamespace(source=directory.source(key, "w13_weight_scale"),
                           destination=torch.empty(3, 4), nbytes=48, name="scale")
    with pytest.raises(ValueError, match="Quant field"):
        transport.copy_batch([task])
    transport.staging = torch.empty(64, dtype=torch.uint8)
    transport.view.release()
    transport.__init__(Store(), transport.staging, Delegate())
    expected = fields(0)["w13_weight_scale"]
    destination = torch.empty(3, 4).t()
    task = SimpleNamespace(source=directory.source(key, "w13_weight_scale"),
                           destination=destination, nbytes=48, name="scale")
    with patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: None), create=True):
        transport.copy_batch([task])
    assert torch.equal(destination, expected)
    transport.view.release()
    region.close()
