# SPDX-License-Identifier: Apache-2.0
"""Delayed DMA regression tests: staging must not be overwritten in flight."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.expert_offload.global_store.config import GlobalStoreConfig
from vllm_ascend.expert_offload.global_store.staging import StagingTransport, staging_buffer_sizes
from vllm_ascend.expert_offload.global_store.transport_rdma import Endpoint, RDMAReader
from vllm_ascend.expert_offload.global_store.types import Bundle, ExpertKey, Field, Source


class DelayedStream:
    def __init__(self):
        self.queue = []
        self.completed = 0
        self.waits = 0
        self.records = 0

    def finish(self, end):
        while self.completed < end:
            self.queue[self.completed]()
            self.completed += 1

    def synchronize(self):
        self.finish(len(self.queue))


class DelayedEvent:
    def record(self, stream):
        self.stream = stream
        self.end = len(stream.queue)
        stream.records += 1

    def query(self):
        return self.stream.completed >= self.end

    def synchronize(self):
        self.stream.waits += 1
        self.stream.finish(self.end)


class DelayedDestination:
    def __init__(self, tensor, stream):
        self.tensor = tensor
        self.stream = stream

    def __getitem__(self, span):
        return DelayedDestination(self.tensor[span], self.stream)

    def copy_(self, source, non_blocking):
        assert non_blocking
        # Read source bytes only when the fake DMA completes, never at submit.
        def complete_copy():
            self.tensor.copy_(torch.tensor(list(source), dtype=torch.uint8))

        self.stream.queue.append(complete_copy)


class Delegate:
    def __init__(self):
        self.syncs = 0
        self.closed = False
        self.tasks = []

    def synchronize(self, stream):
        self.syncs += 1
        stream.synchronize()

    def copy_batch(self, tasks):
        self.tasks.extend(tasks)

    def close(self):
        self.closed = True


class Store:
    def __init__(self):
        self.reads = []

    def read_into(self, source, offset, target, size):
        self.reads.append((source.bundle.key.expert, source.field.name, offset, size))
        target[:size] = bytes([source.bundle.key.expert + 1]) * size


def task(eid, size, stream, name="w13"):
    key = ExpertKey("test", "target", "layer", eid)
    field = Field(name, 0, size, (size,), (1,), "torch.uint8")
    source = Source(Bundle(key, 0, (field,)), field)
    output = torch.zeros(size, dtype=torch.uint8)
    return SimpleNamespace(source=source, destination=DelayedDestination(output, stream),
                           nbytes=size, name=name), output


def fake_npu(stream):
    return patch.object(torch, "npu", SimpleNamespace(current_stream=lambda: stream,
                                                     Event=DelayedEvent), create=True)


def test_cpu_fill_overlaps_dma_and_waits_only_on_bank_reuse():
    stream, store, delegate = DelayedStream(), Store(), Delegate()
    transport = StagingTransport(store, torch.empty(512, dtype=torch.uint8), delegate)
    tasks, outputs = zip(*(task(i, 64, stream) for i in range(12)))
    with fake_npu(stream):
        transport.copy_batch(tasks)
        # Four fields fit each bank: 12 field copies need three bank submissions,
        # one reuse wait and no stream/field synchronization during submission.
        assert stream.records == 3
        assert stream.waits == 1
        assert delegate.syncs == 0
        assert stream.completed == 4
        assert len(store.reads) == 12
        assert any(bank.references for bank in transport.banks)
        transport.synchronize(stream)
    for i, output in enumerate(outputs):
        assert torch.equal(output, torch.full((64,), i + 1, dtype=torch.uint8))
    assert not any(bank.references or bank.in_flight for bank in transport.banks)
    transport.close()
    transport.close()
    assert delegate.closed
    with pytest.raises(RuntimeError, match="closed"):
        transport.copy_batch(tasks)


def test_repeated_batches_chunking_and_close_preserve_pending_dma():
    stream, store, delegate = DelayedStream(), Store(), Delegate()
    transport = StagingTransport(store, torch.empty(256, dtype=torch.uint8), delegate)
    outputs = []
    with fake_npu(stream):
        for i in range(5):
            item, output = task(i, 320, stream)
            outputs.append(output)
            transport.copy_batch([item])
        # close must drain even when the caller has not explicitly synchronized.
        transport.close()
    assert len(store.reads) == 15
    for i, output in enumerate(outputs):
        assert torch.equal(output, torch.full((320,), i + 1, dtype=torch.uint8))


def test_mixed_tasks_and_failure_keep_prior_submissions_alive():
    stream, store, delegate = DelayedStream(), Store(), Delegate()
    transport = StagingTransport(store, torch.empty(128, dtype=torch.uint8), delegate)
    first, output = task(0, 64, stream)
    other = SimpleNamespace(source=object())
    bad, _ = task(1, 64, stream)
    bad.nbytes = 1
    with fake_npu(stream):
        with pytest.raises(ValueError, match="size mismatch"):
            transport.copy_batch([first, other, bad])
        assert delegate.tasks == [other]
        assert stream.completed == 0
        transport.close()
    assert output.tolist() == [1] * 64


def test_partial_copy_failure_records_event_and_retains_sources():
    stream, delegate = DelayedStream(), Delegate()
    transport = StagingTransport(Store(), torch.empty(256, dtype=torch.uint8), delegate)
    first, output = task(0, 64, stream)
    bad, _ = task(1, 64, stream)
    bad.destination.copy_ = Mock(side_effect=RuntimeError("copy failure"))
    # Fail after the first operation has actually been enqueued.
    original = transport._copy

    def copy(item, *args):
        if item is bad:
            raise RuntimeError("copy failure")
        original(item, *args)

    with fake_npu(stream), patch.object(transport, "_copy", side_effect=copy):
        with pytest.raises(RuntimeError, match="copy failure"):
            transport.copy_batch([first, bad])
        assert stream.records == 1
        assert transport.banks[0].references
        transport.close()
    assert output.tolist() == [1] * 64


def test_budget_uses_largest_physical_expert_all_fields_and_odd_split():
    padded = torch.empty(2, dtype=torch.float32)
    padded.set_(torch.empty(512, dtype=torch.uint8).untyped_storage(), 0, (2,), (1,))
    experts = [(None, {"w13": torch.empty(65, dtype=torch.uint8),
                       "w2": torch.empty(16, dtype=torch.uint8)}),
               (None, {"w13": padded, "scale": torch.empty(3),
                       "offset": torch.empty(2, dtype=torch.int64)})]
    # max(128+64, 512+64+64) = 640 bytes, independent of logical numel.
    assert staging_buffer_sizes(experts, 4) == (1280, 1280)
    assert staging_buffer_sizes(experts, 3) == (640, 1280)
    with pytest.raises(ValueError, match="nonempty"):
        staging_buffer_sizes([], 4)
    for invalid in (True, 1, 0, -1, 2.5):
        with pytest.raises(ValueError, match="staging_buffer_experts"):
            staging_buffer_sizes(experts, invalid)


def test_config_migration_and_expert_count_validation():
    assert GlobalStoreConfig.parse({}).transport["staging_buffer_experts"] == 4
    assert GlobalStoreConfig.parse({"transport": {"staging_buffer_experts": 3}}).transport[
        "staging_buffer_experts"] == 3
    for invalid in (True, 1, 0, -2, 2.5, "4"):
        with pytest.raises(ValueError, match="staging_buffer_experts"):
            GlobalStoreConfig.parse({"transport": {"staging_buffer_experts": invalid}})
    with pytest.raises(ValueError, match="was replaced"):
        GlobalStoreConfig.parse({"transport": {"staging_buffer_mb": 64}})


def test_rdma_respects_bank_and_field_offsets():
    staging = torch.empty(512, dtype=torch.uint8)
    reader = object.__new__(RDMAReader)
    reader.staging = staging
    reader.timeout = 5
    reader.peers = {1: (object(), Endpoint(), None)}
    call = Mock(return_value=0)
    reader.verbs = SimpleNamespace(lib=SimpleNamespace(expert_rdma_read_into=call))
    buffer = (ctypes.c_ubyte * staging.numel()).from_address(staging.data_ptr())
    view = memoryview(buffer).cast("B")
    target = view[320:384]
    try:
        reader.read_into(1, {}, 1234, target, 64)
        assert call.call_args.args[2:] == (1234, 320, 64, 5)
        with pytest.raises(ValueError, match="outside"):
            reader.read_into(1, {}, 0, memoryview(bytearray(64)), 64)
        with pytest.raises(ValueError, match="size"):
            reader.read_into(1, {}, 0, target, 65)
    finally:
        target.release()
        view.release()
