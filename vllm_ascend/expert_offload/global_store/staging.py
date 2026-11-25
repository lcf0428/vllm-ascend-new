"""Bounded ping-pong Host staging; only wait when a DMA bank is reused."""

import ctypes
from dataclasses import dataclass, field

import torch

from .types import Source

STAGING_ALIGNMENT = 64
STAGING_BANK_COUNT = 2
MAX_RDMA_READ_BYTES = (1 << 32) - 1


def _aligned_size(size):
    return (size + STAGING_ALIGNMENT - 1) // STAGING_ALIGNMENT * STAGING_ALIGNMENT


def staging_buffer_sizes(experts, expert_count):
    """Return two bank sizes, budgeting complete physical experts per rank.

    Use the largest published expert across all layers, including every quant
    attribute and raw/NZ storage padding. Each field gets its own aligned range.
    Odd budgets give the second bank one additional expert's worth of space.
    """
    if type(expert_count) is not int or expert_count < STAGING_BANK_COUNT:
        raise ValueError("staging_buffer_experts must be an integer >= 2 (two banks total)")
    expert_bytes = max((sum(_aligned_size(t.untyped_storage().nbytes()) for t in tensors.values())
                        for _, tensors in experts), default=0)
    if not expert_bytes:
        raise ValueError("Cannot size staging without nonempty expert allocations")
    first = expert_count // STAGING_BANK_COUNT
    return expert_bytes * first, expert_bytes * (expert_count - first)


@dataclass
class _Bank:
    offset: int
    capacity: int
    event: object = None
    stream: object = None
    in_flight: bool = False
    references: list = field(default_factory=list)


class StagingTransport:
    supports_remote_sources = True

    def __init__(self, store, staging, delegate, bank_sizes=None):
        self.store = store
        self.staging = staging
        self.delegate = delegate
        if bank_sizes is None:
            first = staging.numel() // (STAGING_BANK_COUNT * STAGING_ALIGNMENT) * STAGING_ALIGNMENT
            bank_sizes = (first, staging.numel() - first)
        if (len(bank_sizes) != STAGING_BANK_COUNT or sum(bank_sizes) != staging.numel()
                or any(size <= 0 or size % STAGING_ALIGNMENT for size in bank_sizes)):
            raise ValueError("Staging requires two positive, 64-byte-aligned banks")
        self.banks = [_Bank(0, bank_sizes[0]), _Bank(bank_sizes[0], bank_sizes[1])]
        self._next_bank = 0
        self._closed = False
        self.buffer = (ctypes.c_ubyte * staging.numel()).from_address(staging.data_ptr())
        self.view = memoryview(self.buffer).cast("B")

    def allocate_host_tensor(self, shape, dtype):
        return self.delegate.allocate_host_tensor(shape, dtype)

    def _acquire_bank(self, stream):
        bank = self.banks[self._next_bank]
        self._next_bank = (self._next_bank + 1) % STAGING_BANK_COUNT
        if bank.in_flight:
            # Never overwrite pinned bytes while H2D or a logical quant copy
            # still reads them. Waiting on the compute/callback stream deadlocks.
            if not bank.event.query():
                bank.event.synchronize()
            bank.in_flight = False
        bank.references.clear()
        bank.stream = stream
        if bank.event is None:
            bank.event = torch.npu.Event()
        return bank

    def _copy(self, task, source_offset, staging_offset, size, bank):
        if isinstance(task.destination, torch.Tensor):
            dtype = getattr(torch, task.source.field.dtype.removeprefix("torch."))
            raw = self.staging[staging_offset:staging_offset + size].view(dtype)
            cpu = raw.as_strided(task.source.field.shape, task.source.field.stride)
            cpu = cpu.reshape(task.destination.shape)
            # reshape may own a new allocation. Keep all logical sources and
            # destinations alive until the bank's completion event has fired.
            bank.references.extend((cpu, task.destination))
            task.destination.copy_(cpu, non_blocking=True)
        else:
            raw = self.staging.untyped_storage()[staging_offset:staging_offset + size]
            destination = task.destination[source_offset:source_offset + size]
            bank.references.extend((raw, destination))
            destination.copy_(raw, non_blocking=True)

    def _submit(self, bank, copies):
        try:
            for task, source_offset, staging_offset, size in copies:
                self._copy(task, source_offset, staging_offset, size, bank)
        finally:
            # Also protect partially submitted batches on copy failure.
            bank.event.record(bank.stream)
            bank.in_flight = True

    def copy_batch(self, tasks):
        if self._closed:
            raise RuntimeError("Staging transport is closed")
        stream = None
        bank = None
        used = 0
        copies = []
        for task in tasks:
            if not isinstance(task.source, Source):
                if copies:
                    self._submit(bank, copies)
                    bank, copies, used = None, [], 0
                self.delegate.copy_batch([task])
                continue
            source = task.source
            if source.nbytes != task.nbytes:
                raise ValueError(f"Expert source size mismatch: {task.name}")
            if not source.nbytes:
                continue
            logical = isinstance(task.destination, torch.Tensor)
            # Normal expert-sized banks fit whole fields. Keep raw chunking for
            # callers supplying smaller custom allocations (e.g. hardware tests).
            if logical and source.nbytes > min(b.capacity for b in self.banks):
                raise ValueError("Quant field exceeds staging bank; increase staging_buffer_experts")
            if stream is None:
                stream = torch.npu.current_stream()
            offset = 0
            while offset < source.nbytes:
                if bank is None:
                    bank = self._acquire_bank(stream)
                start = _aligned_size(used)
                available = bank.capacity - start
                if available <= 0 or ((logical or source.nbytes <= bank.capacity)
                                      and source.nbytes - offset > available):
                    self._submit(bank, copies)
                    bank, copies, used = None, [], 0
                    continue
                size = min(source.nbytes - offset, available)
                position = bank.offset + start
                target = self.view[position:position + size]
                try:
                    self.store.read_into(source, offset, target, size)
                finally:
                    target.release()
                copies.append((task, offset, position, size))
                used = start + size
                offset += size
        if copies:
            self._submit(bank, copies)

    def synchronize(self, stream):
        self.delegate.synchronize(stream)
        for bank in self.banks:
            if bank.stream == stream:
                bank.in_flight = False
                bank.references.clear()

    def close(self):
        if self._closed:
            return
        for bank in self.banks:
            if bank.in_flight:
                bank.event.synchronize()
                bank.in_flight = False
            bank.references.clear()
        self.delegate.close()
        self.view.release()
        self.buffer = None
        self._closed = True
