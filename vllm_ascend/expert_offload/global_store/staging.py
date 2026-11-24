"""One bounded pinned staging allocation per rank, serial reuse after DMA."""

import ctypes

import torch

from .types import Source


class StagingTransport:
    supports_remote_sources = True

    def __init__(self, store, staging, delegate):
        self.store = store
        self.staging = staging
        self.delegate = delegate
        self.buffer = (ctypes.c_ubyte * staging.numel()).from_address(staging.data_ptr())
        self.view = memoryview(self.buffer).cast("B")

    def allocate_host_tensor(self, shape, dtype):
        return self.delegate.allocate_host_tensor(shape, dtype)

    def copy_batch(self, tasks):
        for task in tasks:
            if not isinstance(task.source, Source):
                self.delegate.copy_batch([task])
                continue
            source = task.source
            if source.nbytes != task.nbytes:
                raise ValueError(f"Expert source size mismatch: {task.name}")
            if isinstance(task.destination, torch.Tensor):
                # Quant attributes are copied logically, including strided
                # destinations and possible dtype conversion, like legacy copy_.
                if source.nbytes > self.staging.numel():
                    raise ValueError("Quant field exceeds staging_buffer_mb; increase staging capacity")
                self.store.read_into(source, 0, self.view, source.nbytes)
                dtype = getattr(torch, source.field.dtype.removeprefix("torch."))
                raw = self.staging[:source.nbytes].view(dtype)
                cpu = raw.as_strided(source.field.shape, source.field.stride)
                task.destination.copy_(cpu.reshape(task.destination.shape), non_blocking=True)
                self.synchronize(torch.npu.current_stream())
            else:
                # Packed weights preserve the full NZ/raw allocation. A field
                # may be larger than staging; never allocate a full replica.
                for offset in range(0, source.nbytes, self.staging.numel()):
                    size = min(self.staging.numel(), source.nbytes - offset)
                    self.store.read_into(source, offset, self.view, size)
                    task.destination[offset:offset + size].copy_(
                        self.staging.untyped_storage()[:size], non_blocking=True)
                    self.synchronize(torch.npu.current_stream())

    def synchronize(self, stream):
        self.delegate.synchronize(stream)

    def close(self):
        self.view.release()
        self.buffer = None
        self.delegate.close()
