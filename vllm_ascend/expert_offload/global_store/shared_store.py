"""File-backed MAP_SHARED region; no process-private pointer exchange."""

import ctypes
import mmap
import os
from pathlib import Path

from .types import Bundle, Field

ALIGNMENT = 64


class SharedRegion:
    def __init__(self, path, size, *, create=False, backing="file"):
        self.path = Path(path)
        self.size = size
        self.owner_fd = None
        memory_backed = create and backing == "memfd"
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        if memory_backed:
            fd = os.memfd_create("vllm-expert-owner", os.MFD_CLOEXEC)
            self.owner_fd = fd
            self.path = Path(f"/proc/{os.getpid()}/fd/{fd}")
        else:
            fd = os.open(self.path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, size)
                # Fail before conversion publication if the backing FS is full.
                os.posix_fallocate(fd, 0, size)
            stat = os.fstat(fd)
            self.identity = (stat.st_dev, stat.st_ino)
            if stat.st_size != size:
                raise ValueError("Shared expert region size changed")
            self.mapping = mmap.mmap(fd, size, flags=mmap.MAP_SHARED)
        except BaseException:
            if create and not memory_backed:
                self.path.unlink(missing_ok=True)
            if self.owner_fd is not None:
                self.owner_fd = None
                os.close(fd)
            raise
        finally:
            if not memory_backed:
                os.close(fd)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))

    @classmethod
    def publish(cls, path, owner, experts, *, backing="file"):
        """experts = [(ExpertKey, {field_name: final CPU tensor})]."""
        offset = 0
        bundles = []
        writes = []
        for key, tensors in experts:
            fields = []
            for name, tensor in sorted(tensors.items()):
                if tensor.device.type != "cpu" or tensor.storage_offset() != 0:
                    raise ValueError("Expert publication requires whole CPU allocations")
                offset = (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
                storage = tensor.untyped_storage()
                nbytes = storage.nbytes()
                fields.append(Field(name, offset, nbytes, tuple(tensor.shape),
                                    tuple(tensor.stride()), str(tensor.dtype)))
                writes.append((offset, nbytes, storage.data_ptr()))
                offset += nbytes
            bundles.append(Bundle(key, owner, tuple(fields)))
        region = cls(path, max(offset, 1), create=True, backing=backing)
        for offset, nbytes, pointer in writes:
            ctypes.memmove(region.address + offset, pointer, nbytes)
        return region, tuple(bundles)

    def read_into(self, offset, destination, nbytes):
        if offset < 0 or nbytes < 0 or offset + nbytes > self.size:
            raise ValueError("Expert region read out of bounds")
        if nbytes > len(destination):
            raise ValueError("Expert read destination too small")
        destination[:nbytes] = memoryview(self.mapping)[offset:offset + nbytes]

    def close(self):
        self.mapping.close()
        if self.owner_fd is not None:
            os.close(self.owner_fd)
            self.owner_fd = None
