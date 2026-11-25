# SPDX-License-Identifier: Apache-2.0
"""Opt-in same-host EP collectives; native polling releases Python's GIL.

Only bootstrap/teardown use the supplied object collective. Every rank must
call statistics and ready in the same order. A read acknowledgement protects
each payload until all peers have consumed it, including across layer changes.
"""

import ctypes
import hashlib
import mmap
import os
import subprocess
import threading
import time
from pathlib import Path

import torch

MIN_PAYLOAD_BYTES = 4096
ERROR_BYTES = 4096
STATISTIC_BYTES_PER_EXPERT = 12
MAX_SEQUENCE = (1 << 64) - 1
BUILD_TIMEOUT_S = 60


def build_control_library(directory):
    """Build a tiny CPU-only DSO in this service's private startup directory."""
    source = Path(__file__).with_suffix(".c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library = Path(directory) / f"control-{digest}.so"
    if not library.exists():
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", str(source), "-o", str(library)],
                       check=True, capture_output=True, timeout=BUILD_TIMEOUT_S)
        library.chmod(0o600)
    return library


def _load_library(path):
    # CDLL releases the GIL during each native call, including peer waits.
    library = ctypes.CDLL(str(path))
    pointer, number = ctypes.c_void_p, ctypes.c_uint64
    library.expert_control_size.argtypes = [number, number]
    library.expert_control_size.restype = number
    library.expert_control_init.argtypes = [pointer, number, number]
    library.expert_control_validate.argtypes = [pointer, number, number]
    library.expert_control_abort.argtypes = [pointer, number, ctypes.c_char_p]
    library.expert_control_abort.restype = None
    library.expert_control_statistics.argtypes = [
        pointer, number, number, ctypes.c_int64, number, pointer, pointer,
        pointer, pointer, number, pointer, number,
    ]
    library.expert_control_ready.argtypes = [
        pointer, number, number, ctypes.c_int64, ctypes.c_char_p, number, pointer, number,
    ]
    return library


class SharedMemoryControl:
    def __init__(self, path, library_path, rank, world_size, num_experts, timeout, *, create=False):
        self.path = Path(path)
        self.library_path = Path(library_path)
        self.library = _load_library(library_path)
        self.rank = rank
        self.world_size = world_size
        self.num_experts = num_experts
        self.timeout_ns = int(timeout * 1_000_000_000)
        self.capacity = max(MIN_PAYLOAD_BYTES, num_experts * STATISTIC_BYTES_PER_EXPERT)
        if not 0 <= rank < world_size or num_experts <= 0 or self.timeout_ns <= 0:
            raise ValueError("Invalid shared expert control dimensions or timeout")
        self.size = self.library.expert_control_size(world_size, self.capacity)
        if not self.size:
            raise ValueError("Shared expert control dimensions exceed native limits")
        self.sequence = 0
        self.closed = False
        self._lock = threading.Lock()
        self._owns_files = create
        self._owns_library = False
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, self.size)
            if os.fstat(fd).st_size != self.size:
                raise ValueError("Shared expert control mapping size mismatch")
            self.mapping = mmap.mmap(fd, self.size)
        finally:
            os.close(fd)
        self._buffer = (ctypes.c_char * self.size).from_buffer(self.mapping)
        self.address = ctypes.addressof(self._buffer)
        function = self.library.expert_control_init if create else self.library.expert_control_validate
        if function(self.address, world_size, self.capacity):
            self.close()
            raise ValueError("Shared expert control header mismatch")
        self._error = ctypes.create_string_buffer(ERROR_BYTES)

    @classmethod
    def bootstrap(cls, store, num_experts, all_gather):
        mode = store.config.transport["control_backend"]
        # Startup only: ensure no rank silently chooses a different backend.
        settings = all_gather((mode, num_experts, store.config.transport["timeout_s"]))
        if any(setting != settings[0] for setting in settings):
            raise ValueError("EP ranks disagree on shared expert control configuration")
        if mode == "gloo":
            return None
        world_size = len(store.directory.manifests)
        error = None
        if set(store.regions) != set(range(world_size)):
            error = "Shared expert control requires all EP ranks on the same visible host"
        failures = [item for item in all_gather(error) if item]
        if failures:
            raise RuntimeError("; ".join(failures))
        control = None
        descriptor = None
        error = None
        if store.rank == 0:
            try:
                library = build_control_library(store.job_dir)
                control = cls(store.job_dir / "control.bin", library, store.rank, world_size,
                              num_experts, store.config.transport["timeout_s"], create=True)
                control._owns_library = True
                stat = control.path.stat()
                descriptor = (str(control.path), str(library), stat.st_dev, stat.st_ino)
            except Exception as exc:
                error = f"shared expert control creation: {type(exc).__name__}: {exc}"
        reports = all_gather((error, descriptor))
        failures = [item for item, _ in reports if item]
        if failures:
            if control is not None:
                control.close()
            raise RuntimeError("; ".join(failures))
        path, library, device, inode = reports[0][1]
        error = None
        try:
            stat = Path(path).stat()
            if (stat.st_dev, stat.st_ino) != (device, inode):
                raise ValueError("Shared expert control inode is not locally visible")
            if control is None:
                control = cls(path, library, store.rank, world_size, num_experts,
                              store.config.transport["timeout_s"])
        except Exception as exc:
            error = f"rank {store.rank} control attachment: {type(exc).__name__}: {exc}"
        failures = [item for item in all_gather(error) if item]
        if failures:
            if control is not None:
                control.close()
            raise RuntimeError("; ".join(failures))
        return control

    def _enter(self):
        if self.closed:
            raise RuntimeError("Shared expert control is closed")
        if not self._lock.acquire(blocking=False):
            self.abort("Concurrent collectives on the same EP rank")
            raise RuntimeError("Concurrent shared expert control calls")
        if self.sequence == MAX_SEQUENCE:
            self._lock.release()
            self.abort("Shared expert control sequence exhausted")
            raise RuntimeError("Shared expert control sequence exhausted")
        self.sequence += 1
        self._error.value = b""

    def _check_tensor(self, tensor, dtype):
        if (tensor.device.type != "cpu" or tensor.dtype != dtype or
                tuple(tensor.shape) != (self.num_experts,) or not tensor.is_contiguous()):
            self.abort("Invalid shared expert control statistic tensor")
            raise ValueError("Control statistics require contiguous CPU tensors of the registered shape/dtype")

    def statistics(self, layer, counts, scores=None):
        self._check_tensor(counts, torch.int64)
        if scores is not None:
            self._check_tensor(scores, torch.float32)
        counts_out = torch.empty_like(counts)
        scores_out = None if scores is None else torch.empty_like(scores)
        self._enter()
        try:
            result = self.library.expert_control_statistics(
                self.address, self.rank, self.sequence, layer, self.num_experts,
                counts.data_ptr(), None if scores is None else scores.data_ptr(),
                counts_out.data_ptr(), None if scores_out is None else scores_out.data_ptr(),
                self.timeout_ns, self._error, ERROR_BYTES)
            if result:
                raise RuntimeError(self._error.value.decode(errors="replace"))
            return counts_out, scores_out
        finally:
            self._lock.release()

    def ready(self, layer, failure=None):
        message = b"" if failure is None else str(failure).encode()[:MIN_PAYLOAD_BYTES - 1]
        self._enter()
        try:
            result = self.library.expert_control_ready(
                self.address, self.rank, self.sequence, layer, message,
                self.timeout_ns, self._error, ERROR_BYTES)
            if result < 0:
                raise RuntimeError(self._error.value.decode(errors="replace"))
            return [] if result == 0 else [self._error.value.decode(errors="replace")]
        finally:
            self._lock.release()

    def abort(self, message):
        if not self.closed:
            self.library.expert_control_abort(self.address, self.rank, str(message).encode())

    def close(self):
        """Caller retires graph callbacks and watchdog before releasing the map."""
        if self.closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close shared expert control during a collective")
        try:
            self.closed = True
            self._buffer = None
            self.mapping.close()
            if self._owns_files:
                self.path.unlink(missing_ok=True)
            if self._owns_library:
                self.library_path.unlink(missing_ok=True)
        finally:
            self._lock.release()


class CallbackWatchdog:
    """One sleeping deadline monitor per manager, rather than two timers/layer."""

    def __init__(self, timeout, on_timeout):
        self.timeout = timeout
        self.on_timeout = on_timeout
        self._condition = threading.Condition()
        self._deadline = None
        self._closed = False
        self._thread = threading.Thread(target=self._monitor, name="expert-control-watchdog", daemon=True)
        self._thread.start()

    def arm(self):
        with self._condition:
            if self._closed or self._deadline is not None:
                raise RuntimeError("Overlapping or retired expert callback")
            self._deadline = time.monotonic() + self.timeout
            self._condition.notify()

    def disarm(self):
        with self._condition:
            self._deadline = None
            self._condition.notify()

    def _monitor(self):
        with self._condition:
            while not self._closed:
                if self._deadline is None:
                    self._condition.wait()
                    continue
                remaining = self._deadline - time.monotonic()
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
                self._deadline = None
                # Timeout handlers must not call back into this monitor.
                self.on_timeout()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join()
