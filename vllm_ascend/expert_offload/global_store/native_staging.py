# SPDX-License-Identifier: Apache-2.0
"""Opt-in same-host native staging with bounded, event-protected chunks."""

import hashlib
import importlib.util
import subprocess
import sysconfig
import threading
from pathlib import Path

import torch

from .staging import _aligned_size
from .types import Source

BUILD_TIMEOUT_S = 300
STAT_NAMES = ("pack_ns", "ready_wait_ns", "submit_ns", "first_submit_ns", "total_ns", "chunks")


def staging_chunk_layout(experts, budget, submit):
    """Total expert budget is unchanged; final chunk may hold fewer experts."""
    if type(budget) is not int or budget < 2:
        raise ValueError("staging_buffer_experts must be an integer >= 2")
    if type(submit) is not int or not 0 < submit < budget:
        raise ValueError("staging_submit_experts must be positive and smaller than staging_buffer_experts")
    unit = max((sum(_aligned_size(t.untyped_storage().nbytes()) for t in fields.values())
                for _, fields in experts), default=0)
    if unit <= 0:
        raise ValueError("Cannot size native staging without expert allocations")
    counts = [min(submit, budget - i) for i in range(0, budget, submit)]
    return tuple(unit * n for n in counts), counts


def build_native_staging(directory, *, cpu_test=False):
    """Build against this interpreter's Torch ABI, without pip or global caches."""
    from torch.utils.cpp_extension import include_paths, library_paths

    source = Path(__file__).with_suffix(".cpp")
    identity = (source.read_bytes() + str((torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI,
                                         sysconfig.get_config_var("SOABI"), cpu_test)).encode())
    digest = hashlib.sha256(identity).hexdigest()[:16]
    name = f"expert_staging_{digest}"
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = directory / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    if not output.exists():
        includes = include_paths() + [sysconfig.get_path("include")]
        libraries = library_paths()
        flags = ["-DEXPERT_STAGING_CPU_TEST"] if cpu_test else []
        links = ["-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10"]
        if not cpu_test:
            import torch_npu

            npu = Path(torch_npu.__file__).parent
            # Match the CANN installation used by this process, not system settings.
            cann = Path("/usr/local/Ascend/ascend-toolkit/latest")
            includes += [str(npu / "include"), str(cann / "include")]
            libraries += [str(npu / "lib"), str(cann / "lib64")]
            links += ["-ltorch_npu", "-lascendcl"]
        command = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread",
                   "-Wall", "-Wextra", "-Werror=return-type", f"-DTORCH_EXTENSION_NAME={name}",
                   f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", *flags,
                   *[f"-I{p}" for p in includes], str(source),
                   *[f"-L{p}" for p in libraries], *[f"-Wl,-rpath,{p}" for p in libraries],
                   *links, "-o", str(output)]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=BUILD_TIMEOUT_S)
        except subprocess.CalledProcessError as exc:
            output.unlink(missing_ok=True)
            raise RuntimeError(f"Native staging compilation failed: {exc.stderr.decode(errors='replace')}") from exc
        output.chmod(0o600)
    return output


def load_native_staging(path):
    path = Path(path)
    name = path.name.split(".")[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeStagingTransport:
    supports_remote_sources = True

    def __init__(self, store, staging, delegate, capacities, counts, module, *, cpu_test=False, delay_us=0):
        self.store, self.staging, self.delegate = store, staging, delegate
        config = store.config.transport
        self._lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._destinations = {}
        self._library_cleanup = None
        self._sources = {}
        self._stream = None
        self._cpu_test = cpu_test
        self.engine = module.Pipeline(staging, list(capacities), counts,
                                      config["staging_pack_threads"], config["timeout_s"], delay_us)
        prototypes = {}
        try:
            for bundle in store.directory.bundles.values():
                region = store.regions.get(bundle.owner)
                if region is None:
                    raise ValueError("native staging requires every canonical owner locally mapped")
                for field in bundle.fields:
                    if field.offset < 0 or field.nbytes <= 0 or field.offset + field.nbytes > region.size:
                        raise ValueError("Native source exceeds canonical mapping")
                    dtype = getattr(torch, field.dtype.removeprefix("torch."))
                    if dtype not in prototypes:
                        prototypes[dtype] = torch.empty(0, dtype=dtype)
                    sid = self.engine.add_source(region.address + field.offset, field.nbytes,
                                                 prototypes[dtype], field.shape, field.stride)
                    self._sources[(bundle.key, field.name)] = (Source(bundle, field), sid)
        except BaseException:
            self.engine.close()
            raise

    def allocate_host_tensor(self, shape, dtype):
        return self.delegate.allocate_host_tensor(shape, dtype)

    def _destination(self, task):
        # Cache by address/layout, retaining owners. Prefill creates fresh Python
        # views of the same slots; those must not grow the descriptor registry.
        destination = task.destination
        key = (("tensor", destination.data_ptr(), destination.dtype, tuple(destination.shape),
                tuple(destination.stride()), destination.device) if isinstance(destination, torch.Tensor)
               else ("raw", destination.data_ptr(), destination.nbytes(), destination.device))
        prior = self._destinations.get(key)
        if prior is not None:
            if prior[1] != task.nbytes:
                raise ValueError("Native destination descriptor changed")
            return prior[2]
        if isinstance(destination, torch.Tensor):
            did = self.engine.add_destination(destination, 0, task.nbytes)
        else:
            if destination.nbytes() != task.nbytes:
                raise ValueError("Native raw destination size mismatch")
            did = self.engine.add_destination(torch.empty(0), destination.data_ptr(), task.nbytes)
        self._destinations[key] = (destination, task.nbytes, did)
        return did

    def copy_batch(self, tasks):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Overlapping native staging calls")
        try:
            if self._closed or self._poisoned:
                raise RuntimeError("Native staging is closed or poisoned")
            source_ids, destination_ids, ends = [], [], []
            previous = None
            fields = set()
            for task in tasks:
                if not isinstance(task.source, Source):
                    raise ValueError("Native staging requires canonical Source descriptors")
                source = task.source
                expected, sid = self._sources[(source.bundle.key, source.field.name)]
                if source != expected or source.nbytes != task.nbytes:
                    raise ValueError("Native canonical source size/metadata mismatch")
                if previous is not None and (source.bundle.key != previous or source.field.name in fields):
                    ends.append(len(source_ids))
                    fields.clear()
                previous = source.bundle.key
                fields.add(source.field.name)
                source_ids.append(sid)
                destination_ids.append(self._destination(task))
            if source_ids:
                ends.append(len(source_ids))
                stream = None if self._cpu_test else torch.npu.current_stream()
                if self._stream is not None and stream != self._stream:
                    # Manager normally uses one load stream. Cross-stream reuse
                    # drains the old stream before touching its chunk events.
                    self.engine.synchronize()
                self._stream = stream
                self.engine.run(source_ids, destination_ids, ends, 0 if stream is None else stream.device_index)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._lock.release()

    @property
    def last_stats(self):
        return dict(zip(STAT_NAMES, self.engine.stats()))

    def synchronize(self, stream):
        with self._lock:
            if self._closed:
                return
            # Includes derived scale operations enqueued after native run().
            self.delegate.synchronize(stream)
            self.engine.synchronize()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.engine.close()
            self.delegate.close()
            self._destinations.clear()
            self._closed = True
            if self._library_cleanup is not None:
                self._library_cleanup.unlink(missing_ok=True)
                try:
                    self._library_cleanup.parent.rmdir()
                except OSError:
                    pass


def create_native_staging(store, staging, delegate, capacities, counts):
    """Coordinate startup failures; no automatic hot-path backend fallback."""
    error, library = None, None
    if store.rank == 0:
        try:
            library = str(build_native_staging(store.job_dir / "native-staging"))
        except Exception as exc:
            error = str(exc)
    reports = store._all_gather((error, library))
    if reports[0][0]:
        raise RuntimeError(reports[0][0])
    transport = None
    try:
        module = load_native_staging(reports[0][1])
        transport = NativeStagingTransport(store, staging, delegate, capacities, counts, module)
    except Exception as exc:
        error = f"rank {store.rank}: {type(exc).__name__}: {exc}"
    failures = [e for e in store._all_gather(error) if e]
    if failures:
        if transport is not None:
            transport.close()
        raise RuntimeError("Native staging initialization failed: " + "; ".join(failures))
    if store.rank == 0:
        transport._library_cleanup = Path(reports[0][1])
    return transport
