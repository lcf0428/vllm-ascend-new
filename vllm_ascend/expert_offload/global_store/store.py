"""EP bootstrap and immutable canonical storage lifecycle."""

import gc
import secrets
from pathlib import Path

import torch

from .directory import Directory
from .shared_store import SharedRegion
from .topology import advertise_address, can_map, create_probe
from .transport_tcp import TCPReader, TCPServer
from .types import ExpertKey


class GlobalExpertStore:
    def __init__(self, config, rank, world_size, all_gather, model_id, layers, experts,
                 staging, *, role="target"):
        self.config = config
        self.rank = rank
        self.role = role
        self.model_id = model_id
        self.layers = layers
        self.regions = {}
        self.server = None
        self.reader = None
        self.ready = False
        self.job_dir = None
        self.closed = False
        self._all_gather = all_gather
        ids = all_gather(secrets.token_hex(16) if rank == 0 else None)
        self.job_dir = Path(config.host["runtime_dir"]) / ids[0]
        self.job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        probe = create_probe(self.job_dir, rank)
        local = None
        error = None
        try:
            region, bundles = SharedRegion.publish(self.job_dir / f"owner-{rank}.bin", rank, experts,
                                                   backing=config.host["backing"])
            self.regions[rank] = region
            protocol = config.transport["inter_node"]
            if protocol == "tcp":
                self.server = TCPServer(region, config.transport["bind_address"],
                                        config.transport["timeout_s"], staging.numel())
                self.reader = TCPReader(config.transport["timeout_s"])
            elif protocol == "rdma":
                from .transport_rdma import RDMAReader, RDMAServer

                self.server = RDMAServer(region, config.transport)
                self.reader = RDMAReader(config.transport, staging)
            endpoint = None if self.server is None else {
                "address": advertise_address(config.transport), "port": self.server.port,
                "token": self.server.token,
            }
            local = dict(probe, rank=rank, size=region.size, path=str(region.path),
                         identity=region.identity, bundles=bundles, endpoint=endpoint)
        except Exception as exc:
            error = f"rank {rank} publication: {type(exc).__name__}: {exc}"
        reports = all_gather((error, local))
        failures = [e for e, _ in reports if e]
        if failures:
            raise RuntimeError("Global expert publication failed: " + "; ".join(failures))
        manifests = [m for _, m in reports]
        expected = [self.key(i, eid) for i in range(len(layers)) for eid in range(world_size * len(experts) // len(layers))]
        # Each EP owner publishes an equal number of complete experts.
        self.directory = Directory(manifests, expected)
        error = None
        try:
            for manifest in manifests:
                owner = manifest["rank"]
                if owner == rank:
                    continue
                mapped = None
                if can_map(manifest):
                    try:
                        mapped = SharedRegion(manifest["path"], manifest["size"])
                        if mapped.identity != manifest["identity"]:
                            mapped.close()
                            mapped = None
                    except (OSError, ValueError):
                        # A shared filesystem does not imply that /proc FDs
                        # are visible across separate process namespaces.
                        mapped = None
                if mapped is not None:
                    self.regions[owner] = mapped
                elif self.reader is None:
                    raise RuntimeError("Peer Host store is not locally visible; select inter_node=tcp or rdma")
                elif protocol == "rdma":
                    self.reader.connect(owner, manifest["endpoint"])
        except Exception as exc:
            error = f"rank {rank} connectivity: {type(exc).__name__}: {exc}"
        failures = [e for e in all_gather(error) if e]
        if failures:
            raise RuntimeError("Global expert connectivity failed: " + "; ".join(failures))
        self.ready = True

    def key(self, layer, eid):
        return ExpertKey(self.model_id, self.role, self.layers[layer], eid)

    def source(self, layer, eid, name):
        if not self.ready:
            raise RuntimeError("Expert directory is not READY")
        return self.directory.source(self.key(layer, eid), name)

    def read_into(self, source, offset, target, size):
        if offset < 0 or size <= 0 or offset + size > source.nbytes:
            raise ValueError("Expert field read out of bounds")
        owner = source.bundle.owner
        position = source.field.offset + offset
        if owner in self.regions:
            self.regions[owner].read_into(position, target, size)
        else:
            endpoint = self.directory.manifests[owner]["endpoint"]
            if self.config.transport["inter_node"] == "rdma":
                self.reader.read_into(owner, endpoint, position, target, size)
            else:
                self.reader.read_into(endpoint, position, target, size)

    def local_tensor(self, source):
        """Metadata-compatible view replacing the old private owner tensor."""
        field = source.field
        dtype = getattr(torch, field.dtype.removeprefix("torch."))
        region = self.regions[self.rank]
        raw = torch.frombuffer(region.mapping, dtype=torch.uint8)
        tensor = torch.empty(0, dtype=dtype)
        return tensor.set_(raw.untyped_storage(), field.offset // tensor.element_size(),
                           field.shape, field.stride)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.ready = False
        if self.reader is not None and hasattr(self.reader, "close"):
            self.reader.close()
        # All ranks have drained their own reads before any owner removes its MR.
        self._all_gather(None)
        if self.server is not None:
            self.server.close()
        gc.collect()
        for region in self.regions.values():
            region.close()
        for name in (f"owner-{self.rank}.bin", f"visibility-{self.rank}"):
            (self.job_dir / name).unlink(missing_ok=True)
        self.regions.clear()
        try:
            self.job_dir.rmdir()
        except OSError:
            pass
