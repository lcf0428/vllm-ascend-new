"""Static expert-pool transfers, recorded on the consuming compute stream.

Adapters prepare all views and optional layout staging before capture. The
scheduler emits only async tensor copies and device conversions: these become
graph tasks and execute on EVERY replay, including pool-slot overwrites.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


def byte_view(value, nbytes: int) -> torch.Tensor:
    """Alias a physical byte range, without interpreting packed/NZ metadata."""
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        offset = value.storage_offset() * value.element_size()
    elif isinstance(value, torch.UntypedStorage):
        storage, offset = value, 0
    else:
        raise TypeError("Static graph copies require owned tensors or storage, not pointer-only sources")
    if nbytes <= 0 or offset + nbytes > storage.nbytes():
        raise ValueError("Expert transfer byte range exceeds its backing storage")
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage, offset, (nbytes,), (1,))


@dataclass(frozen=True)
class PoolCopy:
    source: torch.Tensor
    destination: torch.Tensor
    name: str

    @classmethod
    def prepare(cls, source, destination, nbytes: int, name: str):
        src, dst = byte_view(source, nbytes), byte_view(destination, nbytes)
        if src.device.type != "cpu" or not src.is_pinned():
            raise ValueError(f"{name}: graph H2D requires a pinned CPU source")
        if dst.device.type not in ("npu", "privateuseone"):
            raise ValueError(f"{name}: graph H2D requires an NPU destination")
        return cls(src, dst, name)


class PoolDeviceOperation(Protocol):
    """Enqueue fixed-address device work on the current stream, without Host sync."""
    def submit(self) -> None: ...


@dataclass(frozen=True)
class PoolConversion:
    """Device-side copy with dtype conversion into a preallocated cache."""
    source: torch.Tensor
    destination: torch.Tensor

    def submit(self) -> None:
        self.destination.copy_(self.source, non_blocking=True)


@dataclass(frozen=True)
class PoolTransferPlan:
    copies: tuple[PoolCopy, ...]
    conversions: tuple[PoolDeviceOperation, ...] = ()
    # Includes original owners and any pinned layout staging, not just views.
    owners: tuple = ()


class PoolTransferAdapter(Protocol):
    def prepare(self, layer_idx: int, pool_slot: int) -> PoolTransferPlan: ...


class ExpertPoolAdapter:
    """Adapter for the manager's finalized packed weights and quant fields.

    Weight storage has already been converted to the device's physical layout.
    Quant tensors retain the legacy eager copy's LOGICAL semantics, so strided
    layouts are packed into matching pinned storage before graph capture.
    No model-name or dtype heuristic selects fields or post-copy operations.
    """

    QUANT_FIELDS = (
        ("scale_cpu_buffers", "w13_weight_scale", "_prefill_w13_scale"),
        ("scale_cpu_buffers", "w2_weight_scale", "_prefill_w2_scale"),
        ("offset_cpu_buffers", "w13_weight_offset", "_prefill_w13_offset"),
        ("offset_cpu_buffers", "w2_weight_offset", "_prefill_w2_offset"),
        ("scale_bias_cpu_buffers", "w13_scale_bias", "_prefill_w13_scale_bias"),
        ("scale_bias_cpu_buffers", "w2_scale_bias", "_prefill_w2_scale_bias"),
    )

    def __init__(self, manager):
        self.manager = manager

    @staticmethod
    def _quant_source(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        if source.numel() != destination.numel() or source.dtype != destination.dtype:
            raise ValueError("Quant field needs a model/quantization adapter for dtype or size conversion")
        expected_stride = 1
        for size, stride in sorted(zip(destination.shape, destination.stride()), key=lambda entry: entry[1]):
            if size <= 1:
                continue
            if stride != expected_stride:
                raise ValueError("Quant pool adapter requires dense, non-overlapping physical storage")
            expected_stride *= size
        logical = source.reshape(destination.shape)
        same_layout = logical.stride() == destination.stride() or (
            logical.is_contiguous() and destination.is_contiguous()
        )
        if same_layout and logical.is_pinned():
            return logical
        # Reproduce destination.copy_(source.reshape(destination.shape)) byte for
        # byte, outside capture. Immutable checkpoint fields are staged once.
        staged = torch.empty_strided(
            destination.shape, destination.stride(), dtype=source.dtype, device="cpu", pin_memory=True
        )
        staged.copy_(logical)
        return staged

    def prepare(self, layer_idx: int, pool_slot: int) -> PoolTransferPlan:
        mgr = self.manager
        if mgr.native_mxfp4_enabled or getattr(mgr, "_exclusive_pool", None) is not None:
            raise ValueError("Static graph pool transfers require immutable finalized Host expert storage")
        # Quant methods with additional fields/layouts can supply a complete
        # static plan, including custom capture-safe device post-operations.
        layers = getattr(mgr, "moe_layers", ())
        quant_method = getattr(layers[layer_idx], "quant_method", None) if layers else None
        factory = getattr(quant_method, "build_pool_transfer_plan", None)
        if factory is not None:
            plan = factory(mgr, layer_idx, pool_slot)
            if not isinstance(plan, PoolTransferPlan):
                raise TypeError("Quant adapter must return a PoolTransferPlan")
            return plan
        count = mgr.mc_shard_size
        base = mgr.ep_rank * count
        copies, owners, conversions = [], [], []
        for local_eid in range(count):
            global_eid = base + local_eid
            for name in ("w13", "w2"):
                owner = getattr(mgr, f"_prefill_{name}")[pool_slot]
                nbytes = getattr(mgr, f"{name}_expert_size_bytes")
                if owner.untyped_storage().nbytes() != count * nbytes:
                    raise ValueError(f"{name}: pool physical expert stride differs from converted Host weights")
                destination = owner.untyped_storage()[local_eid * nbytes : (local_eid + 1) * nbytes]
                source = mgr._expert_src_storage(layer_idx, global_eid, name)
                copies.append(PoolCopy.prepare(source, destination, nbytes, f"L{layer_idx}:{name}:{global_eid}"))
                owners.extend((source, owner))
            for buffers_name, field, pool_name in self.QUANT_FIELDS:
                buffers, pools = getattr(mgr, buffers_name), getattr(mgr, pool_name)
                if pool_slot >= len(pools):
                    if field in buffers:
                        raise ValueError(f"{field}: Host field has no matching device pool")
                    continue
                if field not in buffers or layer_idx >= len(buffers[field]):
                    raise ValueError(f"{field}: missing Host quantization field for layer {layer_idx}")
                source_eid = mgr._shard_local(global_eid)
                if source_eid is None:
                    raise ValueError("Static pool adapter must only read the local EP shard")
                original = buffers[field][layer_idx][source_eid]
                destination = pools[pool_slot][local_eid]
                source = self._quant_source(original, destination)
                nbytes = destination.numel() * destination.element_size()
                copies.append(PoolCopy.prepare(source, destination, nbytes, f"L{layer_idx}:{field}:{global_eid}"))
                owners.extend((original, source, pools[pool_slot]))
        # W8A8's FP32 cache is a device conversion, not an extra Host copy.
        # MXFP4 has no such field. Use actual pool capabilities, not dtype guesses.
        if pool_slot < len(mgr._prefill_w13_scale_fp32):
            source = mgr._prefill_w13_scale[pool_slot]
            destination = mgr._prefill_w13_scale_fp32[pool_slot]
            if source.shape != destination.shape:
                raise ValueError("Derived FP32 scale cache has incompatible shape")
            conversions.append(PoolConversion(source, destination))
            owners.extend((source, destination))
        return PoolTransferPlan(tuple(copies), tuple(conversions), tuple(owners))


class PoolTransferScheduler:
    """Serial same-stream scheduler with graph-lifetime plan ownership.

    Prepare outside capture after weights/layouts are finalized. Plans must not
    be replaced while any graph references them; model reload requires graph
    destruction and a new scheduler. A worker must serialize graph/eager forwards
    that share a pool. No residency shortcuts are valid across replay.
    """

    def __init__(self, adapter: PoolTransferAdapter):
        self.adapter = adapter
        self.plans: dict[tuple[int, int], PoolTransferPlan] = {}
        self.streams: dict[int, object] = {}

    def submit(self, layer_idx: int, pool_slot: int) -> None:
        key = (layer_idx, pool_slot)
        plan = self.plans.get(key)
        if plan is None:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError(f"Expert pool plan {key} was not prepared by eager warmup before capture")
            plan = self.adapter.prepare(layer_idx, pool_slot)
            self.plans[key] = plan
        stream = torch.npu.current_stream()
        self.streams[stream.npu_stream] = stream
        for copy in plan.copies:
            copy.destination.copy_(copy.source, non_blocking=True)
        for operation in plan.conversions:
            operation.submit()

    def close(self) -> None:
        """Only after all referencing graphs have been retired."""
        for stream in self.streams.values():
            stream.synchronize()
        self.plans.clear()
        self.streams.clear()
