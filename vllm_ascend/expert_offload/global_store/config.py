"""Strict, opt-in configuration; legacy offload keeps its existing defaults."""

from dataclasses import dataclass, field
from pathlib import Path


def _section(value, defaults, name):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    unknown = value.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return dict(defaults, **value)


@dataclass(frozen=True)
class GlobalStoreConfig:
    host: dict = field(default_factory=dict)
    transport: dict = field(default_factory=dict)
    placement: dict = field(default_factory=dict)
    graph: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, values):
        host = _section(values.get("host_storage"), {
            "mode": "legacy", "replicas": 1, "runtime_dir": None,
            "startup_timeout_s": 600,
            "backing": "memfd",
        }, "host_storage")
        if "staging_buffer_mb" in (values.get("transport") or {}):
            raise ValueError("staging_buffer_mb was replaced by staging_buffer_experts "
                             "(total expert capacity across two banks per rank; minimum 2)")
        transport = _section(values.get("transport"), {
            "intra_node": "shared_memory", "inter_node": "disabled",
            "control_backend": "gloo",
            "staging_buffer_experts": 4, "timeout_s": 60,
            "bind_address": "0.0.0.0", "advertise_address": None,
            "rdma_library": None,
            "rdma_device": "", "rdma_port": 1, "rdma_gid_index": 0,
        }, "transport")
        placement = _section(values.get("placement"), {
            "scope": "ep_group", "policy": "stable",
        }, "placement")
        graph = _section(values.get("graph"), {"mc2_decode": False}, "graph")
        if host["mode"] not in ("legacy", "global_sharded"):
            raise ValueError("host_storage.mode must be legacy or global_sharded")
        if host["backing"] not in ("memfd", "file"):
            raise ValueError("host_storage.backing must be memfd or file")
        if type(host["replicas"]) is not int or host["replicas"] != 1:
            raise ValueError("Version 1 supports exactly one Host replica")
        if transport["intra_node"] != "shared_memory":
            raise ValueError("Version 1 requires intra_node=shared_memory")
        if transport["inter_node"] not in ("disabled", "tcp", "rdma"):
            raise ValueError("transport.inter_node must be disabled, tcp or rdma")
        if transport["control_backend"] not in ("gloo", "shared_memory"):
            raise ValueError("transport.control_backend must be gloo or shared_memory")
        if transport["control_backend"] == "shared_memory" and host["mode"] != "global_sharded":
            raise ValueError("shared_memory control requires host_storage.mode=global_sharded")
        if placement != {"scope": "ep_group", "policy": "stable"}:
            raise ValueError("Version 1 requires stable placement within the EP group")
        if type(graph["mc2_decode"]) is not bool:
            raise TypeError("graph.mc2_decode must be boolean")
        for name, value in (("staging_buffer_experts", transport["staging_buffer_experts"]),
                            ("timeout_s", transport["timeout_s"]),
                            ("startup_timeout_s", host["startup_timeout_s"])):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(transport["rdma_port"]) is not int or transport["rdma_port"] <= 0:
            raise ValueError("rdma_port must be positive")
        if type(transport["rdma_gid_index"]) is not int or transport["rdma_gid_index"] < 0:
            raise ValueError("rdma_gid_index must be nonnegative")
        if transport["staging_buffer_experts"] < 2:
            raise ValueError("staging_buffer_experts must be >= 2 (two banks total)")
        if host["mode"] == "global_sharded":
            if not host["runtime_dir"] or not Path(host["runtime_dir"]).is_absolute():
                raise ValueError("global_sharded requires an absolute host_storage.runtime_dir")
            if "shard_per_rank" in values and values["shard_per_rank"] is not True:
                raise ValueError("global_sharded loads one owner shard; shard_per_rank=False conflicts")
            if not values.get("expert_offload") or not values.get("enable_multi_card"):
                raise ValueError("global_sharded requires expert_offload and enable_multi_card")
            if values.get("h2d_backend", "torch") != "torch":
                raise ValueError("global_sharded requires h2d_backend=torch")
            for name in ("exclusive_expert_storage", "exclusive_load_pool_enabled",
                         "expert_prefetch_enabled", "expert_substitution_enabled",
                         "experts_pruning_enabled", "prefill_delta_load", "prefill_split_gmm",
                         "hot_expert_preload"):
                if values.get(name):
                    raise ValueError(f"global_sharded version 1 does not support {name}")
            if values.get("cpu_moe", {}).get("enabled", False):
                raise ValueError("global_sharded version 1 does not support cpu_moe")
        elif graph["mc2_decode"]:
            raise ValueError("graph.mc2_decode requires host_storage.mode=global_sharded")
        return cls(host, transport, placement, graph)

    def validate_parallel(self, parallel, *, enforce_eager):
        """Keep physical TP separate from the full-expert EP weight layout."""
        if not self.enabled:
            return
        for name in ("pipeline_parallel_size", "prefill_context_parallel_size",
                     "decode_context_parallel_size"):
            if getattr(parallel, name, 1) != 1:
                raise ValueError("global_sharded requires PP=PCP=DCP=1")
        if not parallel.enable_expert_parallel:
            raise ValueError("global_sharded requires enable_expert_parallel")
        if getattr(parallel, "enable_eplb", False) or getattr(parallel, "enable_elastic_ep", False):
            raise ValueError("global_sharded requires static EP without EPLB")
        if parallel.tensor_parallel_size > 1:
            if parallel.data_parallel_size != 1:
                raise ValueError("global_sharded TP>1 currently requires DP=1")

    @property
    def enabled(self):
        return self.host["mode"] == "global_sharded"
