"""Validate all owner manifests before permitting arbitrary placement."""

from .types import Source


class Directory:
    def __init__(self, manifests, expected_keys):
        self.bundles = {}
        self.manifests = {m["rank"]: m for m in manifests}
        if len(self.manifests) != len(manifests):
            raise ValueError("Duplicate owner rank in expert directory")
        for manifest in manifests:
            intervals = []
            for bundle in manifest["bundles"]:
                if bundle.key in self.bundles or bundle.owner != manifest["rank"]:
                    raise ValueError(f"Duplicate or invalid expert owner: {bundle.key}")
                names = set()
                for f in bundle.fields:
                    if f.name in names or f.nbytes <= 0 or f.offset < 0:
                        raise ValueError(f"Invalid field descriptor: {bundle.key}/{f.name}")
                    if f.offset + f.nbytes > manifest["size"]:
                        raise ValueError("Expert field exceeds owner region")
                    names.add(f.name)
                    intervals.append((f.offset, f.offset + f.nbytes))
                if not {"w13", "w2"} <= names:
                    raise ValueError("Expert bundle must include both weight matrices")
                self.bundles[bundle.key] = bundle
            intervals.sort()
            if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
                raise ValueError("Overlapping expert fields")
        if self.bundles.keys() != set(expected_keys):
            raise ValueError("Incomplete or unexpected global expert directory")
        # Immutable canonical descriptors: reuse them across graph callbacks
        # instead of invalidating the manager's cached H2D tasks every layer.
        self._sources = {(key, field.name): Source(bundle, field)
                         for key, bundle in self.bundles.items() for field in bundle.fields}
        # Every peer must consume exactly the same raw layout for a layer.
        layouts = {}
        for key, bundle in self.bundles.items():
            layer = (key.model_id, key.role, key.layer, key.layout_version)
            layout = tuple((f.name, f.nbytes, f.shape, f.stride, f.dtype) for f in bundle.fields)
            if layouts.setdefault(layer, layout) != layout:
                raise ValueError(f"Incompatible expert layouts for {layer}")

    def source(self, key, name):
        return self._sources[(key, name)]
