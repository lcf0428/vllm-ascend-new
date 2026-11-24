"""Serializable directory entries describe raw, post-conversion bytes."""

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ExpertKey:
    model_id: str
    role: str
    layer: str
    expert: int
    layout_version: int = 1


@dataclass(frozen=True)
class Field:
    name: str
    offset: int
    nbytes: int
    shape: tuple
    stride: tuple
    dtype: str


@dataclass(frozen=True)
class Bundle:
    key: ExpertKey
    owner: int
    fields: tuple[Field, ...]


@dataclass(frozen=True)
class Source:
    bundle: Bundle
    field: Field

    @property
    def nbytes(self):
        return self.field.nbytes
