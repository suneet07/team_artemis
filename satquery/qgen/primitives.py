import dataclasses
from collections.abc import Iterator
from typing import Any, Protocol


@dataclasses.dataclass
class Primitive:
    """Base class for all annotation primitives."""
    sample_id: str
    image_paths: list[str]
    effective_gsd_m: list[float]
    source: str
    source_ann_id: str
    licence: str
    provenance_chain: list[str]
    split: str


@dataclasses.dataclass
class P1MultiLabel(Primitive):
    """Multi-label class vector (+ optional pixel map). e.g., BEN.txt"""
    labels: list[str]


@dataclasses.dataclass
class P2BoundingBox(Primitive):
    """Bounding boxes with class + attributes. e.g., RarePlanes, LS-SSDD"""
    # {"class": str, "bbox": [ymin, xmin, ymax, xmax], "attributes": dict}
    bboxes: list[dict[str, Any]]


@dataclasses.dataclass
class P3Polygon(Primitive):
    """Polygon footprints, optionally with persistent IDs. e.g., SpaceNet 6, SpaceNet 7"""
    polygons: list[dict[str, Any]] # format: {"id": str, "class": str, "geometry": str (WKT)}


@dataclasses.dataclass
class P4SegMask(Primitive):
    """Semantic segmentation mask. e.g., OEM-SAR, HRSCD, BEN CLC map"""
    mask_path: str
    classes_present: list[str]


@dataclasses.dataclass
class P5Change(Primitive):
    """Bi-temporal pair + change mask / ID delta. e.g., SpaceNet 7, HRSCD"""
    change_mask_path: str | None
    id_delta: dict[str, Any] | None # added, removed, persisted counts


@dataclasses.dataclass
class P6CrossModal(Primitive):
    """Co-registered modality pair. e.g., BEN S1+S2, SpaceNet 6, OEM-SAR"""
    labels: Any  # Can be P1-P4


class Loader(Protocol):
    """Protocol for dataset loaders to emit primitives."""
    source: str
    licence: str
    provenance_chain: list[str]

    def emit(self) -> Iterator[Primitive]:
        ...
