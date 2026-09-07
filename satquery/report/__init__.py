"""P8 — evidence, masks and reporting (master plan section 4.8)."""

from satquery.report.exporter import ReportExporter
from satquery.report.masks import (
    MaskConformanceError,
    MaskRecord,
    check_mask_conformance,
    write_mask,
)
from satquery.report.overlays import MASK_COLOURS, write_overlay, write_rgb_preview
from satquery.report.report_builder import ReportBuilder

__all__ = [
    "MASK_COLOURS",
    "MaskConformanceError",
    "MaskRecord",
    "ReportBuilder",
    "ReportExporter",
    "check_mask_conformance",
    "write_mask",
    "write_overlay",
    "write_rgb_preview",
]
