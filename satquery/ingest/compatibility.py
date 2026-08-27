from dataclasses import dataclass, field


@dataclass
class CompatibilityReport:
    format_ok: bool
    crs_valid: bool
    modality: str
    modality_source: str | None = None
    bands_present: list[str] = field(default_factory=list)
    computable_indices: list[str] = field(default_factory=list)
    nodata_frac: float = 0.0
    bit_depth: int = 8
    bit_depth_source: str | None = None
    pixel_size_m: float | None = None
    native_gsd_m: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_readable(self) -> bool:
        return self.format_ok

    @property
    def is_georeferenced(self) -> bool:
        return self.crs_valid
