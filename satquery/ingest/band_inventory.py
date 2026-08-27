from dataclasses import dataclass, field


@dataclass
class BandInventory:
    bands: dict[str, int] = field(default_factory=dict)
    has_swir: bool = False
    has_nir: bool = False
    is_pan_only: bool = False
    polarisations: list[str] = field(default_factory=list)
    sar_band: str | None = None
    sensor_hint: str | None = None
    computable_indices: list[str] = field(default_factory=list)

    def has_band(self, name: str) -> bool:
        lowered = name.lower()
        present = {b.lower() for b in self.bands}
        if lowered == "swir":
            return self.has_swir or lowered in present
        if lowered == "nir":
            return self.has_nir or lowered in present
        return lowered in present
