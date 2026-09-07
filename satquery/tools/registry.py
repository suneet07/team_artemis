from dataclasses import dataclass, field
from typing import Any

from satquery.ingest.band_inventory import BandInventory
from satquery.paths import TOOLS_DIR
from satquery.tools.manifest import ToolManifest


@dataclass
class ParameterCheckResult:
    passed: bool
    rejected: list[str] = field(default_factory=list)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalise_modalities(modalities: "str | list[str] | None") -> set[str] | None:
    if modalities is None:
        return None
    if isinstance(modalities, str):
        return {modalities}
    return {m for m in modalities}


def check_parameters(
    manifest: ToolManifest,
    params: dict[str, Any],
    band_inventory: BandInventory | None = None,
    modalities: str | list[str] | None = None,
) -> ParameterCheckResult:
    rejected: list[str] = []

    provided_modalities = _normalise_modalities(modalities)
    if provided_modalities is not None:
        if manifest.modality_mode == "any":
            # The tool operates on whichever of its modalities it is given, so
            # one is enough. Requiring all of them would reject a SAR-only scene
            # from a tool whose entire job is handling SAR-only scenes.
            if not provided_modalities & set(manifest.required_modalities):
                rejected.append(
                    f"tool '{manifest.name}' accepts modalities "
                    f"{list(manifest.required_modalities)} but the query context "
                    f"provides {sorted(provided_modalities)}"
                )
        else:
            missing = [m for m in manifest.required_modalities if m not in provided_modalities]
            for required in missing:
                rejected.append(
                    f"tool '{manifest.name}' requires modality '{required}' "
                    f"but the query context provides {sorted(provided_modalities)}"
                )

    for name, value in params.items():
        spec = manifest.permitted_parameters.get(name)
        if spec is None:
            rejected.append(f"unknown parameter '{name}' for tool '{manifest.name}'")
            continue
        if spec.type == "enum":
            if not isinstance(value, str) or value not in spec.values:
                rejected.append(
                    f"parameter '{name}'={value!r} not in permitted values {list(spec.values)}"
                )
        elif spec.type in ("float", "int"):
            if not _is_number(value):
                rejected.append(
                    f"parameter '{name}'={value!r} must be "
                    f"{'an integer' if spec.type == 'int' else 'a number'}"
                )
            elif spec.range is not None and not spec.range[0] <= value <= spec.range[1]:
                rejected.append(
                    f"parameter '{name}'={value!r} outside permitted range "
                    f"[{spec.range[0]}, {spec.range[1]}]"
                )
            if spec.type == "int" and _is_number(value) and float(value) != int(value):
                rejected.append(f"parameter '{name}'={value!r} must be an integer")
        elif spec.type == "bool":
            if not isinstance(value, bool):
                rejected.append(f"parameter '{name}'={value!r} must be a boolean")
        elif spec.type == "string":
            if not isinstance(value, str):
                rejected.append(f"parameter '{name}'={value!r} must be a string")
        if spec.requires_bands is not None and band_inventory is not None:
            needed = spec.requires_bands.get(value) if isinstance(value, str) else None
            for band in needed or ():
                if not band_inventory.has_band(band):
                    rejected.append(
                        f"parameter '{name}'={value!r} requires band '{band}' "
                        f"which the source does not provide"
                    )
    for name, spec in manifest.permitted_parameters.items():
        if name not in params and not spec.optional and not spec.has_default:
            rejected.append(f"missing required parameter '{name}' for tool '{manifest.name}'")
    return ParameterCheckResult(passed=not rejected, rejected=rejected)


def effective_params(manifest: ToolManifest, params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(params)
    for name, spec in manifest.permitted_parameters.items():
        if name not in merged and spec.has_default:
            merged[name] = spec.default
    return merged


class ToolRegistry:
    def __init__(self) -> None:
        self._manifests: dict[str, ToolManifest] = {}

    @classmethod
    def default(cls) -> "ToolRegistry":
        registry = cls()
        for path in sorted(TOOLS_DIR.glob("*.yaml")):
            registry.register(ToolManifest.from_yaml(path))
        return registry

    def register(self, manifest: ToolManifest) -> None:
        if manifest.name in self._manifests:
            raise ValueError(f"duplicate tool name '{manifest.name}'")
        self._manifests[manifest.name] = manifest

    def get(self, name: str) -> ToolManifest:
        if name not in self._manifests:
            raise KeyError(f"tool '{name}' is not registered")
        return self._manifests[name]

    def names(self) -> list[str]:
        return sorted(self._manifests)
