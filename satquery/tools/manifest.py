import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from satquery.paths import TOOL_MANIFEST_SCHEMA_PATH

_SCHEMA: dict[str, Any] | None = None


def _schema() -> dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = json.loads(TOOL_MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _SCHEMA


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    values: tuple[Any, ...] | None = None
    range: tuple[float, float] | None = None
    optional: bool = False
    default: Any = None
    has_default: bool = False
    requires_bands: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class ToolManifest:
    name: str
    description: str
    version: int
    required_modalities: tuple[str, ...]
    permitted_parameters: dict[str, ParamSpec]
    outputs: dict[str, dict[str, Any]]
    modality_mode: str = "all"
    confidence_source: str | None = None
    low_confidence_proposer: bool = False
    expected_latency_ms: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ToolManifest":
        jsonschema.Draft202012Validator(_schema()).validate(raw)
        params: dict[str, ParamSpec] = {}
        for name, spec in raw.get("permitted_parameters", {}).items():
            requires = spec.get("requires_bands")
            params[name] = ParamSpec(
                name=name,
                type=spec["type"],
                values=tuple(spec["values"]) if "values" in spec else None,
                range=tuple(spec["range"]) if "range" in spec else None,
                optional=spec.get("optional", False),
                default=spec.get("default"),
                has_default="default" in spec,
                requires_bands={k: tuple(v) for k, v in requires.items()} if requires else None,
            )
        return cls(
            name=raw["name"],
            description=raw["description"],
            version=raw.get("version", 1),
            required_modalities=tuple(raw["required_modalities"]),
            permitted_parameters=params,
            outputs=dict(raw["outputs"]),
            modality_mode=raw.get("modality_mode", "all"),
            confidence_source=raw.get("confidence_source"),
            low_confidence_proposer=raw.get("low_confidence_proposer", False),
            expected_latency_ms=raw.get("expected_latency_ms"),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "ToolManifest":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)
