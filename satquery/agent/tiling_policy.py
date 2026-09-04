from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from satquery.agent.bundle import ImageBundle
from satquery.paths import PREPROCESSING_CONFIG_PATH


@dataclass
class TilePlan:
    is_tiled: bool
    deterministic_tiles: list[dict[str, Any]] | None  # None => WHOLE_SCENE
    learned_tiles: list[dict[str, Any]] | None  # None => WHOLE_SCENE
    total_tile_count: int
    selected_tile_count: int
    coverage_frac: float
    selection_method: str  # "whole_scene" | "tile_scorer" | "fallback_variance"
    note: str = ""


def _get_learned_tile_budget(config_path: Path | None = None) -> int:
    path = config_path or PREPROCESSING_CONFIG_PATH
    if path.exists():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            agent_cfg = cfg.get("agent") or {}
            if "learned_tool_tile_budget" in agent_cfg:
                return int(agent_cfg["learned_tool_tile_budget"])
        except Exception:
            pass
    return 4


def decide_tile_plan(
    bundle: ImageBundle,
    config_path: Path | None = None,
) -> TilePlan:
    """Computes the tiling policy for the query according to §12.

    Deterministic tools see all tiles; learned tools see a budgeted subset.
    """
    if bundle.tiles is None or bundle.tiles.tile_count <= 1:
        return TilePlan(
            is_tiled=False,
            deterministic_tiles=None,
            learned_tiles=None,
            total_tile_count=1,
            selected_tile_count=1,
            coverage_frac=1.0,
            selection_method="whole_scene",
            note="tiles: whole scene processing (below tiling threshold)",
        )

    all_tiles = list(bundle.tiles.tiles)
    total_count = bundle.tiles.tile_count
    budget = _get_learned_tile_budget(config_path)

    # Filter/rank tiles using tile_scorer
    if all_tiles and any("score" in t for t in all_tiles):
        method = "tile_scorer"
        sorted_tiles = sorted(all_tiles, key=lambda t: t.get("score", 0.0), reverse=True)
    else:
        try:
            from satquery.tools import tile_scorer

            scorer_res = tile_scorer.execute({"top_k": budget}, context={"bundle": bundle})
            score_map = {s["tile_id"]: s["score"] for s in scorer_res.get("scores", [])}
            for t in all_tiles:
                tid = t.get("tile_id", str(t))
                t["score"] = score_map.get(tid, 0.5)
            method = "tile_scorer"
            sorted_tiles = sorted(all_tiles, key=lambda t: t.get("score", 0.0), reverse=True)
        except Exception:
            method = "fallback_variance"
            sorted_tiles = sorted(
                all_tiles,
                key=lambda t: (t.get("valid_frac", 1.0), t.get("variance", 0.0)),
                reverse=True,
            )

    selected_tiles = sorted_tiles[:budget]
    selected_count = len(selected_tiles)
    coverage_frac = selected_count / total_count if total_count > 0 else 1.0

    note = (
        f"tiles: {selected_count} of {total_count} selected by {method} "
        f"(coverage {coverage_frac * 100:.0f}% of valid pixels)"
    )

    return TilePlan(
        is_tiled=True,
        deterministic_tiles=all_tiles,
        learned_tiles=selected_tiles,
        total_tile_count=total_count,
        selected_tile_count=selected_count,
        coverage_frac=coverage_frac,
        selection_method=method,
        note=note,
    )
