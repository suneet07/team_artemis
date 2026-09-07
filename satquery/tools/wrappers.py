"""``coreg_check`` and ``tile_scorer`` — P3/P4 exposed as tools (section 4.6.7).

These are thin wrappers on purpose. The plan's reason for them existing at all is
that a step which does not appear in the trace did not happen as far as a judge
is concerned: co-registration and tile selection both change the answer, so both
have to be visible in ``steps[]`` with their parameters and outputs, exactly like
any other tool.
"""

import numpy as np

from satquery.coreg import register_pair
from satquery.tiling import plan_tiles, score_tiles
from satquery.tools.base import ToolContext, ToolResult

__all__ = ["CoregCheckTool", "TileScorerTool"]


class CoregCheckTool:
    name = "coreg_check"

    def run(self, context: ToolContext) -> ToolResult:
        if len(context.scenes) < 2:
            return ToolResult(
                outputs={"coregistered": None},
                confidence=0.0,
                confidence_basis="heuristic",
                warnings=["coreg_check needs two scenes; only one was supplied"],
            )
        reference, moving = context.scenes[0], context.scenes[1]
        verify_only = bool(context.params.get("verify_only", True))

        def plane(scene) -> np.ndarray:
            stack = np.stack([np.asarray(v, dtype=np.float64) for v in scene.bands.values()])
            return stack.mean(axis=0)

        corrected, report = register_pair(
            plane(reference),
            plane(moving),
            reference_modality=reference.modality,
            moving_modality=moving.modality,
            reference_crs=reference.crs,
            moving_crs=moving.crs,
            reference_pixel_size_m=reference.pixel_size_m,
            moving_pixel_size_m=moving.pixel_size_m,
            pregeoreferenced=verify_only,
            config=context.config,
        )
        if corrected is not None:
            context.artifacts["coregistered_moving"] = corrected
        context.artifacts["coreg_report"] = report

        warnings = list(report.warnings)
        if report.refusal:
            warnings.append(report.refusal)
        confidence = 0.0 if report.refusal else (0.9 if report.coregistered else 0.4)
        return ToolResult(
            outputs={
                "coregistered": report.coregistered,
                "rmse_px": None if report.rmse_px is None else round(report.rmse_px, 4),
                "method": report.method,
                "correction_applied": report.correction_applied,
                "checks_passed": report.checks_passed,
                "verify_only": report.verify_only,
            },
            confidence=confidence,
            confidence_basis="threshold_statistics",
            warnings=warnings,
            param_provenance={"verify_only": verify_only},
        )


class TileScorerTool:
    name = "tile_scorer"

    def run(self, context: ToolContext) -> ToolResult:
        scene = context.primary
        top_k = int(context.params.get("top_k", context.config.tiling.top_k_tiles))
        plan = plan_tiles(scene.shape, scene.transform, config=context.config)

        stack = np.stack([np.asarray(v, dtype=np.float64) for v in scene.bands.values()])
        plane = stack.mean(axis=0)
        crops = []
        for tile in plan.tiles:
            row, col, height, width = tile.bounds_px
            crops.append((tile.index, plane[row : row + height, col : col + width]))
        scored = score_tiles(context.query_text, crops)

        by_index = {entry.index: entry.score for entry in scored}
        plan.tiles = [
            tile.__class__(
                index=tile.index,
                window=tile.window,
                transform=tile.transform,
                is_overview=tile.is_overview,
                score=by_index.get(tile.index, 0.0),
            )
            for tile in plan.tiles
        ]
        selected = plan.top_k(top_k)
        context.artifacts["tile_plan"] = plan
        context.artifacts["selected_tiles"] = selected

        return ToolResult(
            outputs={
                "tile_count": len(plan.tiles),
                "tile_side_px": plan.tile_side_px,
                "overlap_px": plan.overlap_px,
                "selected": [tile.as_trace_entry() for tile in selected],
                "scoring_basis": scored[0].basis if scored else "content_statistics",
            },
            confidence=0.7,
            confidence_basis="heuristic",
            warnings=list(plan.warnings),
            param_provenance={"top_k": top_k},
        )
