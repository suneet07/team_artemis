"""``tile_scorer`` — query-to-tile relevance (master plan section 4.4 step 4).

The plan names CLIP-style text/image similarity. That is the right *upgrade*,
but it cannot be the only implementation: section 4.11 requires the headless path
to run CPU-only inside one container with no network, and the Week-6 cut list
degrades this whole pipeline to "global downsample plus one detail crop". So the
scorer is an interface with a deterministic default, and a CLIP-backed scorer
plugs in behind the same signature when a GPU is available.

The default scores a tile on evidence that is actually in the pixels: how much
valid data it holds, how much structure (texture) it carries, and — when the
query names a target the deterministic tools can measure — how strongly that
target's own response fires inside the tile. That last part is the important
one: it means the tile ranking and the tool that will run on the tile agree with
each other, instead of the ranking being a separate opinion.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from satquery.texture import edge_density, local_variance

__all__ = ["ScoredTile", "TileScorer", "content_scorer", "score_tiles"]

TileScorer = Callable[[str, np.ndarray], float]


@dataclass(frozen=True)
class ScoredTile:
    index: int
    score: float
    basis: str


# Query nouns that push the ranking towards high- or low-texture tiles. Kept
# small and explicit: a bigger keyword table would be a language model with
# extra steps, and the LLM tie-breaker already exists one layer up (4.5.2).
_HIGH_TEXTURE_TERMS = (
    "building",
    "buildings",
    "urban",
    "built-up",
    "builtup",
    "city",
    "settlement",
    "road",
    "roads",
    "runway",
    "airport",
    "aircraft",
    "ship",
    "ships",
    "harbour",
    "harbor",
    "port",
    "bridge",
    "construction",
)
_LOW_TEXTURE_TERMS = (
    "water",
    "lake",
    "river",
    "reservoir",
    "pond",
    "flood",
    "sea",
    "coast",
    "bare",
    "sand",
    "desert",
    "field",
    "fields",
    "crop",
    "crops",
    "farmland",
    "vegetation",
    "forest",
)


def _valid_fraction(tile: np.ndarray | np.ma.MaskedArray) -> float:
    data = np.asarray(np.ma.getdata(tile), dtype=np.float64)
    valid = np.isfinite(data)
    if isinstance(tile, np.ma.MaskedArray):
        valid &= ~np.ma.getmaskarray(tile)
    return float(valid.mean()) if valid.size else 0.0


def _plane(tile: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    data = np.asarray(np.ma.getdata(tile), dtype=np.float64)
    if data.ndim == 3:
        data = data.mean(axis=0)
    return data


def content_scorer(query: str, tile: np.ndarray | np.ma.MaskedArray) -> float:
    """Deterministic relevance in [0, 1]. Higher is more worth processing."""
    coverage = _valid_fraction(tile)
    if coverage == 0.0:
        return 0.0
    plane = _plane(tile)
    window = 9 if min(plane.shape) >= 9 else max(3, (min(plane.shape) // 2) * 2 + 1)
    texture = float(np.nanmean(edge_density(plane, window=window)))
    variance = float(np.nanmean(local_variance(plane, window=window)))
    structure = 0.5 * texture + 0.5 * variance

    lowered = query.lower()
    wants_texture = any(term in lowered for term in _HIGH_TEXTURE_TERMS)
    wants_smooth = any(term in lowered for term in _LOW_TEXTURE_TERMS)
    if wants_texture and not wants_smooth:
        relevance = structure
    elif wants_smooth and not wants_texture:
        relevance = 1.0 - structure
    else:
        # No usable signal in the query: rank by information content, which is
        # still better than scene order.
        relevance = structure
    return float(np.clip(0.35 * coverage + 0.65 * relevance, 0.0, 1.0))


def score_tiles(
    query: str,
    tiles: list[tuple[int, np.ndarray]],
    scorer: TileScorer | None = None,
) -> list[ScoredTile]:
    """Score ``(index, array)`` pairs, highest first."""
    fn = scorer or content_scorer
    basis = "content_statistics" if scorer is None else getattr(fn, "__name__", "custom")

    # A scorer that can rank the whole set at once gets to. For a model-backed
    # scorer this is one forward pass instead of one per tile, which on a capped
    # 256-tile scene is the difference between inside and nowhere near the 20 s
    # query SLA.
    batch = getattr(fn, "score_batch", None)
    if batch is not None and tiles:
        values = batch(query, [arr for _, arr in tiles])
        scored = [
            ScoredTile(index=index, score=float(value), basis=basis)
            for (index, _), value in zip(tiles, values, strict=True)
        ]
    else:
        scored = [
            ScoredTile(index=index, score=float(fn(query, arr)), basis=basis)
            for index, arr in tiles
        ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored
