"""Choose *which* patches and questions enter a corpus, and why.

Taking the first N rows of an annotation file is the default and it is wrong
here. reBEN files are ordered by patch id, which sorts by tile, then sensing
date, then grid position -- so the head of the file is one tile, on one day, in
one country, in one season. Measured on the existing micro-split: **21 patches,
1 tile of 54, 1 date, and a contiguous 4x8 grid block**, which at 1.2 km per
patch is a single 5x10 km rectangle in Austria on a June morning. The full
dataset offers 10 countries, 54 tiles and all 12 months.

Three things follow, and this module does all three.

**Stratify by country and season.** A model trained on one country in one season
learns that country in that season. The Indian hidden set shares neither.

**Space the patches out.** Neighbouring grid cells are 1.2 km apart, same
sensor, same day -- near-duplicate imagery carrying near-duplicate labels. They
inflate the row count without adding information, and they leak between train
and validation splits drawn from the same tile. :func:`spread_within_tile`
enforces a minimum grid separation.

**Balance question types.** BEN.txt mixes yes/no, multiple choice, bounding box,
referring and captioning in one file. Sampling rows without regard to type gives
whatever mixture the file happens to have, which is not the mixture any adapter
wants.

Everything here is seeded and deterministic, so four people staging on four
accounts build the same corpus. A corpus that differs per teammate makes their
adapters incomparable.
"""

import hashlib
from collections import defaultdict
from typing import Any

__all__ = [
    "MIN_GRID_SEPARATION",
    "SEASONS",
    "season_of",
    "spread_within_tile",
    "stratified_patches",
    "balance_question_types",
    "coverage_report",
]

#: Minimum Chebyshev distance, in patches, between two chosen patches of the
#: same tile and date. 3 puts ~3.6 km between chips, which is enough that they
#: are not looking at the same fields.
MIN_GRID_SEPARATION = 3

SEASONS = {
    "12": "winter", "01": "winter", "02": "winter",
    "03": "spring", "04": "spring", "05": "spring",
    "06": "summer", "07": "summer", "08": "summer",
    "09": "autumn", "10": "autumn", "11": "autumn",
}


def _seed(*parts: str) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(parts).encode()).digest()[:8], "big"
    )


def season_of(patch_id: str) -> str:
    """Season from the sensing date embedded in the patch id."""
    try:
        return SEASONS.get(patch_id.split("_")[2][4:6], "unknown")
    except IndexError:
        return "unknown"


def _grid(patch_id: str) -> tuple[int, int] | None:
    parts = patch_id.split("_")
    try:
        return int(parts[-2]), int(parts[-1])
    except (ValueError, IndexError):
        return None


def spread_within_tile(
    patch_ids: list[str], separation: int = MIN_GRID_SEPARATION
) -> list[str]:
    """Drop patches that sit within ``separation`` cells of one already kept.

    Greedy and order-dependent, so the caller shuffles deterministically first.
    Patches whose id carries no grid position are kept -- this filter is about
    known adjacency, and dropping unknowns would silently shrink the corpus.
    """
    kept: list[str] = []
    taken: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for patch_id in patch_ids:
        cell = _grid(patch_id)
        if cell is None:
            kept.append(patch_id)
            continue
        # Same tile *and* same date: a different date over the same ground is a
        # genuinely different observation and is not a duplicate.
        parts = patch_id.split("_")
        key = f"{parts[5]}_{parts[2]}" if len(parts) > 5 else patch_id
        if any(
            max(abs(cell[0] - r), abs(cell[1] - c)) < separation for r, c in taken[key]
        ):
            continue
        taken[key].append(cell)
        kept.append(patch_id)
    return kept


def stratified_patches(
    metadata: list[dict[str, Any]],
    target: int,
    *,
    strata_keys: tuple[str, ...] = ("country", "season"),
    separation: int = MIN_GRID_SEPARATION,
    seed: str = "satquery",
) -> list[str]:
    """Pick ``target`` patch ids spread evenly across the strata.

    ``metadata`` rows need ``patch_id`` and whichever ``strata_keys`` are named;
    ``season`` is derived when absent. Strata are filled round-robin so a
    country with 155k patches does not swamp one with 8k -- the aim is coverage,
    not proportional representation, because the hidden set resembles neither
    distribution.
    """
    import random

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for row in metadata:
        patch_id = row["patch_id"]
        key = tuple(
            row.get(name) or (season_of(patch_id) if name == "season" else "unknown")
            for name in strata_keys
        )
        buckets[key].append(patch_id)

    for key, ids in buckets.items():
        rng = random.Random(_seed(seed, *map(str, key)))
        rng.shuffle(ids)
        buckets[key] = spread_within_tile(ids, separation)

    chosen: list[str] = []
    order = sorted(buckets)
    index = 0
    while len(chosen) < target:
        progressed = False
        for key in order:
            pool = buckets[key]
            if index < len(pool):
                chosen.append(pool[index])
                progressed = True
                if len(chosen) >= target:
                    break
        if not progressed:
            break  # every stratum exhausted; the corpus is as big as it can be
        index += 1
    return chosen


def balance_question_types(
    rows: list[dict[str, Any]],
    per_type: dict[str, int],
    *,
    seed: str = "satquery",
) -> list[dict[str, Any]]:
    """Take at most ``per_type[type]`` rows of each question type.

    Missing types are dropped rather than back-filled from another type. A quota
    silently met with the wrong question shape is worse than a smaller corpus,
    because nothing downstream can see it happened.
    """
    import random

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row.get("question_type", "other")].append(row)

    out: list[dict[str, Any]] = []
    for question_type, limit in per_type.items():
        pool = by_type.get(question_type, [])
        rng = random.Random(_seed(seed, question_type))
        rng.shuffle(pool)
        out.extend(pool[:limit])
    return out


def coverage_report(patch_ids: list[str], metadata: dict[str, dict]) -> dict[str, Any]:
    """What the chosen set actually spans. Print it; do not assume it."""
    from collections import Counter

    countries = Counter()
    seasons = Counter()
    tiles = Counter()
    for patch_id in patch_ids:
        row = metadata.get(patch_id, {})
        countries[row.get("country", "unknown")] += 1
        seasons[season_of(patch_id)] += 1
        parts = patch_id.split("_")
        tiles[parts[5] if len(parts) > 5 else "?"] += 1
    return {
        "patches": len(patch_ids),
        "countries": dict(countries),
        "seasons": dict(seasons),
        "tiles": len(tiles),
        "largest_tile_share": (
            round(max(tiles.values()) / len(patch_ids), 3) if patch_ids else 0.0
        ),
    }
