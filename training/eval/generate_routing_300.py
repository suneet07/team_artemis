"""Build the 300-query routing evaluation set (master plan Phase 0, item 1).

The plan is specific about why this exists and when: it is built **in the same
sitting as the task enum freeze**, and zero-shot routing accuracy — rules alone,
LLM alone, hybrid — is measured in **Week 1, not Week 5**. The risk register's
trigger is concrete: *hybrid routing below 90% on unambiguous cases means the
rules get fixed now*. A set that cannot detect that is worse than no set.

Design rules this generator holds to, each of which the first version broke:

* **Ground truth is authored, never sampled.** Each case is a hand-specified
  (query shape, input context, expected task) triple. Drawing the expected task
  or the expected router path at random produces a file that looks like a
  dataset and measures nothing.
* **Deterministic.** One fixed seed, no shuffling that depends on hash order,
  and a byte-identical file on every run. A routing accuracy that moves because
  the eval set moved is not a measurement.
* **Repo-relative paths.** The first version wrote to a hard-coded path on one
  developer's machine.
* **Real variety.** Templates with slot fills rather than 32 strings sampled 300
  times, so a rule that pattern-matches one phrasing does not score 100%.
* **The hard cases are the point.** Refusals, D3 band-availability reroutes,
  single-pol and pan-only stress cases, and genuinely ambiguous phrasing all
  carry their expected outcome, because those are the cases section 6.2's
  invalid-config catch rate is measured on.

Run: ``python -m training.eval.generate_routing_300`` (or execute this file).
"""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from satquery.agent.task_enum import RouterPath, Task

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "training" / "eval" / "routing_300.jsonl"
SEED = 26167  # the problem statement number, so the seed is not arbitrary either

# --------------------------------------------------------------------------
# Input contexts. Each is a realistic sensor configuration, named so the
# expected outcome can be read against it.
# --------------------------------------------------------------------------

CARTOSAT_MX = {
    "modalities": ["optical"],
    "bands": ["blue", "green", "red", "nir"],
    "computable_indices": ["NDVI", "NDWI"],
    "crs": "EPSG:32644",
    "image_count": 1,
    "sensor": "Cartosat-2S MX",
    "native_gsd_m": 1.6,
}
SENTINEL2_FULL = {
    "modalities": ["optical"],
    "bands": ["blue", "green", "red", "nir", "swir"],
    "computable_indices": ["NDVI", "NDWI", "MNDWI", "NDBI"],
    "crs": "EPSG:32643",
    "image_count": 1,
    "sensor": "Sentinel-2",
    "native_gsd_m": 10.0,
}
RGB_ONLY = {
    "modalities": ["optical"],
    "bands": ["blue", "green", "red"],
    "computable_indices": [],
    "crs": "EPSG:32644",
    "image_count": 1,
    "sensor": "RGB orthophoto",
    "native_gsd_m": 0.5,
}
PAN_ONLY = {
    "modalities": ["optical"],
    "bands": ["pan"],
    "computable_indices": [],
    "crs": "EPSG:32644",
    "image_count": 1,
    "sensor": "Cartosat-2S PAN",
    "native_gsd_m": 0.65,
}
SAR_DUAL = {
    "modalities": ["sar"],
    "bands": ["VV", "VH"],
    "polarisations": ["VV", "VH"],
    "sar_band": "C",
    "computable_indices": [],
    "crs": "EPSG:32644",
    "image_count": 1,
    "sensor": "Sentinel-1 IW",
    "native_gsd_m": 10.0,
}
SAR_SINGLE_X = {
    "modalities": ["sar"],
    "bands": ["HH"],
    "polarisations": ["HH"],
    "sar_band": "X",
    "computable_indices": [],
    "crs": "EPSG:32644",
    "image_count": 1,
    "sensor": "RISAT-2B spotlight",
    "native_gsd_m": 1.0,
}
OPTICAL_PAIR = {**CARTOSAT_MX, "image_count": 2, "dates": ["2023-01-14", "2024-01-19"]}
SENTINEL2_PAIR = {**SENTINEL2_FULL, "image_count": 2, "dates": ["2022-11-03", "2024-11-08"]}
SAR_PAIR = {**SAR_DUAL, "image_count": 2, "dates": ["2023-03-02", "2024-03-06"]}
CROSS_MODAL = {
    "modalities": ["optical", "sar"],
    "bands": ["blue", "green", "red", "nir", "VV", "VH"],
    "polarisations": ["VV", "VH"],
    "sar_band": "C",
    "computable_indices": ["NDVI", "NDWI"],
    "crs": "EPSG:32644",
    "image_count": 2,
    "sensor": "Cartosat-2S MX + Sentinel-1",
}

# --------------------------------------------------------------------------
# Template banks. Slots keep the phrasing varied without inventing semantics.
# --------------------------------------------------------------------------

_WATER = ["water body", "lake", "river", "reservoir", "flooded area"]
_BUILT = ["built-up area", "buildings", "urban extent", "settlement"]
_VEG = ["vegetation", "forest cover", "cropland", "tree cover"]
_UNCOVERED = ["storage tanks", "bridges", "vehicles", "harbour cranes", "oil tanks"]
_COVERED = ["buildings", "aircraft", "ships", "the water body", "the forest area"]

TEMPLATES: dict[Task, list[str]] = {
    Task.SINGLE_VQA: [
        "Is there a {water} in this image?",
        "How many separate {built} regions are visible?",
        "Is the area mostly {veg}?",
        "What is the predominant land cover here?",
        "Does this scene contain any {built}?",
        "Is more than half of this scene {veg}?",
        "Are there roads crossing this scene?",
    ],
    Task.SINGLE_CAPTION: [
        "Describe the land cover in this scene.",
        "Provide a caption for this satellite image.",
        "Summarise what is visible in this acquisition.",
        "Describe the contents of this image.",
        "Write a short description of this scene.",
    ],
    Task.SINGLE_GROUNDING: [
        "Where is the {water}?",
        "Locate all {covered} in this image.",
        "Provide a bounding box for the {built}.",
        "Highlight the {veg} in this scene.",
        "Find the {water} and mark it.",
        "Segment the {built} in this acquisition.",
    ],
    Task.CHANGE_DESCRIPTION: [
        "Describe the changes between the two acquisitions.",
        "Describe what is different between the earlier and later image.",
        "Summarise how this area changed over time.",
        "Describe the differences visible across the two dates.",
    ],
    Task.CHANGE_VQA: [
        "Have new {built} appeared between the two dates?",
        "Is there evidence of {veg} loss between the acquisitions?",
        "Did the {water} extent change between the two dates?",
        "Has anything changed in this area since the first acquisition?",
    ],
    Task.CHANGE_MAP: [
        "Generate a change mask for this area.",
        "Produce a raster of the changed pixels.",
        "Give me a segmentation of the change between the two dates.",
        "Produce a map of new construction between the acquisitions.",
    ],
    Task.CROSSMODAL_EXTRACTION: [
        "Produce a {water} mask using both the optical and the SAR acquisition.",
        "Give me a raster of the {built} combining radar and optical evidence.",
        "Delineate the {water} from both sensors.",
        "What fraction of this scene is {water}, using optical and SAR together?",
    ],
    Task.CROSSMODAL_VQA: [
        "Is the {water} visible in both the optical and the SAR image?",
        "Do the radar and optical acquisitions agree about the {built}?",
        "Does the SAR show {water} where the optical does not?",
        "Is there {built} that appears in the radar but not the optical image?",
    ],
}

_SLOTS = {"water": _WATER, "built": _BUILT, "veg": _VEG, "covered": _COVERED}


@dataclass
class Case:
    """One authored routing case."""

    query_text: str
    input_context: dict
    ground_truth: dict
    id: str = ""
    notes: str = ""


@dataclass
class CaseSpec:
    """A family of cases: one task, one context, one expected outcome."""

    task: Task
    context: dict
    category: str
    expected_tools: list[str] = field(default_factory=list)
    router_path: RouterPath = RouterPath.RULES
    refusal_category: str | None = None
    templates: list[str] | None = None
    unambiguous: bool = True
    note: str = ""


#: Deterministic tools the rules router is expected to plan for each case family.
#: Learned tools are listed too — the manifest exists whether or not the adapter
#: has landed, and the router plans against manifests.
SPECS: list[CaseSpec] = [
    # ---- single image, multispectral: the happy path -----------------------
    CaseSpec(Task.SINGLE_VQA, CARTOSAT_MX, "valid", ["rs_vqa"]),
    CaseSpec(Task.SINGLE_VQA, SENTINEL2_FULL, "valid", ["rs_vqa"]),
    CaseSpec(Task.SINGLE_CAPTION, CARTOSAT_MX, "valid", ["rs_vqa"]),
    CaseSpec(Task.SINGLE_CAPTION, SAR_DUAL, "valid", ["rs_vqa"]),
    CaseSpec(
        Task.SINGLE_GROUNDING,
        CARTOSAT_MX,
        "valid",
        ["rs_ground_caption"],
        note="covered vocabulary: regions, buildings, aircraft, ships",
    ),
    CaseSpec(Task.SINGLE_GROUNDING, SENTINEL2_FULL, "valid", ["rs_ground_caption"]),
    # ---- grounding outside the trained vocabulary (C39) ---------------------
    CaseSpec(
        Task.SINGLE_GROUNDING,
        CARTOSAT_MX,
        "deterministic_fallback",
        ["object_box_fallback"],
        templates=[f"Locate the {noun} in this image." for noun in _UNCOVERED]
        + [f"Where are the {noun}?" for noun in _UNCOVERED],
        note="uncovered class: must reach object_box_fallback, never learned grounding",
    ),
    # ---- change: two images -------------------------------------------------
    CaseSpec(Task.CHANGE_VQA, OPTICAL_PAIR, "valid", ["change_map", "change_stats"]),
    CaseSpec(Task.CHANGE_DESCRIPTION, SENTINEL2_PAIR, "valid", ["change_map"]),
    CaseSpec(Task.CHANGE_MAP, OPTICAL_PAIR, "valid", ["change_map", "change_stats"]),
    CaseSpec(Task.CHANGE_VQA, SAR_PAIR, "valid", ["change_map", "change_stats"]),
    # ---- cross-modal --------------------------------------------------------
    CaseSpec(Task.CROSSMODAL_EXTRACTION, CROSS_MODAL, "valid", ["coreg_check"]),
    CaseSpec(Task.CROSSMODAL_VQA, CROSS_MODAL, "valid", ["coreg_check"]),
    # ---- refusals: the invalid-config catch rate (section 6.2) --------------
    CaseSpec(
        Task.CHANGE_VQA,
        CARTOSAT_MX,
        "refusal",
        [],
        refusal_category="missing_input",
        templates=[
            "Have new buildings appeared since the earlier acquisition?",
            "Describe the changes in this area over time.",
            "Did the water extent change between the two dates?",
            "What is different compared with before?",
        ],
        note="change language, one image: refuse and request the second acquisition",
    ),
    CaseSpec(
        Task.SINGLE_VQA,
        SAR_DUAL,
        "refusal",
        [],
        refusal_category="modality_limitation",
        templates=[
            "What colour is the water in this image?",
            "How green is the vegetation here?",
            "What shade is the built-up area?",
            "Is the colour of this field consistent?",
        ],
        note="SAR carries no colour information at all",
    ),
    CaseSpec(
        Task.SINGLE_VQA,
        PAN_ONLY,
        "refusal",
        [],
        refusal_category="modality_limitation",
        templates=[
            "Is the vegetation healthy in this scene?",
            "What is the NDVI of this area?",
            "How much chlorophyll does this canopy show?",
            "Is there moisture stress in these fields?",
        ],
        note="pan-only optical cannot answer a spectral question (4.5.3)",
    ),
    # ---- D3 band-availability reroutes -------------------------------------
    CaseSpec(
        Task.SINGLE_VQA,
        RGB_ONLY,
        "reroute_d3",
        ["texture_seg", "rs_vqa"],
        templates=[
            "How much of this scene is built-up?",
            "Is there a large settlement in this image?",
            "Are there dense buildings here?",
            "What fraction of this scene is urban?",
        ],
        note="no NIR or SWIR: falls back to texture and morphology evidence",
    ),
    CaseSpec(
        Task.SINGLE_VQA,
        CARTOSAT_MX,
        "reroute_d3",
        ["spectral_index", "rs_vqa"],
        templates=[
            "Is there open water in this scene?",
            "How much water does this image contain?",
            "Is the lake still present?",
            "Does this scene contain a reservoir?",
        ],
        note="no SWIR: MNDWI unavailable, NDWI substituted (D3)",
    ),
    # ---- single-pol / X-band stress ----------------------------------------
    CaseSpec(
        Task.SINGLE_VQA,
        SAR_SINGLE_X,
        "stress_single_pol",
        ["sar_backscatter", "rs_vqa"],
        templates=[
            "Is there water in this radar scene?",
            "What is the polarimetric signature of this area?",
            "How much of this scene is built-up according to the radar?",
            "Does the backscatter indicate a smooth surface?",
        ],
        note="single-pol X-band: answer what one channel supports, state the limit",
    ),
    # ---- genuinely ambiguous: the LLM tie-breaker's whole jurisdiction ------
    CaseSpec(
        Task.SINGLE_VQA,
        CARTOSAT_MX,
        "ambiguous",
        ["rs_vqa"],
        router_path=RouterPath.LLM,
        unambiguous=False,
        templates=[
            "Describe this scene — how many buildings are there?",
            "Caption this image and tell me if there is water?",
            "Describe what you see, and is it urban?",
            "Summarise the scene: is there any forest?",
        ],
        note="caption and question phrasing both present; rules must defer",
    ),
]


def _fill(template: str, rng: random.Random) -> str:
    text = template
    for slot, values in _SLOTS.items():
        token = "{" + slot + "}"
        while token in text:
            text = text.replace(token, rng.choice(values), 1)
    return text


def build_cases(total: int = 300, seed: int = SEED) -> list[Case]:
    rng = random.Random(seed)
    per_spec = total // len(SPECS)
    remainder = total % len(SPECS)

    cases: list[Case] = []
    for position, spec in enumerate(SPECS):
        count = per_spec + (1 if position < remainder else 0)
        templates = spec.templates or TEMPLATES[spec.task]
        produced: list[Case] = []
        seen: set[str] = set()
        # Prefer unseen phrasings; once the bank is exhausted, repeats are fine —
        # what must not happen is 300 rows drawn from a handful of strings.
        for attempt in range(count * 40):
            if len(produced) >= count:
                break
            text = _fill(rng.choice(templates), rng)
            if text in seen and attempt < count * 20:
                continue
            seen.add(text)
            produced.append(
                Case(
                    query_text=text,
                    input_context=dict(spec.context),
                    ground_truth={
                        "expected_task": spec.task.value,
                        "expected_tools": list(spec.expected_tools),
                        "expected_router_path": spec.router_path.value,
                        "expected_category": spec.category,
                        "expected_refusal_category": spec.refusal_category,
                        "unambiguous": spec.unambiguous,
                    },
                    notes=spec.note,
                )
            )
        cases.extend(produced)

    # Deterministic interleave: sort by a seeded key rather than shuffling in
    # place, so the file is byte-identical on every run and on every platform.
    keyed = sorted(cases, key=lambda case: rng.random())
    for index, case in enumerate(keyed, start=1):
        case.id = f"q_{index:04d}"
    return keyed


def write_dataset(path: Path = OUTPUT_PATH, total: int = 300, seed: int = SEED) -> Path:
    cases = build_cases(total=total, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            record = asdict(case)
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "query_text": record["query_text"],
                        "input_context": record["input_context"],
                        "ground_truth": record["ground_truth"],
                        "notes": record["notes"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


if __name__ == "__main__":
    written = write_dataset()
    print(f"wrote {written} ({sum(1 for _ in written.open(encoding='utf-8'))} queries)")
