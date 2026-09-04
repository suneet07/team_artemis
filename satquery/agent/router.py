import re
from typing import Any

from satquery.agent.task_enum import RouterPath, Task

# §10 N2 temporal keywords
TEMPORAL_KEYWORDS = {
    "before",
    "after",
    "change",
    "changed",
    "changes",
    "between",
    "since",
    "growth",
    "new",
    "removed",
    "demolished",
    "expanded",
    "modified",
    "deforestation",
    "constructed",
    "differences",
    "over time",
}

CROSSMODAL_PATTERNS = [
    "across modalities",
    "both optical and sar",
    "combining radar and optical",
    "combining optical and sar",
    "sar but not optical",
    "optical but not sar",
    "radar and optical",
    "optical and radar",
    "between the two captures",
    "in both optical and sar",
    "two captures",
]

DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4})\b",
    re.IGNORECASE,
)


def _has_temporal_signal(text: str) -> bool:
    lower = text.lower()
    if any(re.search(r"\b" + re.escape(w) + r"\b", lower) for w in TEMPORAL_KEYWORDS):
        return True
    dates = DATE_PATTERN.findall(text)
    return len(dates) >= 2


def _has_crossmodal_signal(text: str) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in CROSSMODAL_PATTERNS)


def route_query_rules(
    question: str,
    modalities: list[str] | None = None,
    pair_type: str | None = None,
    image_count: int = 1,
) -> tuple[Task, list[str]]:
    """Stage 1: Deterministic rules (R1–R8) for task selection.

    Returns (task, notes).
    """
    ql = question.lower().strip()
    notes: list[str] = []
    modalities_set = set(modalities or [])
    is_multi_modal = (
        len(modalities_set) > 1
        or _has_crossmodal_signal(question)
        or pair_type == "crossmodal"
    )

    # Cross-modal rules (R4, R5)
    if is_multi_modal:
        extraction_markers = [
            "extract",
            "footprint",
            "combining",
            "combine",
            "identify urban features",
            "find structures",
            "measure",
        ]
        if any(marker in ql for marker in extraction_markers):
            notes.append("Routed to crossmodal_extraction via cross-modal extraction markers (R4)")
            return Task.CROSSMODAL_EXTRACTION, notes
        notes.append("Routed to crossmodal_vqa via multi-modal pair context (R5)")
        return Task.CROSSMODAL_VQA, notes

    # Change / bi-temporal rules (R1, R2, R3)
    has_temporal = (
        _has_temporal_signal(question)
        or pair_type == "bitemporal"
        or image_count >= 2
    )

    explicit_map = any(
        k in ql for k in ("raster", "change mask", "highlighting the differences", "map of")
    )

    if has_temporal or explicit_map:
        map_markers = [
            "raster",
            "mask",
            "map of",
            "map",
            "highlighting the differences",
            "how much",
            "how many",
            "ratio",
            "count",
        ]
        if any(m in ql for m in map_markers) or re.search(r"\barea\b", ql):
            notes.append("Routed to change_map via temporal + map/statistic markers (R1)")
            return Task.CHANGE_MAP, notes

        desc_markers = [
            "describe the changes",
            "what changed",
            "describe",
            "how has",
            "what differences",
            "areas that have been modified",
        ]
        if any(m in ql for m in desc_markers):
            notes.append("Routed to change_description via temporal description markers (R2)")
            return Task.CHANGE_DESCRIPTION, notes

        notes.append("Routed to change_vqa via temporal question markers (R3)")
        return Task.CHANGE_VQA, notes

    # Single-image rules (R6, R7, R8)
    grounding_markers = [
        "where is",
        "where are",
        "locate",
        "find the",
        "find all",
        "bounding box",
        "bbox",
        "highlight the",
        "highlight",
        "show me",
    ]
    if any(m in ql for m in grounding_markers):
        notes.append("Routed to single_grounding via localization markers (R6)")
        return Task.SINGLE_GROUNDING, notes

    caption_markers = [
        "describe the contents",
        "description of",
        "describe",
        "caption",
        "summarise",
        "summarize",
        "what does this region look like",
    ]
    if any(m in ql for m in caption_markers):
        notes.append("Routed to single_caption via captioning markers (R7)")
        return Task.SINGLE_CAPTION, notes

    notes.append("Routed to single_vqa by default (R8)")
    return Task.SINGLE_VQA, notes


def route_query(
    question: str,
    modalities: list[str] | None = None,
    pair_type: str | None = None,
    image_count: int = 1,
    *,
    allow_llm: bool = True,
    llm_client: Any | None = None,
) -> tuple[Task, RouterPath, list[str]]:
    """Routes a query using Stage 1 rules, with fallback/tie-break handling.

    Returns (task, router_path, routing_notes).
    """
    # Deterministic Stage 1
    task, notes = route_query_rules(
        question=question,
        modalities=modalities,
        pair_type=pair_type,
        image_count=image_count,
    )
    return task, RouterPath.RULES, notes
