import json
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


ROUTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "enum": [
                "single_vqa",
                "single_caption",
                "single_grounding",
                "change_description",
                "change_vqa",
                "change_map",
                "crossmodal_extraction",
                "crossmodal_vqa",
            ],
        },
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["task", "reason"],
    "additionalProperties": False,
}


def _is_ambiguous(question: str, notes: list[str]) -> bool:
    """Detects whether Stage 1 rules had ambiguous tie or fallback default."""
    ql = question.lower()
    has_loc = any(w in ql for w in ("where", "locate", "find", "bounding box"))
    has_desc = any(w in ql for w in ("describe", "caption", "what does", "summarize"))
    has_vqa = any(w in ql for w in ("is there", "are there", "how many", "what is"))
    signals = sum([bool(has_loc), bool(has_desc), bool(has_vqa)])
    if signals >= 2:
        return True
    if any("default (R8)" in n for n in notes):
        return True
    return False


def route_query_llm(
    question: str,
    modalities: list[str] | None = None,
    pair_type: str | None = None,
    image_count: int = 1,
    llm_client: Any | None = None,
) -> tuple[Task, str] | None:
    """Stage 2: Constrained LLM decoding (Outlines/vLLM guided) for ambiguous queries."""
    if llm_client is not None:
        try:
            if callable(llm_client):
                res = llm_client(question)
            elif hasattr(llm_client, "generate"):
                res = llm_client.generate(question, schema=ROUTER_JSON_SCHEMA)
            elif hasattr(llm_client, "chat"):
                res = llm_client.chat(question)
            else:
                res = None

            if isinstance(res, str):
                res = json.loads(res)
            if isinstance(res, dict) and "task" in res and "reason" in res:
                task = Task(res["task"])
                return task, str(res["reason"])
        except Exception:
            return None

    # Stage 2 path: try constrained JSON decoding via outlines (Rule 4)
    # outlines uses structured generation to guarantee the output matches ROUTER_JSON_SCHEMA.
    # If outlines is not installed, or the model is unavailable, we return None and the
    # caller falls back to Stage 1 rules (Rule 5: rules first, LLM second).
    try:
        import outlines  # type: ignore  # noqa: F401
        import outlines.generate  # type: ignore
        import outlines.models  # type: ignore

        # Build a prompt that includes task vocabulary context
        task_enum_str = ", ".join([
            "single_vqa", "single_caption", "single_grounding",
            "change_description", "change_vqa", "change_map",
            "crossmodal_extraction", "crossmodal_vqa",
        ])
        context_parts: list[str] = []
        if modalities:
            context_parts.append(f"Modalities: {', '.join(modalities)}")
        if pair_type:
            context_parts.append(f"Pair type: {pair_type}")
        if image_count > 1:
            context_parts.append(f"Image count: {image_count}")
        context_str = ". ".join(context_parts)

        prompt = (
            f"You are a satellite image query router. Choose the correct task type.\n"
            f"Tasks: {task_enum_str}.\n"
            f"{context_str}\n"
            f'Query: "{question}"\n'
            f"Respond with JSON: {{\"task\": \"<task>\", \"reason\": \"<brief reason>\"}}"
        )

        # Try to find a loaded outlines-compatible model (e.g. vLLM backend)
        # outlines.models.get_model() returns None if no model is registered
        model = getattr(outlines.models, "get_model", lambda: None)()
        if model is None:
            return None  # No model available; fall back to rules

        generator = outlines.generate.json(model, ROUTER_JSON_SCHEMA)
        raw = generator(prompt)
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict) and "task" in raw and "reason" in raw:
            task = Task(raw["task"])
            return task, str(raw["reason"])
    except ImportError:
        # outlines not installed — this is expected in CPU CI environments
        pass
    except Exception:
        # Model unavailable or generation failed — fall back silently
        pass

    return None



def route_query(
    question: str,
    modalities: list[str] | None = None,
    pair_type: str | None = None,
    image_count: int = 1,
    *,
    allow_llm: bool = True,
    llm_client: Any | None = None,
) -> tuple[Task, RouterPath, list[str]]:
    """Routes a query using Stage 1 rules, with Stage 2 LLM tie-break when ambiguous.

    Returns (task, router_path, routing_notes).
    """
    task, notes = route_query_rules(
        question=question,
        modalities=modalities,
        pair_type=pair_type,
        image_count=image_count,
    )

    if allow_llm and (llm_client is not None or _is_ambiguous(question, notes)):
        llm_res = route_query_llm(
            question=question,
            modalities=modalities,
            pair_type=pair_type,
            image_count=image_count,
            llm_client=llm_client,
        )
        if llm_res is not None:
            llm_task, reason = llm_res
            return llm_task, RouterPath.LLM, notes + [f"Stage 2 LLM tie-break: {reason}"]
        else:
            notes.append("Stage 2 LLM tie-break skipped or unavailable; fell back to Stage 1 rules")

    return task, RouterPath.RULES, notes
