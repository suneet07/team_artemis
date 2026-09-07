from typing import TypedDict


class RefusalWithRemedy(TypedDict):
    """
    Structured refusal providing a reason and a remedy.
    """
    category: str
    reason: str
    remedy: str


class RefusalCategory:
    """Frozen Refusal Categories as defined in the frontend contract."""
    PARAMETER_GATE = "parameter_gate"
    VALIDATOR = "validator"
    MODALITY_LIMITATION = "modality_limitation"
    MISSING_INPUT = "missing_input"
    UNSUPPORTED_CLASS = "unsupported_class"


def create_refusal(category: str, reason: str, remedy: str) -> RefusalWithRemedy:
    """
    Creates a standardized refusal dictionary.

    `remedy` is for the UI: it tells the frontend which affordance to offer
    ("add_input" -> show the second-image slot). It is deliberately NOT part of
    the trace payload -- see `trace_refusal`.
    """
    return {
        "category": category,
        "reason": reason,
        "remedy": remedy
    }


def trace_refusal(refusal: RefusalWithRemedy | dict[str, str] | None) -> dict[str, str] | None:
    """Reduce a refusal to the two fields the frozen trace schema allows.

    The schema's `Refusal` is `additionalProperties: false` with exactly `reason`
    and `category`, so writing the UI's `remedy` field into the trace made
    `TraceBuilder.build()` fail schema validation -- turning every graceful
    refusal, which the plan says scores better than a confident hallucination,
    into a hard error instead.
    """
    if not refusal:
        return None
    return {"reason": refusal["reason"], "category": refusal["category"]}
