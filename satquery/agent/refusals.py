from typing import Any, Literal

RefusalCategory = Literal[
    "parameter_gate",
    "validator",
    "modality_limitation",
    "missing_input",
    "unsupported_class",
]

ACTION_ADD_2ND_IMAGE = "add_second_image"

RemedyAction = Literal[
    "add_optical",
    "add_sar",
    "ask_different_question",
    "reupload",
    "none",
] | str

# Per-task representative suggested questions drawn from supported_tasks vocabulary (§20).
# These are displayed in the UI when a task is refused, helping users rephrase their query.
_TASK_SUGGESTED_QUESTIONS: dict[str, list[str]] = {
    "single_vqa": [
        "Is there a water body in this image?",
        "What land cover types are visible?",
    ],
    "single_caption": [
        "Describe the contents of this satellite image.",
        "Summarize the land use visible in this scene.",
    ],
    "single_grounding": [
        "Where are the buildings in this scene?",
        "Locate the water bodies.",
    ],
    "change_description": [
        "What changed between these two dates?",
        "Describe the changes in land cover.",
    ],
    "change_vqa": [
        "Did the vegetation cover decrease?",
        "Is there evidence of urban expansion?",
    ],
    "change_map": [
        "How much area changed between the two acquisitions?",
        "What is the ratio of changed area?",
    ],
    "crossmodal_extraction": [
        "Extract built-up areas using optical and SAR.",
        "Identify flooded regions across optical and SAR.",
    ],
    "crossmodal_vqa": [
        "What does the optical show compared to SAR?",
        "Are optical and SAR results consistent for water?",
    ],
}


def get_suggested_questions(
    supported_tasks: list[str] | None = None,
    max_questions: int = 3,
) -> list[str]:
    """Returns suggested rephrasing questions drawn from the supported-task vocabulary (§20).

    If `supported_tasks` is provided, only questions for those tasks are returned.
    Otherwise questions from all tasks are sampled.
    """
    tasks = supported_tasks or list(_TASK_SUGGESTED_QUESTIONS.keys())
    questions: list[str] = []
    for task in tasks:
        for q in _TASK_SUGGESTED_QUESTIONS.get(task, []):
            if q not in questions:
                questions.append(q)
            if len(questions) >= max_questions:
                return questions
    return questions


def create_refusal(
    category: RefusalCategory,
    reason: str,
    action: RemedyAction = "none",
    label: str = "",
    suggested_questions: list[str] | None = None,
    supported_tasks: list[str] | None = None,
) -> dict[str, Any]:
    """Builds a frozen RefusalWithRemedy structure for the API and trace.

    If `suggested_questions` is not explicitly provided, derives them from `supported_tasks`
    so users see actionable alternatives rather than hard-coded strings (§20).
    """
    refusal: dict[str, Any] = {
        "category": category,
        "reason": reason,
        "remedy": {
            "action": action,
            "label": label or action.replace("_", " ").capitalize(),
        },
    }
    # Derive suggested questions from supported_tasks if not explicitly supplied
    questions = suggested_questions
    if questions is None and supported_tasks is not None:
        questions = get_suggested_questions(supported_tasks=supported_tasks)
    if questions:
        refusal["remedy"]["suggested_questions"] = questions
    return refusal
