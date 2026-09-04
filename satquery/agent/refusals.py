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


def create_refusal(
    category: RefusalCategory,
    reason: str,
    action: RemedyAction = "none",
    label: str = "",
    suggested_questions: list[str] | None = None,
) -> dict[str, Any]:
    """Builds a frozen RefusalWithRemedy structure for the API and trace."""
    refusal: dict[str, Any] = {
        "category": category,
        "reason": reason,
        "remedy": {
            "action": action,
            "label": label or action.replace("_", " ").capitalize(),
        },
    }
    if suggested_questions:
        refusal["remedy"]["suggested_questions"] = suggested_questions
    return refusal
