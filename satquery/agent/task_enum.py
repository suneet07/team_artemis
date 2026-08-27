from enum import StrEnum


class Task(StrEnum):
    SINGLE_VQA = "single_vqa"
    SINGLE_CAPTION = "single_caption"
    SINGLE_GROUNDING = "single_grounding"
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_VQA = "change_vqa"
    CHANGE_MAP = "change_map"
    CROSSMODAL_EXTRACTION = "crossmodal_extraction"
    CROSSMODAL_VQA = "crossmodal_vqa"


class RouterPath(StrEnum):
    RULES = "rules"
    LLM = "llm"
