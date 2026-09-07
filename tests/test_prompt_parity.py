"""The serving prompt must be byte-identical to the training prompt.

A prompt the model was never trained on is indistinguishable from a weak model:
it does not raise, it answers fluently, and every score moves with no cause a
metric can point at. ``satquery.training.dataset.scale_prefix`` is the single
implementation for that reason, and these pin every other path to it.

The specific hazard this guards: ``assemble_prompt`` used to build its own
format -- ``<image>{role}</image>`` placeholders and ``[sensor: X | GSD: Y m]``
tags -- which no adapter ever saw. It had no callers, but it carried the name a
caller would reach for while wiring the backend.
"""

from satquery.agent.prompt import assemble_prompt
from satquery.training.dataset import CanonicalSample, gsd_prompt_prefix, scale_prefix


def _row(**over):
    row = {
        "sample_id": "x_0001",
        "adapter": "change_vqa",
        "task": "change_or_not",
        "images": ["eval/second/im1/00001.png", "eval/second/im2/00001.png"],
        "image_roles": ["first", "second"],
        "modality": ["optical", "optical"],
        "effective_gsd_m": [0.5, 0.5],
        "question": "Have the areas of trees changed?",
        "answer": "no",
        "answer_type": "yesno",
        "split": "train",
        "source": "CDVQA (Yuan et al., TGRS 2022)",
    }
    row.update(over)
    return row


def test_assemble_prompt_matches_what_training_builds():
    row = _row()
    sample = CanonicalSample.from_row(row)

    trained = f"{gsd_prompt_prefix(sample)}{sample.question}"
    served = assemble_prompt(
        row["images"],
        row["image_roles"],
        row["modality"],
        row["effective_gsd_m"],
        row["question"],
        source=row["source"],
    )
    assert served == trained


def test_no_placeholder_tokens_leak_into_the_prompt():
    """The model takes images as processor content, not as text placeholders."""
    row = _row()
    served = assemble_prompt(
        row["images"],
        row["image_roles"],
        row["modality"],
        row["effective_gsd_m"],
        row["question"],
        source=row["source"],
    )
    for leaked in ("<image>", "</image>", "[sensor:", "{first}", "{second}"):
        assert leaked not in served, f"{leaked!r} is not a token training ever used"


def test_missing_gsd_yields_no_prefix_rather_than_an_invented_one():
    """A wrong scale is worse than none: the model learns to trust the field."""
    served = assemble_prompt(
        ["a.png"], ["single"], ["optical"], [], "What is here?", source=""
    )
    assert served == "What is here?"


def test_scale_prefix_stays_the_single_implementation():
    row = _row(effective_gsd_m=[0.5, 10.0])
    direct = scale_prefix(row["effective_gsd_m"], row["question"], row["source"], None)
    served = assemble_prompt(
        row["images"],
        row["image_roles"],
        row["modality"],
        row["effective_gsd_m"],
        row["question"],
        source=row["source"],
    )
    assert served.startswith(direct)
    assert "ground sample distance per view" in direct
