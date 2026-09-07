"""A refusal must fire on questions that genuinely need a missing band.

The refusal path is a graded deliverable -- "that refusal is a feature, not an
error" -- which cuts both ways: refusing a question the scene *can* answer is as
wrong as fabricating one it cannot.

`SPECTRAL_TERMS` used to contain bare "green", "colour" and "color", so
"is there a green car in the image" on an RGB JPEG refused with *"this sensor
carries no near-infrared or shortwave-infrared band"* -- for a question that
needs no infrared at all. A colour question is answered from the visible bands,
which every RGB source has.
"""

from __future__ import annotations

import pytest

from satquery.agent.router import QueryContext, route
from satquery.agent.validator import validate
from satquery.ingest.band_inventory import BandInventory


def _rgb(question: str):
    """An RGB-only source: `computable_indices` is empty, so no index exists.

    Bands are a name->index mapping and the flags are explicit, matching what
    ingest produces for a plain JPEG.
    """
    context = QueryContext(
        query_text=question,
        modalities=["optical"],
        inventories=[
            BandInventory(
                bands={"red": 1, "green": 2, "blue": 3},
                has_nir=False,
                has_swir=False,
                computable_indices=[],
            )
        ],
        image_count=1,
        dates=["2020-01-01"],
    )
    return context, route(context)


ANSWERABLE = [
    "is there a green car in the image",
    "what colour is the large roof",
    "are the fields green",
    "find the red vehicle",
]

REFUSABLE = [
    "is the vegetation healthy",
    "what is the NDVI of this field",
    "how much soil moisture is there",
    "what is the chlorophyll content",
]


@pytest.mark.parametrize("question", ANSWERABLE)
def test_colour_questions_are_not_refused_on_an_rgb_scene(question):
    """Visible-band questions are what an RGB scene is best equipped for."""
    result = validate(*_rgb(question))
    assert result.passed, (
        f"{question!r} was refused: {getattr(result.refusal, 'reason', '')[:120]}"
    )


@pytest.mark.parametrize("question", REFUSABLE)
def test_radiometric_questions_are_still_refused(question):
    """Health, chlorophyll, moisture and the normalised indices are radiometric
    quantities that need a band this source does not carry. Answering them from
    texture would be a confident fabrication, which scores worse than a refusal."""
    result = validate(*_rgb(question))
    assert not result.passed, f"{question!r} should have been refused"
    assert result.refusal.category == "modality_limitation"


def test_the_refusal_explains_the_fix():
    """A refusal that does not say what would work is a dead end."""
    result = validate(*_rgb("is the vegetation healthy"))
    reason = result.refusal.reason.lower()
    assert "near-infrared" in reason or "shortwave" in reason
    assert "multispectral" in reason or "sar" in reason
