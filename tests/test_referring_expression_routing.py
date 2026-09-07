"""VRSBench referring rows must reach the grounding path.

62.7% acc@0.5 was measured by handing referring text straight to the grounding
prompt. The router never saw those rows, and when the held-out gallery finally
put them through it, 16 of 22 failures were the same thing: a referring
expression is a *statement*, so it matched no grounding term and fell through to
the VQA adapter, which answered "yes." and produced no box at all.

The risk in fixing that is over-firing -- pulling presence questions or captions
into grounding -- so the guards below matter as much as the positives.
"""

import pytest

from satquery.agent.router import QueryContext, Task, route
from satquery.ingest.band_inventory import BandInventory


def _route(text: str, image_count: int = 1):
    inventory = BandInventory(bands={"blue": 1, "green": 2, "red": 3})
    return route(
        QueryContext(
            query_text=text,
            modalities=["optical"],
            inventories=[inventory],
            image_count=image_count,
        )
    )


REFERRING = [
    # Verbatim VRSBench EVAL rows, each a routing miss before this landed.
    "The relatively larger airplane positioned near the right edge of the image "
    "can be found in the bottom-right corner.",
    "A roundabout is visible in the middle-left section of the image.",
    "The left smallest storage-tank positioned in the top-left section of the image.",
    "The baseball diamond situated at the bottom right of the cross-like formation.",
    "The soccer ball field can be identified by its white markings and is "
    "situated adjacent to a road.",
    "The overpass is the large diagonal structure crossing the image.",
    # "between the two" is spatial here, not bi-temporal; this used to be
    # refused as a change query asked of a single image.
    "The ship is centered in the image, positioned on the water between the two small harbors.",
]


@pytest.mark.parametrize("text", REFERRING)
def test_referring_expressions_route_to_grounding(text):
    assert _route(text).task is Task.SINGLE_GROUNDING


@pytest.mark.parametrize(
    "text,expected",
    [
        # Presence, not location: the object is named and placed, but it is a
        # question, and the answer is yes/no rather than a box.
        ("Is there a ship on the left side of the image?", Task.SINGLE_VQA),
        ("Is a water area present?", Task.SINGLE_VQA),
        ("How many buildings are there?", Task.SINGLE_VQA),
        ("Describe this image.", Task.SINGLE_CAPTION),
        ("Where is the runway?", Task.SINGLE_GROUNDING),
    ],
)
def test_the_detector_does_not_over_fire(text, expected):
    assert _route(text).task is expected


def test_a_real_change_question_still_routes_to_change():
    assert _route("What changed between the two images?", image_count=2).task is Task.CHANGE_VQA


def test_the_whole_sentence_reaches_the_grounding_prompt():
    """Parity with the measured arm.

    ``eval_grounding_pipelines.py`` does ``phrase = row["question"].rstrip(". ")``
    -- the entire referring sentence. Reducing it to the extracted noun drops
    the half that disambiguates one airplane from the others in frame.
    """
    text = REFERRING[0]
    decision = _route(text)
    phrase = next(
        step["params"].get("phrase")
        for step in decision.plan
        if step["tool"] == "rs_ground_caption"
    )
    assert phrase == text.rstrip(". ")


def test_an_interrogative_still_passes_the_extracted_noun():
    """"Find this object in the satellite image: Where is the runway?" is not a
    phrase, and is not what 62.7% was measured on either."""
    decision = _route("Where is the runway?")
    phrase = next(
        step["params"].get("phrase")
        for step in decision.plan
        if step["tool"] == "rs_ground_caption"
    )
    assert phrase == "runway"
