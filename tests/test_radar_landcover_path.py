"""A land-cover question about a radar scene must reach the classifier.

Three separate faults sat between BIFOLD's measured 74.95% and the answer a user
got, and the held-out gallery hit all three at once -- 23/50, below the 50.1%
majority floor:

1. Sentinel-1 GeoTIFFs name no bands, so the inventory came back empty and the
   tool refused with "needs band 'VH'". Band 1 is VH, band 2 is VV, and that
   order is measured (0.817 against 0.499 reversed), not assumed.
2. A tool that raises is dropped from `execution.steps` and survives only as a
   warning, so the refusal was invisible in the trace.
3. Even planned and running, the classifier only produced `labels`. The VLM
   answered instead -- the component scoring 0.4968 on this exact task
   speaking over the one scoring 0.7525.
"""


from satquery.agent.router import QueryContext, route
from satquery.ingest.band_inventory import BandInventory
from satquery.tools.lulc import CLASSES, answer_for, names_a_known_class


def _radar_context(question: str) -> QueryContext:
    inventory = BandInventory(
        bands={"vh": 1, "vv": 2}, polarisations=["vv", "vh"], sar_band="C"
    )
    return QueryContext(
        query_text=question, modalities=["sar"], inventories=[inventory], image_count=1
    )


def test_undescribed_sentinel1_still_yields_vh_and_vv():
    """The inference itself, at the config level rather than through a file."""
    from satquery.config import preprocessing_config

    orders = preprocessing_config().bands.assumed_sar_orders
    assert orders[2] == ("vh", "vv"), (
        "VH must come first: BIFOLD scores 0.817 that way round and 0.499 reversed"
    )


def test_radar_landcover_question_reaches_the_classifier():
    plan = [step["tool"] for step in route(_radar_context("Is there forest in this image?")).plan]
    assert "lulc_classifier" in plan


def test_the_vlm_does_not_answer_over_the_classifier_on_radar():
    plan = [step["tool"] for step in route(_radar_context("Is there forest in this image?")).plan]
    assert "rs_vqa" not in plan


def test_an_open_radar_question_still_reaches_the_vlm():
    """The scoping. Only questions the classifier can actually answer are taken
    from the VLM; it remains the only component that can answer an open one."""
    plan = [step["tool"] for step in route(_radar_context("What is happening here?")).plan]
    assert "rs_vqa" in plan


def test_the_classifier_answers_with_the_benchmark_rule():
    probs = [0.0] * len(CLASSES)
    probs[CLASSES.index("Coniferous forest")] = 0.9
    assert answer_for("Is there forest in this image?", probs) == "yes"
    probs[CLASSES.index("Coniferous forest")] = 0.1
    assert answer_for("Is there forest in this image?", probs) == "no"


def test_an_unknown_class_abstains_rather_than_guessing():
    assert answer_for("Is there a giraffe here?", [0.9] * len(CLASSES)) is None
    assert not names_a_known_class("Is there a giraffe here?")


def test_eval_and_serving_share_one_rule():
    """Two copies of a scoring rule is how a benchmark number and a product
    answer stop meaning the same thing."""
    source = __import__("pathlib").Path("scripts/eval_bifold.py").read_text(encoding="utf-8")
    assert "from satquery.tools.lulc import answer_for" in source
    assert "def answer_for" not in source
