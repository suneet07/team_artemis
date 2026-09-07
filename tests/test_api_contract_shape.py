"""Every payload the console reads carries the fields its contract declares.

Five separate crashes in one session came from the same cause: the API omitted
a field that ``frontend/contracts/types.ts`` types as required, and the console
has no error boundary, so one missing key blanked the whole workspace. Each was
found by clicking, which is the expensive way.

The contract file is the shared artifact between the two halves of this project,
so it is the thing to test against. Required means "no ``?`` after the name" --
a required-but-nullable field must still be present and explicitly ``null``,
because ``undefined`` is what ``resolveUrl`` and ``.length`` choke on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = PROJECT_ROOT / "frontend" / "contracts" / "types.ts"

pytestmark = pytest.mark.skipif(
    not CONTRACTS.exists(), reason="frontend contracts not present in this checkout"
)


def required_fields(interface: str) -> list[str]:
    """Field names an interface declares without ``?``."""
    source = CONTRACTS.read_text(encoding="utf-8")
    block = re.search(
        r"export interface " + interface + r" \{(.*?)\n\}", source, re.S
    )
    if block is None:
        raise AssertionError(f"no interface {interface!r} in {CONTRACTS.name}")
    names: list[str] = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "/*", "*")):
            continue
        field = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\??):", line)
        if field and field.group(2) != "?":
            names.append(field.group(1))
    return names


@pytest.fixture(scope="module")
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    rasterio = pytest.importorskip("rasterio")
    import tempfile

    import numpy as np
    from rasterio.transform import from_origin

    from satquery.api.server import build_app

    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.tif"
    # A real georeferenced raster, not a stub: bounds_wgs84 is reprojected from
    # the file's own CRS, so a scene without one would exercise the null path
    # and prove nothing about the populated one.
    with rasterio.open(
        scene,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint16",
        crs="EPSG:32633",
        transform=from_origin(400000, 5330000, 10, 10),
    ) as dst:
        for band in range(1, 4):
            dst.write(
                np.random.randint(0, 4000, (64, 64)).astype("uint16"), band
            )

    app = build_app(upload_dir=str(tmp / "uploads"))
    with fastapi_testclient.TestClient(app) as http:
        yield http, scene


@pytest.fixture(scope="module")
def prepared(client):
    """One scene uploaded, bundled and queried -- the console's whole path."""
    http, scene = client
    with scene.open("rb") as handle:
        upload = http.post("/api/v1/scenes", files={"file": ("scene.tif", handle)})
    assert upload.status_code == 200, upload.text
    scene_id = upload.json()["scene_id"]

    bundle = http.post(
        "/api/v1/bundles", json={"scene_ids": [scene_id], "label": "contract"}
    )
    assert bundle.status_code == 200, bundle.text
    bundle_id = bundle.json()["bundle_id"]

    # Addressed by bundle, the way the console does it -- not by scene_ids.
    query = http.post(
        "/api/v1/queries",
        json={"bundle_id": bundle_id, "question": "What is the vegetation cover?"},
    )
    assert query.status_code == 200, query.text
    return http, bundle_id, query.json()


def _assert_shape(interface: str, payload: dict) -> None:
    missing = [f for f in required_fields(interface) if f not in payload]
    assert not missing, (
        f"{interface} is missing required field(s) {missing}. The console reads "
        "these without a guard, so an absent key blanks the screen rather than "
        "degrading. Emit an explicit null when there is no value."
    )


def test_bundle_payload_matches_the_contract(prepared):
    http, bundle_id, _ = prepared
    _assert_shape("Bundle", http.get(f"/api/v1/bundles/{bundle_id}").json())


def test_every_scene_in_a_bundle_matches_the_contract(prepared):
    http, bundle_id, _ = prepared
    for scene in http.get(f"/api/v1/bundles/{bundle_id}").json()["scenes"]:
        _assert_shape("Scene", scene)


def test_query_result_matches_the_contract(prepared):
    _, _, query = prepared
    _assert_shape("QueryResult", query)


def test_a_query_carries_a_stream_url(prepared):
    """The console opens its EventSource only when this is present."""
    _, _, query = prepared
    assert query.get("stream_url"), (
        "without stream_url the chat panel sits on 'validating inputs' forever, "
        "however fast the query itself finished"
    )


def test_every_evidence_asset_matches_the_contract(prepared):
    _, _, query = prepared
    assets = query.get("evidence") or []
    assert assets, "the deterministic path should produce evidence for this scene"
    for asset in assets:
        _assert_shape("AssetRef", asset)


def test_the_evidence_event_carries_whole_assets_not_ids(prepared):
    """``addLayers`` reads ``download_url`` off each entry."""
    http, _, query = prepared
    events = http.get(f"/api/v1/queries/{query['query_id']}/events").text
    assert "event: evidence" in events
    for asset in query.get("evidence") or []:
        assert asset["asset_id"] in events
        assert asset["download_url"] in events, (
            "the evidence frame carried ids rather than asset records, which is "
            "what threw 'undefined (reading startsWith)' inside resolveUrl"
        )


def test_the_answer_reaches_the_console_as_a_fusion_frame(prepared):
    """A turn renders its text from the stream, not from the POST body."""
    http, _, query = prepared
    events = http.get(f"/api/v1/queries/{query['query_id']}/events").text
    assert "event: fusion" in events, (
        "without a fusion frame the turn shows its latency and trace link above "
        "an empty answer"
    )


def test_a_bundle_that_does_not_exist_is_a_404_not_a_500(prepared):
    http, _, _ = prepared
    response = http.post(
        "/api/v1/queries", json={"bundle_id": "bn_nope", "question": "anything?"}
    )
    assert response.status_code == 404


def test_meta_tools_says_which_tools_can_actually_run(prepared):
    """A manifest is a promise; ``available`` is whether it can be kept."""
    http, _, _ = prepared
    tools = http.get("/api/v1/meta/tools").json()["tools"]
    assert tools
    for tool in tools:
        assert "available" in tool, f"{tool['name']} does not say if it can run"
    names = {t["name"]: t["available"] for t in tools}
    assert names.get("spectral_index") is True
    # No adapters are registered by build_app here, so the learned tools must
    # report absent rather than inheriting their manifest's existence.
    assert names.get("change_vqa") is False
