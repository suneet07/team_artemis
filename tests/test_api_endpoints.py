from fastapi.testclient import TestClient

from satquery.api.app import app

client = TestClient(app)


def test_api_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "serving" in data
    assert "adapters_loaded" in data


def test_api_meta_tools():
    resp = client.get("/meta/tools")
    assert resp.status_code == 200
    tools = resp.json()
    assert isinstance(tools, list)
    names = [t["name"] for t in tools]
    assert "spectral_index" in names
    assert "sar_backscatter" in names
    assert len(names) == 15


def test_api_meta_disagreement_causes():
    resp = client.get("/meta/disagreement-causes")
    assert resp.status_code == 200
    data = resp.json()
    assert "causes" in data
    codes = [c["code"] for c in data["causes"]]
    assert "cloud_over_water" in codes
    assert "wet_smooth_soil" in codes


def test_api_probe_bundle():
    resp = client.get("/bundles/test_probe_bundle/probe")
    assert resp.status_code == 200
    data = resp.json()
    assert "supported_tasks" in data
    assert "blocked_tasks" in data
    assert "single_vqa" in data["supported_tasks"]


def test_api_submit_query_json_success():
    payload = {"question": "What is the vegetation extent?"}
    resp = client.post("/bundles/b_test/queries", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "succeeded"
    assert data["query_id"] is not None
    assert "trace" in data

    # Test GET /queries/{query_id}
    q_id = data["query_id"]
    get_resp = client.get(f"/queries/{q_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["query_id"] == q_id


def test_api_submit_query_sse_streaming():
    payload = {"question": "What is the vegetation extent?"}
    headers = {"Accept": "text/event-stream"}
    resp = client.post("/bundles/b_test/queries", json=payload, headers=headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "event: accepted" in text
    assert "event: router" in text
    assert "event: validator" in text
    assert "event: plan" in text
    assert "event: fusion" in text
    assert "event: done" in text
    # Verify id: field exists on SSE events
    assert "id: " in text


def test_api_post_queries_frozen_endpoint():
    payload = {"bundle_id": "b_frozen_test", "question": "What is the vegetation extent?"}
    resp = client.post("/queries", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "succeeded"
    assert data["bundle_id"] == "b_frozen_test"

    # Test GET /queries listing
    list_resp = client.get("/queries?bundle_id=b_frozen_test")
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert len(items) >= 1
    assert items[0]["bundle_id"] == "b_frozen_test"


def test_api_cancel_reports_the_real_state_of_a_finished_query():
    """§24: cancellation is cooperative. A finished query cannot be un-finished.

    This used to rewrite a completed result's state to "cancelled", which made
    the endpoint look functional while doing nothing to a running query.
    """
    resp = client.post("/bundles/b_test/queries", json={"question": "What is here?"})
    q_id = resp.json()["query_id"]
    actual_state = client.get(f"/queries/{q_id}").json()["state"]

    cancel_resp = client.post(f"/queries/{q_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == actual_state

    # The stored result is untouched -- no retroactive relabelling.
    assert client.get(f"/queries/{q_id}").json()["state"] == actual_state


def test_api_cancel_unknown_query_is_404():
    assert client.post("/queries/does-not-exist/cancel").status_code == 404


def test_api_get_query_trace():
    payload = {"question": "What is the vegetation extent?"}
    resp = client.post("/bundles/b_test/queries", json=payload)
    assert resp.status_code == 200
    q_id = resp.json()["query_id"]

    trace_resp = client.get(f"/queries/{q_id}/trace")
    assert trace_resp.status_code == 200
    data = trace_resp.json()
    assert data["schema_version"] == 2
    assert "graded" in data
    assert data["query_id"] == q_id


