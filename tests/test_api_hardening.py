"""The API is public and unauthenticated, so its limits have to be real.

Three findings from the pre-deployment audit, each of which was invisible to
the 440 tests that already passed because none of them asked what happens on
input nobody intended to send.

The endpoint being open is a decision, not an oversight: a demo anyone can
open is the point. What that decision requires is that everything *around* the
open door is bounded -- upload size, path scope, and an opt-in key for the day
the door should close.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
rasterio = pytest.importorskip("rasterio")

from fastapi.testclient import TestClient  # noqa: E402

from satquery.api import server as srv  # noqa: E402


def _client(monkeypatch, **overrides) -> TestClient:
    for name, value in overrides.items():
        monkeypatch.setattr(srv, name, value)
    return TestClient(srv.build_app(upload_dir=Path(tempfile.mkdtemp())))


# -- uploads ---------------------------------------------------------------
def test_an_oversized_upload_is_refused_rather_than_read_into_memory(monkeypatch):
    """``await file.read()`` with no argument buffered the whole body first.

    The endpoint takes no credentials, so a single POST decided how much of the
    container's memory a stranger could allocate. The frontend advertises a
    4 GB ceiling, but a limit that exists only in the client is not a limit.
    """
    client = _client(monkeypatch, MAX_UPLOAD_BYTES=1024)
    response = client.post(
        "/api/v1/scenes", files={"file": ("big.tif", io.BytesIO(b"x" * 5000))}
    )
    assert response.status_code == 413


def test_a_refused_upload_leaves_nothing_behind(monkeypatch):
    """Streaming writes as it reads, so the partial file has to be cleaned up.

    Otherwise refusing the request still spends the disk it was refused for.
    """
    upload_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 1024)
    client = TestClient(srv.build_app(upload_dir=upload_dir))
    client.post("/api/v1/scenes", files={"file": ("big.tif", io.BytesIO(b"x" * 5000))})
    assert list(upload_dir.glob("up_*")) == []


def test_the_default_ceiling_matches_what_the_frontend_advertises():
    """4 GB on both sides, so the client never offers what the server refuses."""
    assert srv.MAX_UPLOAD_BYTES == 4 * 1024**3


def test_an_upload_that_is_not_an_image_is_refused(monkeypatch):
    """The frontend filters by type, but only the server's check holds."""
    upload_dir = Path(tempfile.mkdtemp())
    client = TestClient(srv.build_app(upload_dir=upload_dir))
    response = client.post(
        "/api/v1/scenes",
        files={"file": ("notes.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert response.status_code == 415
    assert list(upload_dir.glob("up_*")) == []


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("scene.TIF", "application/octet-stream"),
        ("scene.jp2", ""),
        ("scene.webp", "image/webp"),
        ("scene", "image/png"),
    ],
)
def test_image_uploads_are_recognised_by_extension_or_type(filename, content_type):
    assert srv._is_image_upload(filename, content_type)


def test_svg_is_not_a_raster_upload():
    assert not srv._is_image_upload("drawing.svg", "image/svg+xml")


# -- optional key ----------------------------------------------------------
def test_the_api_is_open_when_no_key_is_configured(monkeypatch):
    """The default has to stay open, or the demo breaks the day this ships."""
    client = _client(monkeypatch, API_KEY=None)
    assert client.get("/api/v1/meta/tasks").status_code == 200


def test_a_configured_key_is_required_on_everything_but_health(monkeypatch):
    client = _client(monkeypatch, API_KEY="s3cret")
    assert client.get("/api/v1/meta/tasks").status_code == 401
    assert client.get("/api/v1/meta/tasks", headers={"X-Api-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/meta/tasks", headers={"X-Api-Key": "s3cret"}).status_code == 200


def test_health_stays_reachable_without_the_key(monkeypatch):
    """Modal probes it. A gated health check reads as a dead container."""
    client = _client(monkeypatch, API_KEY="s3cret")
    assert client.get("/health").status_code == 200


# -- path containment ------------------------------------------------------
@pytest.mark.parametrize(
    "probe",
    [
        "../data_backup/leak.png",
        "../database/leak.png",
        "../../etc/passwd",
    ],
)
def test_a_sibling_sharing_the_root_prefix_is_not_inside_the_root(probe):
    """``str(target).startswith(str(root))`` was the wrong question.

    With a root of ``/data`` it also admits ``/data_backup`` and ``/database``:
    a sibling whose name merely extends the root shares the prefix while living
    entirely outside it. ``is_relative_to`` compares path components, which is
    what "inside the root" actually means. This mirrors the guard in
    ``scripts/corpus_server.py``; the equivalent in ``gallery_preview`` was
    already correct and is what it was rewritten to match.
    """
    root = Path("/data").resolve()
    assert not (root / probe).resolve().is_relative_to(root)


def test_an_ordinary_path_is_still_served():
    root = Path("/data").resolve()
    assert (root / "images/a.png").resolve().is_relative_to(root)


# -- SNAP graph ------------------------------------------------------------
def test_a_filename_cannot_rewrite_the_snap_processing_graph():
    """Paths land inside XML element text, and can come from an upload."""
    from satquery.sar.normalisation import xml_escape

    escaped = xml_escape("/data/x</file><exec>whatever</exec><file>.tif")
    assert "<exec>" not in escaped
    assert "&lt;exec&gt;" in escaped
