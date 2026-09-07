"""P8 report generation (master plan section 4.8).

The plan lists what a report contains: query, answer, confidence, evidence
images, the **full trace table, tool parameters**, warnings and a timestamp. The
earlier test asserted only that the query, the answer and a list of tool *names*
appeared — all of which a report can contain while omitting every graded field,
which is what it did.

Two properties beyond content are load-bearing. The report has to be
self-contained, because the Week-6 cut list degrades the PDF to "styled HTML
print/screenshot export" and a venue machine will open it away from the mask
files. And it has to escape its inputs: the query is user text, and a query
containing `<` broke the document.
"""

from pathlib import Path

import pytest

from satquery.report.report_builder import ReportBuilder


@pytest.fixture
def trace() -> dict:
    return {
        "schema_version": 2,
        "query_id": "q-1",
        "query_text": "Find water bodies",
        "timestamp": "2024-01-01T12:00:00Z",
        "router_path": "rules",
        "graded": {
            "task_selected": "single_vqa",
            "tools_invoked": ["spectral_index", "sar_backscatter"],
            "permitted_parameters": [
                {
                    "tool": "spectral_index",
                    "params": {"index": "NDWI", "threshold_method": "otsu",
                               "threshold_value": 0.14},
                    "within_manifest": True,
                }
            ],
            "parameter_check": {"passed": True, "rejected": []},
            "outputs": {
                "answer": "Yes, water is present.",
                "confidence": 0.81,
                "area_km2": 3.4,
            },
        },
        "steps": [
            {
                "tool": "spectral_index",
                "params": {"index": "NDWI", "threshold_value": 0.14},
                "outputs": {"area_km2": 3.4},
                "confidence": 0.79,
                "latency_ms": 180,
            }
        ],
        "agreement": {"iou": 0.86, "verdict": "consistent", "disagreement_cause": None},
        "warnings": ["NDBI unavailable: source lacks SWIR band"],
    }


def test_html_generation_does_not_crash(trace):
    html = ReportBuilder.generate_html_report(trace, ["dummy_overlay.png"])
    assert "SatQuery" in html
    assert "Find water bodies" in html
    assert "spectral_index" in html
    assert "Yes, water is present." in html
    assert "<html" in html


def test_report_contains_every_field_section_4_8_names(trace):
    html = ReportBuilder.generate_html_report(trace)
    for required in (
        "0.81",  # confidence
        "threshold_value",  # tool parameters - a graded field
        "0.14",
        "parameter check",  # the gate result, readable by a judge
        "180",  # per-step latency
        "NDBI unavailable",  # warnings
        "2024-01-01",  # timestamp
        "consistent",  # cross-modal agreement
    ):
        assert required in html, f"the report omits {required!r}, which section 4.8 requires"


def test_report_escapes_user_text(trace):
    """The query is user input and lands in the HTML."""
    trace["query_text"] = 'Is there <script>alert("x")</script> water?'
    html = ReportBuilder.generate_html_report(trace)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_report_is_self_contained(trace, tmp_path):
    """Evidence inlines as data URIs; a moved report must still render."""
    from PIL import Image

    image = tmp_path / "overlay.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(image)

    html = ReportBuilder.generate_html_report(trace, [str(image)])
    assert "data:image/png;base64," in html
    assert str(image) not in html, "a filesystem path will not resolve once the report moves"


def test_write_html_report_round_trips(trace, tmp_path):
    written = ReportBuilder.write_html_report(tmp_path / "report.html", trace)
    assert written.exists()
    assert "Find water bodies" in written.read_text(encoding="utf-8")


def test_write_pdf_report_falls_back_to_html_without_an_engine(trace, tmp_path):
    """A venue machine with no PDF engine still gets a complete deliverable."""
    written = ReportBuilder.write_pdf_report(tmp_path / "report.pdf", trace)
    assert written.exists()
    assert written.suffix in (".pdf", ".html")
    assert Path(tmp_path / "report.html").exists()
