"""P8 — the report (master plan section 4.8).

``ReportBuilder.generate_html_report`` is the entry point the rest of the system
calls. It previously rendered the query, the answer, the task and a one-column
list of tool *names* — which omitted most of what section 4.8 asks a report to
contain, and specifically omitted **tool parameters and the parameter check**,
the fields the problem statement grades. It also interpolated the query straight
into the HTML with no escaping (a query containing ``<`` broke the document) and
referenced evidence by filesystem path, so a report opened anywhere other than
its build directory showed broken images.

Contents, from the plan: query, answer, confidence, evidence images, the full
trace table, tool parameters, warnings, timestamp.

Format: a single self-contained HTML file with the images inlined as data URIs,
which the browser prints to PDF. The Week-6 cut list already nominates
"PDF report -> styled HTML print/screenshot export" as an acceptable degradation,
so building the HTML first and treating PDF as a print of it means the cut costs
nothing and the offline venue demo has no extra dependency to install. If a PDF
engine is present, :func:`write_pdf` uses it; if not, the HTML is the deliverable
and says so rather than failing.

The trace table is not decoration. The problem statement grades the trace, so the
report renders the ``graded`` block first and in full, with the parameter check
visible — a judge should be able to answer "were only permitted parameters used"
from the printed page alone.
"""

import base64
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["ReportBuilder", "render_html", "write_html", "write_pdf"]

_STYLE = """
:root { color-scheme: light; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
       max-width: 62rem; padding: 2rem 1.5rem; color: #16202c; line-height: 1.5; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 .5rem; text-transform: uppercase;
     letter-spacing: .06em; color: #4a5b6e; border-bottom: 1px solid #dbe2ea;
     padding-bottom: .3rem; }
.meta { color: #6b7a8c; font-size: .85rem; }
.answer { font-size: 1.15rem; background: #f3f7fb; border-left: 4px solid #0072b2;
          padding: 1rem 1.15rem; border-radius: 0 6px 6px 0; margin: 1rem 0; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #e6ebf1;
         vertical-align: top; }
th { background: #f7f9fc; font-weight: 600; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; }
pre { background: #f7f9fc; padding: .8rem; border-radius: 6px; overflow-x: auto; }
.pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px;
        font-size: .75rem; font-weight: 600; }
.pass { background: #d8f3e3; color: #0b6b3a; }
.fail { background: #fbe0d8; color: #8c2d0c; }
.warn { background: #fdf3d8; color: #7a5a08; }
figure { margin: 0 0 1rem; }
figure img { max-width: 100%; border: 1px solid #dbe2ea; border-radius: 6px; }
figcaption { font-size: .8rem; color: #6b7a8c; margin-top: .3rem; }
@media print { body { max-width: none; } h2 { break-after: avoid; } }
"""


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix)
    if mime is None:
        return None
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{_escape(key)}</th><td>{value}</td></tr>" for key, value in rows
    )
    return f"<table>{body}</table>"


def render_html(trace: dict[str, Any], evidence_paths: list[Path] | None = None) -> str:
    """Render one trace as a self-contained report page."""
    graded = trace.get("graded", {})
    outputs = graded.get("outputs", {})
    check = graded.get("parameter_check", {})
    passed = bool(check.get("passed"))
    pill = (
        f'<span class="pill {"pass" if passed else "fail"}">'
        f'parameter check {"passed" if passed else "REJECTED"}</span>'
    )

    parts: list[str] = [
        "<h1>SatQuery AI — query report</h1>",
        f'<p class="meta">{_escape(trace.get("timestamp", ""))} · '
        f'query {_escape(trace.get("query_id", ""))} · '
        f'router path <code>{_escape(trace.get("router_path", "?"))}</code></p>',
        f"<h2>Question</h2><p>{_escape(trace.get('query_text', ''))}</p>",
    ]

    answer = outputs.get("answer")
    if answer:
        parts.append(f'<div class="answer">{_escape(answer)}</div>')
    refusal = outputs.get("refusal")
    if refusal:
        parts.append(
            f'<div class="answer"><span class="pill warn">refused '
            f'({_escape(refusal.get("category"))})</span><br>{_escape(refusal.get("reason"))}</div>'
        )

    summary = [
        ("Task selected", f"<code>{_escape(graded.get('task_selected'))}</code>"),
        (
            "Tools invoked",
            ", ".join(f"<code>{_escape(name)}</code>" for name in graded.get("tools_invoked", []))
            or "<em>none</em>",
        ),
        ("Confidence", _escape(outputs.get("confidence", "—"))),
        ("Confidence basis", _escape(outputs.get("confidence_basis", "heuristic"))),
        ("Parameter check", pill),
    ]
    if outputs.get("area_km2") is not None:
        summary.append(("Area", f"{_escape(outputs['area_km2'])} km²"))
    parts.append("<h2>Graded summary</h2>" + _table(summary))

    permitted = graded.get("permitted_parameters", [])
    if permitted:
        rows = "".join(
            "<tr>"
            f"<td><code>{_escape(entry.get('tool'))}</code></td>"
            f"<td><code>{_escape(json.dumps(entry.get('params', {}), sort_keys=True))}</code></td>"
            f'<td><span class="pill {"pass" if entry.get("within_manifest") else "fail"}">'
            f'{"within manifest" if entry.get("within_manifest") else "rejected"}</span></td>'
            "</tr>"
            for entry in permitted
        )
        parts.append(
            "<h2>Permitted parameters</h2>"
            "<table><tr><th>Tool</th><th>Parameters</th><th>Manifest</th></tr>"
            f"{rows}</table>"
        )
    if check.get("rejected"):
        parts.append(
            "<h2>Rejected parameters</h2><ul>"
            + "".join(f"<li>{_escape(item)}</li>" for item in check["rejected"])
            + "</ul>"
        )

    inputs = trace.get("inputs", [])
    if inputs:
        rows = "".join(
            "<tr>"
            f"<td>{_escape(entry.get('file'))}</td>"
            f"<td>{_escape(entry.get('modality'))}</td>"
            f"<td>{_escape(entry.get('crs', '—'))}</td>"
            f"<td>{_escape(', '.join(entry.get('bands', [])) or '—')}</td>"
            f"<td>{_escape(entry.get('native_gsd_m', '—'))} / "
            f"{_escape(entry.get('pixel_size_m', '—'))}</td>"
            "</tr>"
            for entry in inputs
        )
        parts.append(
            "<h2>Inputs</h2><table>"
            "<tr><th>File</th><th>Modality</th><th>CRS</th><th>Bands</th>"
            "<th>Native GSD / pixel size (m)</th></tr>"
            f"{rows}</table>"
        )

    steps = trace.get("steps", [])
    if steps:
        rows = "".join(
            "<tr>"
            f"<td><code>{_escape(step.get('tool'))}</code></td>"
            f"<td><code>{_escape(json.dumps(step.get('params', {}), sort_keys=True))}</code></td>"
            f"<td><code>"
            f"{_escape(json.dumps(step.get('outputs', {}), sort_keys=True)[:400])}"
            f"</code></td>"
            f"<td>{_escape(step.get('confidence', '—'))}</td>"
            f"<td>{_escape(step.get('latency_ms', '—'))}</td>"
            "</tr>"
            for step in steps
        )
        parts.append(
            "<h2>Execution trace</h2><table>"
            "<tr><th>Tool</th><th>Parameters</th><th>Outputs</th><th>Conf.</th><th>ms</th></tr>"
            f"{rows}</table>"
        )

    agreement = trace.get("agreement")
    if agreement:
        parts.append(
            "<h2>Cross-modal agreement</h2>"
            + _table(
                [
                    ("IoU", _escape(agreement.get("iou"))),
                    ("Verdict", _escape(agreement.get("verdict"))),
                    ("Disagreement cause", _escape(agreement.get("disagreement_cause") or "—")),
                ]
            )
        )

    figures = []
    for path in evidence_paths or []:
        uri = _data_uri(Path(path))
        if uri:
            figures.append(
                f'<figure><img src="{uri}" alt="{_escape(Path(path).name)}">'
                f"<figcaption>{_escape(Path(path).name)}</figcaption></figure>"
            )
    if figures:
        parts.append("<h2>Evidence</h2>" + "".join(figures))

    evidence = trace.get("evidence", [])
    if evidence:
        parts.append(
            "<h2>Evidence artifacts</h2><ul>"
            + "".join(f"<li><code>{_escape(item)}</code></li>" for item in evidence)
            + "</ul>"
        )

    warnings = trace.get("warnings", [])
    if warnings:
        parts.append(
            "<h2>Warnings</h2><ul>"
            + "".join(f"<li>{_escape(item)}</li>" for item in warnings)
            + "</ul>"
        )
    notes = trace.get("routing_notes", [])
    if notes:
        parts.append(
            "<h2>Routing notes</h2><ul>"
            + "".join(f"<li>{_escape(item)}</li>" for item in notes)
            + "</ul>"
        )

    parts.append(
        "<h2>Full trace</h2><pre>"
        + _escape(json.dumps(trace, indent=2, sort_keys=False))
        + "</pre>"
    )
    parts.append(
        f'<p class="meta">Generated {datetime.now(UTC).isoformat(timespec="seconds")} · '
        f"SatQuery AI · sources and licences in CREDITS.md</p>"
    )

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>SatQuery report — {_escape(trace.get('query_id', ''))}</title>"
        f"<style>{_STYLE}</style></head><body>" + "".join(parts) + "</body></html>"
    )


def write_html(
    path: Path | str, trace: dict[str, Any], evidence_paths: list[Path] | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(trace, evidence_paths), encoding="utf-8")
    return path


def write_pdf(
    path: Path | str, trace: dict[str, Any], evidence_paths: list[Path] | None = None
) -> Path:
    """Write a PDF if an engine is installed, else the HTML next to it.

    Returns whichever file was actually written. The caller records the real
    path in the trace, so a venue machine with no PDF engine still produces a
    complete deliverable rather than a missing one.
    """
    path = Path(path)
    html_path = path.with_suffix(".html")
    write_html(html_path, trace, evidence_paths)
    try:
        from weasyprint import HTML  # noqa: PLC0415 - optional, probed at call time
    except ImportError:
        return html_path
    path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_path.read_text(encoding="utf-8")).write_pdf(str(path))
    return path


class ReportBuilder:
    """P8 report generation, kept as the class the rest of the system calls."""

    @staticmethod
    def _generate_css() -> str:
        return f"<style>{_STYLE}</style>"

    @classmethod
    def generate_html_report(
        cls, trace: dict[str, Any], evidence_paths: list[str] | None = None
    ) -> str:
        """Render a trace as a complete, self-contained HTML report.

        Self-contained matters: the Week-6 cut list degrades the PDF to "styled
        HTML print/screenshot export", so the HTML has to survive being moved,
        emailed or printed on a venue machine with no access to the mask files.
        Evidence is inlined as data URIs for exactly that reason.
        """
        return render_html(trace, [Path(p) for p in (evidence_paths or [])])

    @classmethod
    def write_html_report(
        cls,
        path: Path | str,
        trace: dict[str, Any],
        evidence_paths: list[str] | None = None,
    ) -> Path:
        return write_html(path, trace, [Path(p) for p in (evidence_paths or [])])

    @classmethod
    def write_pdf_report(
        cls,
        path: Path | str,
        trace: dict[str, Any],
        evidence_paths: list[str] | None = None,
    ) -> Path:
        """Write a PDF if an engine is installed, else the HTML beside it.

        Returns whichever file was actually written, so the caller records the
        real path rather than a promised one.
        """
        return write_pdf(path, trace, [Path(p) for p in (evidence_paths or [])])
