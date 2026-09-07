"""Headless evaluation mode (master plan section 4.11).

    python -m satquery.evalcli \\
        --images scene_opt.tif scene_sar.tif \\
        --question "..." \\
        --out answer.json trace.json

    python -m satquery.evalcli --batch queries.jsonl --out-dir results/

This mode is simultaneously three things, which is why it is not optional:

* the artifact we submit if the organisers run our code;
* the venue-demo parachute, for a room with no internet and no cloud GPU;
* the CI smoke test.

Constraints it holds to, from the plan: single container, single command, no
frontend, no Redis, no Celery, no network. Deterministic tools run CPU-only when
no GPU is present. Whatever the submission spec turns out to say, this satisfies
the most restrictive plausible version of it — which is the assumption the plan
tells us to hold until the spec arrives.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from satquery.agent.pipeline import answer_query
from satquery.evalcli.formatter import BENCHMARK_FORMATS, format_answer

__all__ = ["main", "run_batch", "run_single"]


def _answer_payload(outcome, benchmark: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query_id": outcome.trace["query_id"],
        "question": outcome.trace["query_text"],
        "answer": outcome.answer,
        "confidence": round(outcome.confidence, 4),
        "task": outcome.trace["graded"]["task_selected"],
        "refused": outcome.refused,
    }
    graded_outputs = outcome.trace["graded"]["outputs"]
    if graded_outputs.get("masks"):
        payload["masks"] = graded_outputs["masks"]
    if graded_outputs.get("area_km2") is not None:
        payload["area_km2"] = graded_outputs["area_km2"]
    if benchmark:
        formatted = format_answer(outcome.answer, benchmark)
        payload["formatted_answer"] = formatted.text
        payload["formatter_matched"] = formatted.matched
        payload["formatter_basis"] = formatted.basis
    return payload


def run_single(
    question: str,
    images: list[str],
    *,
    modalities: list[str] | None = None,
    dates: list[str] | None = None,
    output_dir: Path | None = None,
    benchmark: str | None = None,
    write_evidence: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Answer one query. Returns ``(answer_payload, trace)``."""
    outcome = answer_query(
        question,
        images,
        modalities=modalities,
        dates=dates,
        output_dir=output_dir,
        write_evidence=write_evidence and output_dir is not None,
    )
    return _answer_payload(outcome, benchmark), outcome.trace


def run_batch(batch_path: Path, out_dir: Path, *, write_evidence: bool = False) -> int:
    """Consume a JSONL of ``{images, question, ...}`` rows.

    This is the shape any benchmark harness or hidden-set evaluation will take.
    One malformed or unreadable row is recorded as a failed row and the run
    continues — a batch of 5,000 must not be lost to one bad path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    traces_path = out_dir / "traces.jsonl"
    failures = 0

    with (
        batch_path.open(encoding="utf-8") as source,
        answers_path.open("w", encoding="utf-8") as answers_file,
        traces_path.open("w", encoding="utf-8") as traces_file,
    ):
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                payload, trace = run_single(
                    row["question"],
                    row["images"],
                    modalities=row.get("modalities"),
                    dates=row.get("dates"),
                    output_dir=out_dir / f"row_{line_number:06d}" if write_evidence else None,
                    benchmark=row.get("benchmark"),
                    write_evidence=write_evidence,
                )
                if "id" in row:
                    payload["id"] = row["id"]
                    trace = {**trace}
            except Exception as error:  # noqa: BLE001 - one row must not kill the batch
                failures += 1
                payload = {
                    "id": None,
                    "error": f"{type(error).__name__}: {error}",
                    "line": line_number,
                }
                trace = {"error": payload["error"], "line": line_number}
            answers_file.write(json.dumps(payload) + "\n")
            traces_file.write(json.dumps(trace) + "\n")

    print(f"wrote {answers_path} and {traces_path}", file=sys.stderr)
    if failures:
        print(f"{failures} row(s) failed; see the error field in answers.jsonl", file=sys.stderr)
    return failures


def _selftest() -> int:
    """Answer one query on a synthetic scene, with no data and no network.

    Section 4.11 requires this mode to work inside a single container with no
    frontend, no Redis, no Celery and no network, on CPU. The cheapest way to
    keep that true is to prove it on every commit.
    """
    import tempfile

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from satquery.agent.trace import validate_trace

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selftest.tif"
        rng = np.random.default_rng(0)
        data = np.clip(
            np.linspace(400, 3600, 4 * 64 * 64).reshape(4, 64, 64)
            + rng.integers(-150, 150, (4, 64, 64)),
            0,
            4095,
        ).astype("uint16")
        profile = {
            "driver": "GTiff",
            "height": 64,
            "width": 64,
            "count": 4,
            "dtype": "uint16",
            "crs": "EPSG:32644",
            "transform": from_origin(0.0, 640.0, 10.0, 10.0),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
            for index, name in enumerate(("Blue", "Green", "Red", "NIR"), start=1):
                dst.set_band_description(index, name)

        payload, trace = run_single(
            "Where is the water body?", [str(path)], write_evidence=False
        )
        validate_trace(trace)
        graded = trace["graded"]
        if not graded["parameter_check"]["passed"]:
            print(
                "selftest FAILED: the parameter gate rejected its own plan: "
                f"{graded['parameter_check']['rejected']}",
                file=sys.stderr,
            )
            return 1
        if not graded["tools_invoked"]:
            print("selftest FAILED: no tool ran on a valid optical scene", file=sys.stderr)
            return 1
    print(
        f"selftest OK - task={graded['task_selected']} "
        f"tools={','.join(graded['tools_invoked'])} "
        f"confidence={payload['confidence']}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m satquery.evalcli",
        description="SatQuery headless evaluation mode (master plan section 4.11).",
    )
    parser.add_argument("--question", help="the question to answer")
    parser.add_argument("--images", nargs="+", help="one or two input rasters")
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["optical", "sar"],
        help="declare each input's modality instead of detecting it",
    )
    parser.add_argument("--dates", nargs="+", help="acquisition date per input, for change queries")
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_FORMATS),
        help="normalise the answer onto this benchmark's closed vocabulary (section 6.4)",
    )
    parser.add_argument(
        "--out",
        nargs=2,
        metavar=("ANSWER_JSON", "TRACE_JSON"),
        help="write the answer and the trace to these two files",
    )
    parser.add_argument("--evidence-dir", type=Path, help="write masks and overlays here")
    parser.add_argument("--batch", type=Path, help="JSONL of {images, question} rows")
    parser.add_argument("--out-dir", type=Path, help="output directory for --batch")
    parser.add_argument(
        "--evidence", action="store_true", help="write per-row evidence in batch mode"
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "run one query end to end on a synthetic scene and exit non-zero if the "
            "trace is not schema-valid. This is the CI smoke test, and it is the same "
            "code path the venue demo falls back to."
        ),
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.batch:
        if not args.out_dir:
            parser.error("--batch requires --out-dir")
        failures = run_batch(args.batch, args.out_dir, write_evidence=args.evidence)
        return 1 if failures else 0

    if not args.question or not args.images:
        parser.error("--question and --images are required unless --batch is used")

    payload, trace = run_single(
        args.question,
        args.images,
        modalities=args.modalities,
        dates=args.dates,
        output_dir=args.evidence_dir,
        benchmark=args.benchmark,
        write_evidence=args.evidence_dir is not None,
    )
    if args.out:
        answer_path, trace_path = (Path(p) for p in args.out)
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        answer_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    else:
        print(json.dumps({"answer": payload, "trace": trace}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
