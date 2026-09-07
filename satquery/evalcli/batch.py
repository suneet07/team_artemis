"""Headless batch evaluation (master plan section 4.11).

    python -m satquery.evalcli.batch --input queries.jsonl --output results/

Each input row is ``{"images": [...], "question": "...", ...}`` — the shape any
benchmark harness or hidden-set evaluation will take. Rows may also carry
``id``, ``modalities``, ``dates`` and ``benchmark``; ``benchmark`` runs the
answer through the section 6.4 formatter so the string matches what the official
scorer expects.

This module used to write ``"Mocked answer for spectral_index"`` and set
``parameter_check.passed = True`` unconditionally. Both are worse than a missing
feature: the second is a *false statement in a graded field*, and it would have
been indistinguishable from a real run in the submitted artifact. Every row now
goes through the real pipeline, and a row that fails is recorded as a failure
rather than as a confident answer.
"""

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["process_batch"]


def process_batch(
    input_path: str, output_path: str, *, write_evidence: bool = False
) -> dict[str, int]:
    """Answer every row of ``input_path``, writing answers and traces.

    ``output_path`` may be a directory (answers.jsonl + traces.jsonl are written
    into it) or a single ``.jsonl`` file, in which case traces go next to it.
    Returns a ``{"rows": n, "failed": n}`` summary.
    """
    from satquery.evalcli.__main__ import run_single

    in_file = Path(input_path)
    out = Path(output_path)
    if out.suffix == ".jsonl":
        out.parent.mkdir(parents=True, exist_ok=True)
        answers_path = out
        traces_path = out.with_name(out.stem + "_traces.jsonl")
        evidence_root = out.parent
    else:
        out.mkdir(parents=True, exist_ok=True)
        answers_path = out / "answers.jsonl"
        traces_path = out / "traces.jsonl"
        evidence_root = out

    rows = 0
    failed = 0
    with (
        in_file.open(encoding="utf-8") as source,
        answers_path.open("w", encoding="utf-8") as answers_file,
        traces_path.open("w", encoding="utf-8") as traces_file,
    ):
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            payload: dict[str, Any]
            try:
                record = json.loads(line)
                images = record.get("images") or record.get("image_paths")
                question = record.get("question") or record.get("query")
                if not images or not question:
                    raise ValueError("row needs both 'images' and 'question'")
                payload, trace = run_single(
                    question,
                    list(images),
                    modalities=record.get("modalities"),
                    dates=record.get("dates"),
                    output_dir=(
                        evidence_root / f"row_{line_number:06d}" if write_evidence else None
                    ),
                    benchmark=record.get("benchmark"),
                    write_evidence=write_evidence,
                )
                if "id" in record:
                    payload["id"] = record["id"]
            except Exception as error:  # noqa: BLE001 - one row must not lose the batch
                failed += 1
                message = f"{type(error).__name__}: {error}"
                payload = {"id": None, "line": line_number, "error": message}
                trace = {"line": line_number, "error": message}
            answers_file.write(json.dumps(payload) + "\n")
            traces_file.write(json.dumps(trace) + "\n")

    print(f"wrote {answers_path} and {traces_path}")
    if failed:
        print(f"{failed} of {rows} row(s) failed; see the error field in answers.jsonl")
    return {"rows": rows, "failed": failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SatQuery headless batch runner (4.11)")
    parser.add_argument("--input", required=True, help="input JSONL file")
    parser.add_argument("--output", required=True, help="output directory or .jsonl file")
    parser.add_argument(
        "--evidence", action="store_true", help="write masks and overlays per row"
    )
    args = parser.parse_args(argv)
    summary = process_batch(args.input, args.output, write_evidence=args.evidence)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
