"""Run every gallery item through the live app and record what came back.

The gallery's whole claim is that these items behave in the product the way they
behaved in the benchmark. That is only worth stating if it has been checked, so
this uploads each item's imagery, asks its question through the ordinary
``/queries`` route -- the same route the UI uses, no special path -- and scores
the reply.

**Scored through the benchmark's own contract.** Each item carries the
``contract`` that ``contract_for(benchmark, question_type)`` resolved at build
time, and the reply goes through ``format_answer`` with it before comparison.
Skipping that would compare a raw generation against a gold produced by a
formatter: RSVQA ``count`` quantisation alone was worth +7.3 points, so "about
12" would read as a failure when the benchmark counted it right. Grounding is
scored by IoU >= 0.5 against the reference box and captioning by ROUGE-L
against the reference caption, because those are what their benchmarks used.

Nothing is dropped silently. Every item keeps its result and its reason, and the
summary reports pass rate per segment, so a gallery built from this is a
documented sample rather than a hand-picked one.

    python scripts/verify_gallery.py --base https://...  [--segments rs_vqa,sar]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "gallery" / "manifest.json"
RESULTS = ROOT / "data" / "gallery" / "verified.json"

sys.path.insert(0, str(ROOT))


def _post_scene(base: str, path: Path, requests):
    with path.open("rb") as handle:
        response = requests.post(
            f"{base}/scenes", files={"file": (path.name, handle)}, timeout=600
        )
    response.raise_for_status()
    return response.json()["scene_id"]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _boxes_from(result: dict) -> list[list[float]]:
    """Every box the trace carries, whichever step produced it."""
    boxes: list[list[float]] = []
    for step in (result.get("trace") or {}).get("steps") or []:
        for key in ("boxes", "bbox_list"):
            for box in (step.get("outputs") or {}).get(key) or []:
                if isinstance(box, dict):
                    box = box.get("bbox_xyxy_normalised") or box.get("bbox")
                if isinstance(box, list) and len(box) == 4:
                    boxes.append([float(v) for v in box])
    return boxes


def _rouge_l(prediction: str, reference: str) -> float:
    """Longest-common-subsequence F1 over whitespace tokens."""
    pred = prediction.lower().split()
    ref = reference.lower().split()
    if not pred or not ref:
        return 0.0
    table = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i, p in enumerate(pred, 1):
        for j, r in enumerate(ref, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if p == r else max(table[i - 1][j], table[i][j - 1])
            )
    lcs = table[-1][-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def _score(item: dict, result: dict) -> tuple[bool, str, str]:
    """(passed, served_answer_as_scored, reason)."""
    from satquery.evalcli.formatter import format_answer

    answer = str(result.get("answer") or "").strip()
    contract = item.get("contract")
    gold = str(item["gold"]).strip()

    if contract == "box_iou_0.5":
        reference = item["gold"]
        if isinstance(reference, str):
            reference = json.loads(reference)
        boxes = _boxes_from(result)
        if not boxes:
            return False, "", "no box returned"
        best = max(_iou(box, list(reference)) for box in boxes)
        return best >= 0.5, json.dumps([round(v, 4) for v in boxes[0]]), f"IoU {best:.3f}"

    if contract == "rouge_l":
        score = _rouge_l(answer, gold)
        floor = float(item.get("blind_floor_rouge_l") or 0.0)
        return score >= floor, answer, f"ROUGE-L {score:.4f} vs floor {floor:.4f}"

    if contract == "yes_no":
        formatted = format_answer(answer, "ben_binary_vqa", question=item["question"])
        return str(formatted).strip().lower() == gold.lower(), str(formatted), formatted.basis

    if contract:
        formatted = format_answer(answer, contract, question=item["question"])
        return (
            str(formatted).strip().lower() == gold.lower(),
            str(formatted),
            f"{contract}: {formatted.basis}",
        )

    return answer.lower() == gold.lower(), answer, "raw comparison, no contract"


def main() -> int:
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="API base, ending /api/v1")
    parser.add_argument("--segments", default="", help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--status",
        default="",
        help="only items with this manifest status (e.g. candidate), so a "
        "top-up costs 31 GPU calls rather than re-running 200 settled ones",
    )
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = manifest["items"]
    if args.segments:
        keep = {s.strip() for s in args.segments.split(",") if s.strip()}
        items = [item for item in items if item["segment"] in keep]
    if args.status:
        items = [item for item in items if item.get("status") == args.status]
    if args.limit:
        items = items[: args.limit]

    print(f"verifying {len(items)} item(s) against {args.base}", flush=True)
    results: list[dict] = []
    scene_cache: dict[str, str] = {}
    started = time.time()

    for index, item in enumerate(items, 1):
        record = dict(item)
        try:
            scene_ids = []
            for image in item["images"]:
                path = ROOT / image["path"]
                if not path.exists():
                    raise FileNotFoundError(path)
                key = str(path)
                if key not in scene_cache:
                    scene_cache[key] = _post_scene(args.base, path, requests)
                scene_ids.append({"scene_id": scene_cache[key], "role": image["role"]})

            bundle = requests.post(
                f"{args.base}/bundles", json={"scenes": scene_ids}, timeout=300
            )
            bundle.raise_for_status()
            query = requests.post(
                f"{args.base}/queries",
                json={"bundle_id": bundle.json()["bundle_id"], "question": item["question"]},
                timeout=900,
            )
            query.raise_for_status()
            result = query.json()

            passed, served, reason = _score(item, result)
            record.update(
                {
                    "served_answer": served,
                    "served_raw": str(result.get("answer") or "")[:400],
                    "passed": passed,
                    "reason": reason,
                    "tools": [
                        s.get("tool")
                        for s in (result.get("trace") or {}).get("steps") or []
                    ],
                    "confidence": result.get("confidence"),
                }
            )
        except Exception as error:  # noqa: BLE001 - one bad item must not end the run
            record.update({"passed": False, "reason": f"{type(error).__name__}: {error}"[:300]})

        results.append(record)
        if index % 10 == 0 or index == len(items):
            ok = sum(1 for r in results if r.get("passed"))
            rate = (time.time() - started) / index
            print(
                f"  {index}/{len(items)}  passing {ok}  ({rate:.1f}s/item)", flush=True
            )

    per_segment: dict[str, Counter] = defaultdict(Counter)
    for record in results:
        per_segment[record["segment"]]["total"] += 1
        per_segment[record["segment"]]["passed" if record.get("passed") else "failed"] += 1

    summary = {
        "base": args.base,
        "checked": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "per_segment": {
            name: {
                **dict(counts),
                "pass_rate": (
                    round(counts["passed"] / counts["total"], 4) if counts["total"] else 0.0
                ),
            }
            for name, counts in sorted(per_segment.items())
        },
        "items": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
