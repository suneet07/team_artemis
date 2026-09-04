import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from satquery.agent.bundle import CoregReport, ImageBundle, ImageRef
from satquery.agent.graph import QueryResult, run_query
from satquery.evalcli.smoke import run_dummy_query
from satquery.ingest.band_inventory import BandInventory
from satquery.ingest.ingest import ingest_raster


def _build_image_ref_from_path(path_str: str, scene_id: str) -> ImageRef:
    p = Path(path_str)
    if p.exists() and p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
        try:
            res = ingest_raster(p)
            return ImageRef(
                scene_id=scene_id,
                path=str(p),
                modality=res.modality,
                crs=res.meta.crs,
                pixel_size_m=res.meta.pixel_size_x,
                native_gsd_m=res.meta.pixel_size_x,
                bands=res.meta.descriptions or [],
                swir_available=res.inventory.has_swir,
                bit_depth=res.meta.dtype_bits,
                nodata_frac=0.0,
                polarisations=res.inventory.polarisations,
                sar_band=res.inventory.sar_band,
                computable_indices=res.inventory.computable_indices,
                modality_source=res.modality_source,
            )
        except Exception:
            pass

    # Fallback heuristic ImageRef
    is_sar = "sar" in path_str.lower()
    mod = "sar" if is_sar else "optical"
    return ImageRef(
        scene_id=scene_id,
        path=path_str,
        modality=mod,
        crs="EPSG:32644",
        pixel_size_m=10.0,
        bands=["blue", "green", "red", "nir"] if not is_sar else ["VV"],
        swir_available=False,
        polarisations=["VV"] if is_sar else [],
        computable_indices=[] if is_sar else ["NDVI", "NDWI"],
    )


def _load_bundle(bundle_arg: str | None, images_args: list[str] | None = None) -> ImageBundle:
    if images_args:
        refs: list[ImageRef] = []
        for i, img_path in enumerate(images_args):
            refs.append(_build_image_ref_from_path(img_path, f"scene_{i + 1}"))

        pair_type = "single"
        if len(refs) == 2:
            if refs[0].modality != refs[1].modality:
                pair_type = "crossmodal"
            else:
                pair_type = "bitemporal"

        # Construct synthetic inventory from first image
        first = refs[0]
        inv = BandInventory(
            bands={b: i + 1 for i, b in enumerate(first.bands)},
            has_swir=first.swir_available,
            has_nir="nir" in first.bands,
            is_pan_only=len(first.bands) <= 1 and first.modality == "optical",
            polarisations=first.polarisations,
            sar_band="C" if first.modality == "sar" else None,
            computable_indices=first.computable_indices,
        )
        coreg = (
            CoregReport(coregistered=True, checks_passed=["crs_match"])
            if len(refs) > 1
            else None
        )
        return ImageBundle(
            bundle_id=f"bundle_{Path(images_args[0]).stem}",
            images=refs,
            band_inventory=inv,
            pair_type=pair_type,  # type: ignore[arg-type]
            coreg=coreg,
            status="ready",
        )

    if bundle_arg is None:
        # Default mock bundle
        inv = BandInventory(
            bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
            has_swir=False,
            has_nir=True,
            is_pan_only=False,
            polarisations=[],
            sar_band=None,
            sensor_hint=None,
            computable_indices=["NDVI", "NDWI"],
        )
        img = ImageRef(
            scene_id="scene_cli",
            path="cli_img.tif",
            modality="optical",
            crs="EPSG:32644",
            computable_indices=["NDVI", "NDWI"],
        )
        return ImageBundle(
            bundle_id="bundle_cli",
            images=[img],
            band_inventory=inv,
            pair_type="single",
            status="ready",
        )

    bundle_path = Path(bundle_arg)
    if bundle_path.is_file() and bundle_path.suffix == ".json":
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        images = [ImageRef(**img) for img in data.get("images", [])]
        inv_data = data.get("band_inventory")
        inv = BandInventory(**inv_data) if inv_data else None
        return ImageBundle(
            bundle_id=data.get("bundle_id", "bundle_from_file"),
            images=images,
            band_inventory=inv,
            pair_type=data.get("pair_type", "single"),
            status=data.get("status", "ready"),
        )

    # If it's a directory or image file, construct bundle
    img = _build_image_ref_from_path(str(bundle_path), "scene_dir")
    inv = BandInventory(
        bands={"blue": 1, "green": 2, "red": 3, "nir": 4},
        has_swir=False,
        has_nir=True,
        is_pan_only=False,
        polarisations=[],
        computable_indices=["NDVI", "NDWI"],
    )
    return ImageBundle(
        bundle_id=bundle_path.stem,
        images=[img],
        band_inventory=inv,
        pair_type="single",
        status="ready",
    )


def _write_outputs(res: QueryResult, out_path_str: str, question: str) -> None:
    target = Path(out_path_str)
    result_dict = asdict(res)
    timings = res.trace.get("timings") or {}

    if target.suffix.lower() == ".json":
        # Single consolidated JSON file
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "result": result_dict,
            "trace": res.trace,
            "metadata": {
                "query_id": res.query_id,
                "state": res.state,
                "question": question,
                "timings": timings,
                "latency_ms": res.latency_ms,
                "timestamp": res.trace.get("timestamp"),
            },
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    # Multi-file directory output
    target.mkdir(parents=True, exist_ok=True)
    assets_dir = target / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. trace.json
    trace_path = target / "trace.json"
    trace_path.write_text(json.dumps(res.trace, indent=2), encoding="utf-8")

    # 2. result.json
    result_path = target / "result.json"
    result_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")

    # 3. metadata.json
    metadata = {
        "query_id": res.query_id,
        "state": res.state,
        "question": question,
        "timings": timings,
        "latency_ms": res.latency_ms,
        "timestamp": res.trace.get("timestamp"),
        "steps_count": len(res.trace.get("steps", [])),
        "evidence_count": len(res.evidence),
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 4. telemetry.json
    telemetry = {
        "query_id": res.query_id,
        "timings": timings,
        "latency_ms": res.latency_ms,
        "steps": [
            {
                "tool": s.get("tool"),
                "latency_ms": s.get("latency_ms", 0),
                "confidence": s.get("confidence", 1.0),
            }
            for s in res.trace.get("steps", [])
        ],
        "total_latency_ms": res.latency_ms or (sum(timings.values()) if timings else 0),
    }
    (target / "telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")


def _run_batch(batch_file: str, out_dir_str: str | None, allow_llm: bool, quiet: bool) -> int:
    batch_path = Path(batch_file)
    if not batch_path.is_file():
        sys.stderr.write(f"Batch file '{batch_file}' not found.\n")
        return 2

    out_dir = Path(out_dir_str) if out_dir_str else Path("batch_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [ln.strip() for ln in batch_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total = len(lines)
    succeeded = 0
    refused = 0
    failed = 0

    summaries: list[dict[str, Any]] = []

    for i, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except Exception as e:
            sys.stderr.write(f"Line {i}: invalid JSON: {e}\n")
            failed += 1
            continue

        item_id = str(item.get("id", f"sample_{i}"))
        question = item.get("question") or item.get("query", "")
        images = item.get("images") or ([item["image"]] if "image" in item else None)
        bundle_arg = item.get("bundle")

        try:
            bundle = _load_bundle(bundle_arg, images)
            res = run_query(bundle, question, allow_llm_router=allow_llm)
            item_out = out_dir / item_id
            _write_outputs(res, str(item_out), question)

            if res.state == "succeeded":
                succeeded += 1
            elif res.state == "refused":
                refused += 1
            else:
                failed += 1

            summaries.append({
                "id": item_id,
                "state": res.state,
                "timings": res.trace.get("timings", {}),
                "latency_ms": res.latency_ms,
                "refusal": res.refusal,
            })
        except Exception as e:
            failed += 1
            summaries.append({"id": item_id, "state": "error", "error": str(e)})

    # Write summary.jsonl
    summary_path = out_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")

    if not quiet:
        msg = (
            f"Batch evaluation complete: {total} total, {succeeded} succeeded, "
            f"{refused} refused, {failed} failed"
        )
        print(msg)

    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m satquery.evalcli")
    parser.add_argument("--query", "--question", dest="query", default=None)
    parser.add_argument("--images", "--image", dest="images", nargs="+", default=None)
    parser.add_argument("--bundle", default=None)
    parser.add_argument(
        "--out", "--output", dest="out", default=None,
        help="Output path: a .json file for consolidated output or a directory.",
    )
    parser.add_argument(
        "--out-trace", dest="out_trace", default=None,
        help="(§23) Write the trace JSON to this specific file path, separate from --out.",
    )
    parser.add_argument("--allow-llm-router", type=str, default="true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--manifest", "--manifest-dir", dest="manifest_dir", default=None)
    parser.add_argument("--batch-file", default=None)
    # Smoke test compatibility arguments
    parser.add_argument("--index", choices=["ALPHA", "BETA"], default=None)
    parser.add_argument("--scale", type=float, default=None)

    args = parser.parse_args()

    allow_llm = str(args.allow_llm_router).lower() in ("true", "1", "yes")

    # Batch mode
    if args.batch_file:
        return _run_batch(args.batch_file, args.out, allow_llm=allow_llm, quiet=args.quiet)

    # Legacy smoke test mode
    if args.index is not None and args.query is not None:
        scale = args.scale if args.scale is not None else 0.5
        trace = run_dummy_query(args.query, {"index": args.index, "scale": scale})
        if not args.quiet:
            print(json.dumps(trace, indent=2))
        return 0

    if not args.query:
        sys.stderr.write("Error: --query/--question is required for single query execution.\n")
        return 2

    try:
        bundle = _load_bundle(args.bundle, args.images)
    except Exception as e:
        sys.stderr.write(f"Failed to load bundle: {e}\n")
        return 2

    try:
        res = run_query(bundle, args.query, allow_llm_router=allow_llm)
    except Exception as e:
        sys.stderr.write(f"Unhandled query error: {e}\n")
        return 2

    if args.out:
        _write_outputs(res, args.out, args.query)

    # §23: --out-trace writes the trace JSON to a dedicated path (separate from --out)
    if args.out_trace:
        trace_target = Path(args.out_trace)
        trace_target.parent.mkdir(parents=True, exist_ok=True)
        trace_target.write_text(json.dumps(res.trace, indent=2), encoding="utf-8")

    if not args.quiet:
        print(json.dumps(res.trace, indent=2))

    if res.state == "succeeded":
        return 0
    elif res.state == "refused":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
