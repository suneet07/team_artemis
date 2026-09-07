"""Score the reBEN reference classifiers on our own cross-modal corpus.

    python scripts/eval_bifold.py probe --corpus /data/manifests/optsar_fusion.jsonl
    python scripts/eval_bifold.py score --corpus /data/manifests/optsar_fusion.jsonl \
        --model BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0 --arm sar

**Why measure a published model at all.** ``resnet50-s1`` reports 0.628 average
precision (macro) and ``resnet50-all`` 0.711, on reBEN's own test split with
reBEN's own metric. Neither number says what the model scores on *our* questions,
which are binary presence questions over a 50% floor, and it is that number the
system would be built on. The published figure is a reason to try it, not a
substitute for measuring it.

**Loaded through timm, not configilm.** Every tensor in the checkpoint is a plain
``resnet50`` key under a ``model.vision_encoder.`` prefix -- ``conv1`` widened to
2 or 12 channels, ``fc`` replaced with 19 outputs. So the documented route
(install ``configilm``, clone the model class from TU Berlin's GitLab) buys
nothing here, and skipping it removes a dependency and a network fetch from
every container start.

**The preprocessing contract is exact and unforgiving.** From
``configilm.extra.BEN_lmdb_utils`` and ``bigearthnet_common.constants``:

* S1 is ``[VH, VV]`` -- VH **first**, which is not the order the filenames or
  the Sentinel-1 convention suggest;
* S2 is ``[B02, B03, B04, B08, B05, B06, B07, B11, B12, B8A]`` -- **not sorted**,
  with B08 fourth;
* channels are standardised with reBEN's own per-band statistics, S1 in decibels
  and S2 in scaled reflectance.

Get any of that wrong and the model still returns confident probabilities; it
just returns worse ones. Which is why ``probe`` exists.

**``probe`` resolves the band order on the train split.** Our staged rasters are
a *repack* -- one stacked GeoTIFF per patch where reBEN ships per-band files --
so the internal band order is undocumented and cannot be assumed. ``probe`` runs
the candidate orders and reports which scores highest, **on train rows only**,
so the val number stays a measurement rather than a fit.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Alphabetical, as ``configilm``'s ``valid_labels_classification`` orders them.
#: The checkpoint's own ``class_names`` are "0".."18" and carry no meaning.
CLASSES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]
BY_LOWER = {c.lower(): i for i, c in enumerate(CLASSES)}

#: The grouped question our corpus asks. Answered by the strongest member.
FOREST = ("Broad-leaved forest", "Coniferous forest", "Mixed forest")

#: reBEN's own per-band statistics. S1 is in decibels, S2 in scaled reflectance.
S1_STATS = {"VV": (-12.619993741972035, 5.115911777546365),
            "VH": (-19.29044597721542, 5.464428464912864)}
S2_STATS = {
    "B02": (429.9430203, 572.41639287), "B03": (614.21682446, 582.87945694),
    "B04": (590.23569706, 675.88746967), "B05": (950.68368468, 729.89827633),
    "B06": (1792.46290469, 1096.01480586), "B07": (2075.46795189, 1273.45393088),
    "B08": (2218.94553375, 1365.45589904), "B8A": (2266.46036911, 1356.13789355),
    "B11": (1594.42694882, 1079.19066363), "B12": (1009.32729131, 818.86747235),
}

#: The order the model was trained on.
S1_ORDER = ["VH", "VV"]
S2_ORDER = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]

#: Candidate orders for what is *in our staged file*, which is a repack and so
#: undocumented. The first is configilm's own order; the second is what a
#: repacker writing bands in sorted order would produce.
S2_CANDIDATES = {
    "configilm": S2_ORDER,
    # Wavelength order, where B8A sits between B08 and B11.
    "sorted": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    # What Python's sorted() gives on the band names: "B11" < "B12" < "B8A",
    # because "1" sorts before "8". A repacker writing bands in sorted filename
    # order produces this, and the channel statistics of our staged rasters
    # match it -- SWIR1 above SWIR2, and B8A last just above B08.
    "string_sorted": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"],
}
S1_CANDIDATES = {"vh_vv": ["VH", "VV"], "vv_vh": ["VV", "VH"]}


def fingerprint(corpus: Path, arm: str, samples: int) -> dict:
    """Match each raster channel to the reBEN band whose statistics it matches.

    Averages each channel over a sample of patches and finds the published band
    mean nearest to it. Distinctive enough to be decisive: the reference means
    span 430 to 2266, and a mis-identified channel shows up as a large residual
    rather than a plausible-looking match.
    """
    import numpy as np
    import rasterio

    rows = load_rows(corpus, arm, "train", samples, 0)
    per_channel: list[list[float]] = []
    shapes = set()
    for row in rows:
        roles = row.get("image_roles", [])
        wanted = "optical" if arm in ("optical", "fused") else "sar"
        paths = [p for p, r in zip(row["images"], roles, strict=False) if r == wanted]
        if not paths:
            continue
        with rasterio.open(paths[0]) as handle:
            data = handle.read().astype("float64")
        shapes.add(data.shape[0])
        if not per_channel:
            per_channel = [[] for _ in range(data.shape[0])]
        if len(per_channel) != data.shape[0]:
            continue
        for index in range(data.shape[0]):
            per_channel[index].append(float(np.nanmean(data[index])))

    reference = S2_STATS if arm != "sar" else {
        k: (v[0], v[1]) for k, v in S1_STATS.items()
    }
    report = {"band_counts_seen": sorted(shapes), "channels": []}
    for index, values in enumerate(per_channel):
        if not values:
            continue
        observed = sum(values) / len(values)
        ranked = sorted(
            ((abs(observed - mean), name, mean) for name, (mean, _std) in reference.items())
        )
        best, second = ranked[0], ranked[1]
        report["channels"].append(
            {
                "channel": index,
                "observed_mean": round(observed, 1),
                "nearest": best[1],
                "nearest_reference_mean": round(best[2], 1),
                "residual": round(best[0], 1),
                "runner_up": second[1],
                "margin": round(second[0] - best[0], 1),
            }
        )
    inferred = [c["nearest"] for c in report["channels"]]
    report["inferred_order"] = inferred
    report["matches_configilm"] = inferred == S2_ORDER
    report["duplicate_assignments"] = sorted(
        {b for b in inferred if inferred.count(b) > 1}
    )
    report["expected_configilm"] = S2_ORDER
    return report


def cmd_bands(args) -> int:
    report = fingerprint(Path(args.corpus), args.arm, args.samples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


def load_model(name: str, channels: int):
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(name, "model.safetensors")
    state = load_file(path)
    state = {k.removeprefix("model.vision_encoder."): v for k, v in state.items()}
    model = timm.create_model("resnet50", in_chans=channels, num_classes=19)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Silence here would mean a randomly initialised layer scoring alongside
        # trained ones, which looks like a weak model rather than a broken load.
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def stack(row, arm, s1_file_order, s2_file_order):
    """Build the model's channel stack from whatever our rasters actually hold."""
    import numpy as np
    import rasterio

    images = row["images"]
    roles = row.get("image_roles", [])
    planes = []

    def read(path, file_order, wanted, stats):
        with rasterio.open(path) as handle:
            data = handle.read().astype("float32")
        index = {name: i for i, name in enumerate(file_order)}
        for name in wanted:
            mean, std = stats[name]
            planes.append((data[index[name]] - mean) / std)

    optical = [p for p, r in zip(images, roles, strict=False) if r == "optical"]
    radar = [p for p, r in zip(images, roles, strict=False) if r == "sar"]

    if arm in ("optical", "fused") and optical:
        read(optical[0], s2_file_order, S2_ORDER, S2_STATS)
    if arm in ("sar", "fused") and radar:
        read(radar[0], s1_file_order, S1_ORDER, S1_STATS)
    return np.stack(planes)


# Moved to `satquery.tools.lulc` and imported, not copied. The served path needs
# exactly this rule, and a second copy of it is how a benchmark number and a
# product answer drift apart without either looking wrong.
from satquery.tools.lulc import answer_for  # noqa: E402


def run(rows, model, arm, s1_order, s2_order, threshold, batch=64):
    import numpy as np
    import torch

    device = next(model.parameters()).device
    correct = total = 0
    per_question: dict[str, list[int]] = defaultdict(list)
    scores, truths = [], []

    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        arrays, keep = [], []
        for row in chunk:
            try:
                arrays.append(stack(row, arm, s1_order, s2_order))
                keep.append(row)
            except Exception:  # noqa: BLE001 - a missing raster skips one row
                continue
        if not arrays:
            continue
        batch_tensor = torch.from_numpy(np.stack(arrays)).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

        for row, prob in zip(keep, probs, strict=False):
            predicted = answer_for(row["question"], prob, threshold)
            if predicted is None:
                continue
            hit = int(predicted == row["answer"])
            correct += hit
            total += 1
            per_question[row["question_type"]].append(hit)
            scores.append(prob)
            truths.append(row["answer"])

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "scored": total,
        "per_question_type": {
            k: round(sum(v) / len(v), 4) for k, v in sorted(per_question.items())
        },
        "majority_answer_floor": round(
            max(Counter(truths).values()) / len(truths), 4
        ) if truths else 0.0,
    }


def load_rows(path: Path, arm: str, split: str, limit: int, seed: int):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("modality_arm") != arm or row.get("split") != split:
                continue
            rows.append(row)
    rng = random.Random(seed)
    if limit and len(rows) > limit:
        rows = rng.sample(rows, limit)
    return rows


def cmd_duet(args) -> int:
    """Run both single-modality models on the fused rows and reconcile."""
    import numpy as np
    import torch

    rows = load_rows(Path(args.corpus), "fused", args.split, args.samples, args.seed)
    print(f"{len(rows)} {args.split} row(s) on the fused arm", flush=True)

    optical_model = load_model(args.optical_model, 10)
    radar_model = load_model(args.radar_model, 2)
    device = next(radar_model.parameters()).device
    s1_order = S1_CANDIDATES[args.s1_order]
    s2_order = S2_CANDIDATES[args.s2_order]

    def predict(model, arm, order_s1, order_s2, chunk):
        arrays, keep = [], []
        for row in chunk:
            try:
                arrays.append(stack(row, arm, order_s1, order_s2))
                keep.append(row)
            except Exception:  # noqa: BLE001 - a missing raster skips one row
                continue
        if not arrays:
            return [], []
        batch = torch.from_numpy(np.stack(arrays)).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).cpu().numpy()
        return keep, probs

    agree_hit = agree_n = 0
    dis_n = dis_s1 = dis_s2 = 0
    s1_hit = s2_hit = both_n = 0

    for start in range(0, len(rows), 48):
        chunk = rows[start : start + 48]
        keep_o, probs_o = predict(optical_model, "optical", s1_order, s2_order, chunk)
        keep_r, probs_r = predict(radar_model, "sar", s1_order, s2_order, chunk)
        by_id = {id(r): i for i, r in enumerate(keep_r)}
        for i, row in enumerate(keep_o):
            j = by_id.get(id(row))
            if j is None:
                continue
            optical = answer_for(row["question"], probs_o[i], args.threshold)
            radar = answer_for(row["question"], probs_r[j], args.threshold)
            if optical is None or radar is None:
                continue
            truth = row["answer"]
            both_n += 1
            s1_hit += int(radar == truth)
            s2_hit += int(optical == truth)
            if optical == radar:
                agree_n += 1
                agree_hit += int(optical == truth)
            else:
                dis_n += 1
                dis_s1 += int(radar == truth)
                dis_s2 += int(optical == truth)

    def rate(hit, n):
        return round(hit / n, 4) if n else None

    report = {
        "rows_scored": both_n,
        "optical_model": args.optical_model,
        "radar_model": args.radar_model,
        "threshold": args.threshold,
        "optical_alone": rate(s2_hit, both_n),
        "radar_alone": rate(s1_hit, both_n),
        "agreement_rate": rate(agree_n, both_n),
        "when_they_agree": {"n": agree_n, "accuracy": rate(agree_hit, agree_n)},
        "when_they_disagree": {
            "n": dis_n,
            "optical_right": rate(dis_s2, dis_n),
            "radar_right": rate(dis_s1, dis_n),
        },
        # The policy a router would implement: take the shared answer where the
        # two agree, fall back to optical where they do not, since optical is
        # the stronger single model (0.714 vs 0.628 published).
        "policy_agree_else_optical": rate(agree_hit + dis_s2, both_n),
        "note": (
            "Both models answer the SAME rows here, so radar_alone and "
            "optical_alone are directly comparable -- unlike the per-arm scores, "
            "where the SAR arm asks only about radar-legible classes."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


def cmd_probe(args) -> int:
    """Resolve the staged band order on TRAIN rows, never on val."""
    rows = load_rows(Path(args.corpus), args.arm, "train", args.samples, args.seed)
    print(f"{len(rows)} train row(s) on the {args.arm} arm", flush=True)
    channels = 2 if args.arm == "sar" else (10 if args.arm == "optical" else 12)
    model = load_model(args.model, channels)

    results = {}
    s1_options = S1_CANDIDATES if args.arm in ("sar", "fused") else {"n/a": S1_ORDER}
    s2_options = S2_CANDIDATES if args.arm in ("optical", "fused") else {"n/a": S2_ORDER}
    for s1_name, s1_order in s1_options.items():
        for s2_name, s2_order in s2_options.items():
            key = f"s1={s1_name},s2={s2_name}"
            results[key] = run(rows, model, args.arm, s1_order, s2_order, args.threshold)
            print(f"  {key}: {results[key]['accuracy']:.4f}", flush=True)

    best = max(results, key=lambda k: results[k]["accuracy"])
    report = {"arm": args.arm, "model": args.model, "candidates": results, "best": best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


def cmd_score(args) -> int:
    rows = load_rows(Path(args.corpus), args.arm, "val", args.samples, args.seed)
    print(f"{len(rows)} val row(s) on the {args.arm} arm", flush=True)
    channels = 2 if args.arm == "sar" else (10 if args.arm == "optical" else 12)
    model = load_model(args.model, channels)

    s1_order = S1_CANDIDATES[args.s1_order]
    s2_order = S2_CANDIDATES[args.s2_order]

    # A threshold sweep, because 0.5 is a convention rather than a calibration
    # and a multi-label head trained with BCE is not calibrated for our binary
    # questions. Both numbers are reported; the swept one is labelled as swept.
    sweep = {}
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        sweep[threshold] = run(rows, model, args.arm, s1_order, s2_order, threshold)
        print(f"  threshold {threshold}: {sweep[threshold]['accuracy']:.4f}", flush=True)

    best = max(sweep, key=lambda t: sweep[t]["accuracy"])
    report = {
        "model": args.model,
        "arm": args.arm,
        "band_order": {"s1": args.s1_order, "s2": args.s2_order},
        "at_0.5": sweep[0.5],
        "best_threshold": best,
        "at_best_threshold": sweep[best],
        "sweep": {str(k): v["accuracy"] for k, v in sweep.items()},
        "note": (
            "at_0.5 is the honest out-of-the-box number. at_best_threshold is "
            "swept on this same split and is therefore optimistic; quote it only "
            "with the sweep beside it."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    duet_cmd = sub.add_parser("duet")
    duet_cmd.add_argument("--corpus", required=True)
    duet_cmd.add_argument("--optical-model", default="BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0")
    duet_cmd.add_argument("--radar-model", default="BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0")
    duet_cmd.add_argument("--split", default="val")
    duet_cmd.add_argument("--samples", type=int, default=4000)
    duet_cmd.add_argument("--threshold", type=float, default=0.5)
    duet_cmd.add_argument("--seed", type=int, default=0)
    duet_cmd.add_argument("--s1-order", default="vh_vv", choices=sorted(S1_CANDIDATES))
    duet_cmd.add_argument("--s2-order", default="configilm", choices=sorted(S2_CANDIDATES))
    duet_cmd.add_argument("--out", default="logs/bifold_duet.json")
    duet_cmd.set_defaults(func=cmd_duet)

    for name, func in (("probe", cmd_probe), ("score", cmd_score), ("bands", cmd_bands)):
        p = sub.add_parser(name)
        p.add_argument("--corpus", required=True)
        p.add_argument("--model", default="BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0")
        p.add_argument("--arm", default="sar", choices=["sar", "optical", "fused"])
        p.add_argument("--samples", type=int, default=2000)
        p.add_argument("--threshold", type=float, default=0.5)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=f"logs/bifold_{name}.json")
        if name == "score":
            p.add_argument("--s1-order", default="vh_vv", choices=sorted(S1_CANDIDATES))
            p.add_argument("--s2-order", default="configilm", choices=sorted(S2_CANDIDATES))
        p.set_defaults(func=func)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
