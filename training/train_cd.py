"""Train the Siamese change-detection model on SpaceNet 7.

    python training/train_cd.py --root /data/eval/sn7 --smoke
    python training/train_cd.py --root /data/eval/sn7 --steps 8000 \
        --out /data/checkpoints/change_map

**Why this exists alongside change_vqa.** The VQA adapter answers change
questions by inference and gets 1.2 points of its counting accuracy from
actually reading the images -- measured, not guessed. This model answers the
same questions by measurement: emit a mask, then presence, direction,
magnitude and location fall out of arithmetic on it. The VLM keeps the
questions it is genuinely good at, where it scored +12 to +19 points above the
guessing floor with real vision use.

**The smoke test reports F1, not loss.** A change mask is 98% background, so a
model predicting nothing scores 98% pixel accuracy and near-zero F1. Loss going
down says very little here; F1 over the changed class is the signal, and it is
unambiguous in a way validation cross-entropy was not for the VQA run.

Held out by AOI. Two crops from one 4 km tile share streets, so a crop-level
split would put near-duplicates on both sides of the boundary.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.cd.dataset import (  # noqa: E402
    BuildingDataset,
    ChangePairDataset,
    build_pairs,
)
from satquery.cd.model import (  # noqa: E402
    BuildingUNet,
    SiameseUNet,
    change_metrics,
    dice_bce_loss,
)

SMOKE_STEPS = 200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="/data/eval/sn7")
    parser.add_argument("--out", default="/data/checkpoints/change_map")
    parser.add_argument("--steps", type=int, default=0, help="0 = one epoch")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--crops-per-pair", type=int, default=2)
    parser.add_argument(
        "--task",
        default="change",
        choices=["change", "buildings"],
        help="'buildings' trains a per-date detector, which is what all three "
        "top SpaceNet 7 solutions actually did; change is then arithmetic",
    )
    parser.add_argument("--upscale", type=int, default=3)
    parser.add_argument("--max-gap-months", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--report", default="logs/cd_smoke.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import random

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    root = Path(args.root)
    pairs = build_pairs(root, args.max_gap_months)
    if not pairs:
        raise SystemExit(f"no usable pairs under {root}")

    aois = sorted({p["aoi"] for p in pairs})
    random.Random(args.seed).shuffle(aois)
    holdout = set(aois[: max(1, round(len(aois) * args.val_fraction))])
    train_pairs = [p for p in pairs if p["aoi"] not in holdout]
    val_pairs = [p for p in pairs if p["aoi"] in holdout]
    print(
        f"{len(pairs)} pair(s) over {len(aois)} AOI(s); "
        f"{len(train_pairs)} train / {len(val_pairs)} val "
        f"({len(holdout)} AOIs held out)",
        flush=True,
    )

    if args.task == "buildings":
        train_set = BuildingDataset(
            train_pairs, crop=args.crop, upscale=args.upscale,
            crops_per_image=args.crops_per_pair, train=True
        )
        val_set = BuildingDataset(
            val_pairs, crop=args.crop, upscale=args.upscale,
            crops_per_image=1, train=False, seed=1234
        )
    else:
        train_set = ChangePairDataset(
            train_pairs, crop=args.crop, crops_per_pair=args.crops_per_pair, train=True
        )
        val_set = ChangePairDataset(
            val_pairs, crop=args.crop, crops_per_pair=1, train=False, seed=1234
        )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(2, args.num_workers // 2),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        BuildingUNet(pretrained=True) if args.task == "buildings"
        else SiameseUNet(pretrained=True)
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {'BuildingUNet' if args.task == 'buildings' else 'SiameseUNet'}  "
        f"trainable={trainable:,}  device={device}  task={args.task}",
        flush=True,
    )

    steps = SMOKE_STEPS if args.smoke else (args.steps or len(loader))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=steps, pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history: list[dict] = []
    best = {"step": 0, "f1": 0.0, "iou": 0.0}
    out = Path(args.out)
    report = Path(args.report)
    started = time.time()
    step = 0
    model.train()
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            pixels = batch["pixels"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(pixels)
                loss = dice_bce_loss(logits, target, valid=valid)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            step += 1

            if step % 25 == 0 or step == 1:
                metrics = change_metrics(logits.float(), target, valid=valid)
                elapsed = time.time() - started
                rate = elapsed / step
                print(
                    f"  step {step:5}/{steps} {100 * step / steps:5.1f}%  "
                    f"loss {loss.item():.4f}  train F1 {metrics['f1']:.3f}  "
                    f"IoU {metrics['iou']:.3f}  "
                    f"changed {100 * metrics['positive_share']:.1f}%  "
                    f"{rate:.3f} s/step  eta {(steps - step) * rate / 60:.0f} min",
                    flush=True,
                )
            if args.val_every and step % args.val_every == 0:
                entry = _validate(model, val_loader, device, step, args)
                history.append(entry)
                # Keep the best-by-F1 checkpoint, not just the last one.
                # change_vqa's best model was step 1000 of 10,976 and only
                # survived because a monitored checkpoint existed; the rolling
                # one had long since been overwritten.
                if entry["f1"] >= best["f1"]:
                    best.update(entry)
                    out.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": model.state_dict(), **entry}, out / "cd_best.pt"
                    )
                    print(f"  [best] F1 {entry['f1']:.4f} at step {step}", flush=True)
                # A crash at 90 minutes should cost one interval, not the run.
                torch.save({"model": model.state_dict(), "step": step}, out / "cd_last.pt")
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(
                    json.dumps({"steps": step, "val_history": history, "best": best}, indent=1),
                    encoding="utf-8",
                )
                model.train()

    elapsed = time.time() - started
    if not history:
        history.append(_validate(model, val_loader, device, step, args))

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step}, out / "cd_model.pt")

    record = {
        "steps": step,
        "wall_hours": round(elapsed / 3600, 4),
        "seconds_per_step": round(elapsed / max(1, step), 4),
        "trainable_params": trainable,
        "batch_size": args.batch_size,
        "crop": args.crop,
        "pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "val_history": history,
        "best": best,
        "smoke": bool(args.smoke),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"\n{step} steps in {elapsed / 3600:.3f} h "
        f"({elapsed / max(1, step):.3f} s/step); "
        f"best val F1 {max((h['f1'] for h in history), default=0):.3f}",
        flush=True,
    )
    print(json.dumps(record, indent=1))
    return 0


def _validate(model, loader, device, step, args) -> dict:
    """F1 over held-out AOIs, accumulated rather than averaged per batch.

    Per-batch F1 averaged across batches is not F1: a batch whose crop happens
    to contain no change contributes a 0 or a 1 that means nothing. Counting
    true positives, false positives and false negatives across the whole pass
    and computing once is the only figure comparable between runs.
    """
    import torch

    model.eval()
    tp = fp = fn = 0.0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= args.val_batches:
                break
            pixels = batch["pixels"].to(device)
            target = batch["mask"].to(device)
            valid = batch["valid"].to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(pixels)
            predicted = (torch.sigmoid(logits.float()) > 0.5).float() * valid
            target = target * valid
            tp += (predicted * target).sum().item()
            fp += (predicted * (1 - target)).sum().item()
            fn += ((1 - predicted) * target).sum().item()
            seen += 1
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    print(f"  [val] step {step:5}  F1 {f1:.4f}  IoU {iou:.4f}", flush=True)
    return {"step": step, "f1": round(f1, 4), "iou": round(iou, 4)}


if __name__ == "__main__":
    raise SystemExit(main())
