"""LoRA adapter training — the script master plan Part 7 lists and the repo lacked.

    python training/train_lora.py --adapter rs_vqa --manifest data/rs_vqa.jsonl \
        --image-root data --out checkpoints/rs_vqa

    python training/train_lora.py --adapter rs_vqa --manifest ... --smoke
        200 steps against guard rail 1, then stop. Ten minutes to protect twelve hours.

Everything numeric comes from the frozen contract by way of
:class:`satquery.training.TrainingConfig`: ``max_pixels`` from
``configs/preprocessing.yaml``, the LoRA hyperparameters and schedule from
section 5.2, the composite count from the adapter. Nothing is restated here, so
this trainer and the timing sweep cannot measure and train different configs --
which is what happened last time and is why the budget table had to be thrown
out.

The section 5.5 guard rails are enforced rather than documented:

1. **200-step smoke on the exact config** before a full run -- ``--smoke``, and
   the trainer refuses a run longer than the smoke threshold unless
   ``--smoke-passed`` names the smoke report it is trusting.
2. **Checkpoint every ~500 steps**, adapter plus optimizer state, so a crash
   costs minutes rather than a run.
3. ``rs_vqa`` runs first -- not enforceable in code, stated in the help text.
4. The budget table is re-derived from measurement, not estimate: every run
   writes its own seconds-per-step so the burn-down is real.
5. Hours consumed are appended to ``logs/hour_burndown.jsonl`` on every run.
"""

import argparse
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.training.collator import MaskedCollator, supervised_span  # noqa: E402
from satquery.training.config import ADAPTERS, TrainingConfig  # noqa: E402
from satquery.training.dataset import (  # noqa: E402
    RealChipDataset,
    load_canonical_manifest,
)
from satquery.training.lora import (  # noqa: E402
    audit_checkpointing,
    build_lora_model,
    lr_lambda_for,
)

#: Guard rail 1. A run longer than this is a real run and needs a smoke first.
SMOKE_STEPS = 200
CHECKPOINT_EVERY = 500


def build_dataloader(args, cfg: TrainingConfig, processor, split: str | None = None):
    """One loader for a split. ``split=None`` uses ``args.split``.

    Validation gets ``shuffle=False`` so the same rows are scored in the same
    order at every check -- a val curve that moves because the sample changed
    tells you nothing about the model.
    """
    import torch

    split = split or args.split
    samples = load_canonical_manifest(
        args.manifest, adapter=args.adapter, split=split
    )
    dataset = RealChipDataset(
        samples,
        root=args.image_root or Path(args.manifest).parent,
        composites=cfg.composites,
        gsd_conditioning=not args.no_gsd_conditioning,
    )
    training = split == "train"
    print(f"Corpus: {len(dataset)} samples for {args.adapter} split={split}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.micro_batch,
        shuffle=training,
        num_workers=cfg.num_workers,
        collate_fn=MaskedCollator(processor),
        drop_last=training,
        pin_memory=torch.cuda.is_available(),
    )


def train(args) -> dict:
    import lightning as pl
    import torch

    cfg = TrainingConfig.for_adapter(
        args.adapter,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
        devices=args.devices,
        seed=args.seed,
        **({"max_pixels": args.max_pixels} if args.max_pixels else {}),
    )
    print(cfg.provenance())
    if not cfg.frozen_max_pixels:
        print(
            "  [warn] max_pixels overridden on a TRAINING run. Section 5.5: extra "
            "VRAM is not a reason to relax it -- train on what the deployed system "
            "will see, or S3 comes back by the back door."
        )

    pl.seed_everything(cfg.seed, workers=True)
    bundle = build_lora_model(cfg)
    audit = audit_checkpointing(bundle.model)
    print(
        f"Model: {bundle.attention}  trainable={bundle.trainable_params:,} "
        f"({bundle.trainable_fraction:.3%})  checkpointing {audit.active}/{audit.layers}"
    )
    if audit.warning():
        print(f"  [warn] {audit.warning()}")

    loader = build_dataloader(args, cfg, bundle.processor)
    steps = args.steps if args.steps else len(loader) // cfg.grad_accum * cfg.epochs

    # Held-out AOIs, scored during the run rather than after it. Capped: the
    # point is a comparable number every few hundred steps, and scoring 8,000
    # rows every check would cost more GPU time than the training between them.
    val_loader = None
    if not args.smoke and not args.no_val:
        try:
            val_loader = build_dataloader(args, cfg, bundle.processor, split="val")
        except ValueError as error:
            print(f"  [warn] no validation split ({error}); running blind.")

    if steps > SMOKE_STEPS and not (args.smoke or args.smoke_passed):
        raise SystemExit(
            f"Refusing a {steps}-step run with no smoke test. Section 5.5 guard rail "
            f"1: never start a full run without a {SMOKE_STEPS}-step smoke on that "
            "exact config -- ten minutes to protect twelve hours. Run with --smoke "
            "first, then pass --smoke-passed <report path>."
        )

    module = _LoraTrainer(cfg, bundle, steps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    class AdapterSnapshot(pl.pytorch.callbacks.Callback):
        """Export the LoRA adapter every N steps, keeping all of them.

        Defined here rather than at module scope because ``pl`` is imported
        lazily inside this function -- the parameter gate and the headless CPU
        path import this module without Lightning installed.

        The ``ModelCheckpoint`` below rotates: a Lightning checkpoint carries
        the frozen 4.5B base beside the 40M LoRA weights, so it is 3.81 GB and
        keeping one per 1,000 steps would cost tens of gigabytes of Volume for
        a base byte-identical in every file. An adapter is ~160 MB, so all of
        them can be kept -- which is what makes "train once, compare
        checkpoints" affordable. That comparison is not hypothetical: on the
        SpaceNet 7 run overall accuracy *fell* from 0.450 at step 1,000 to
        0.416 at step 4,000, and only two surviving checkpoints revealed it.
        """

        def __init__(self, out, every, processor):
            self.out = Path(out)
            self.every = int(every)
            self.processor = processor
            self.written = []

        def on_train_batch_end(self, trainer, module, *args):
            step = trainer.global_step
            if self.every <= 0 or step <= 0 or step % self.every:
                return
            target = self.out / f"adapter_step{step}"
            if target.exists():
                # An earlier run on a different corpus wrote here: the output
                # directory is per *adapter*, not per run, so `adapter_step4000`
                # from a SpaceNet 7 run and from a CDVQA run collide by name.
                #
                # This used to `return`, which silently left the stale adapter
                # in place. Evaluating it would have scored the wrong corpus's
                # weights and read as "the new training failed" -- a wrong
                # conclusion with no way to see the cause. Moving it aside keeps
                # both and makes the collision visible in the file listing.
                stale = self.out / f"{target.name}.superseded"
                if stale.exists():
                    shutil.rmtree(stale, ignore_errors=True)
                target.rename(stale)
                print(
                    f"  [snapshot] {target.name} existed from an earlier run; "
                    f"moved to {stale.name}",
                    flush=True,
                )
            module.model.save_pretrained(target)
            self.processor.save_pretrained(target)
            self.written.append(str(target))
            print(f"  [snapshot] adapter at step {step} -> {target}", flush=True)

    callbacks = [
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=out,
            every_n_train_steps=CHECKPOINT_EVERY,
            # Two recent, not every one ever written. A checkpoint here is
            # 3.81 GB -- Lightning stores the frozen 4.5B base alongside the
            # 40M LoRA weights -- so `save_top_k=-1` over a 10,976-step run
            # meant 84 GB of Volume for weights that are identical in every
            # file and re-downloaded from HuggingFace on resume anyway.
            # Resume needs `last`; picking a model needs `best`, kept below.
            # 1 and not 2: with no `monitor` there is nothing to rank by, and
            # Lightning rejects any top_k above 1 in that configuration.
            save_top_k=1,
            save_last=True,
            filename="{step}",
        )
    ]
    if args.snapshot_every:
        callbacks.append(
            AdapterSnapshot(out, args.snapshot_every, bundle.processor)
        )
    if val_loader is not None:
        # Kept separately from the every-500-steps checkpoints so the best one
        # is identifiable without reading a log: after a run that went too long,
        # `best.ckpt` is the adapter worth evaluating.
        callbacks.append(
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=out,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best",
            )
        )
    logger = _wandb_logger(args, cfg)

    trainer = pl.Trainer(
        max_steps=SMOKE_STEPS if args.smoke else steps,
        accumulate_grad_batches=cfg.grad_accum,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.devices,
        precision="bf16-mixed",
        logger=logger or False,
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=10,
        # Every 500 steps, aligned with the checkpoint interval so a val number
        # always has a checkpoint beside it.
        # Step-based across the whole run, not per epoch. With the default
        # (check every epoch) Lightning requires the interval to be no larger
        # than one epoch's batches, so a 500-step check silently becomes
        # per-epoch on a small corpus and errors outright on a large batch
        # size -- which is exactly how a 2-epoch run would have died at
        # startup after the queue had already been paid for.
        check_val_every_n_epoch=None if val_loader is not None else 1,
        val_check_interval=(
            min(
                args.val_every or CHECKPOINT_EVERY,
                max(1, len(loader) - 1),
            )
            if val_loader is not None
            else 1.0
        ),
        limit_val_batches=args.val_batches,
        num_sanity_val_steps=0,
        enable_progress_bar=not args.quiet,
    )

    started = time.time()
    trainer.fit(module, loader, val_loader, ckpt_path=args.resume)
    elapsed = time.time() - started

    adapter_dir = out / "adapter"
    bundle.model.save_pretrained(adapter_dir)
    bundle.processor.save_pretrained(adapter_dir)

    record = {
        "adapter": args.adapter,
        "smoke": bool(args.smoke),
        "steps": trainer.global_step,
        "val_history": getattr(module, "val_history", []),
        "adapter_snapshots": next(
            (c.written for c in callbacks if hasattr(c, "written")), []
        ),
        "wall_hours": round(elapsed / 3600, 4),
        "sec_per_step": round(elapsed / max(1, trainer.global_step), 4),
        "samples": len(loader.dataset),
        "effective_batch": cfg.effective_batch,
        "attention": bundle.attention,
        "checkpointing_active": f"{audit.active}/{audit.layers}",
        "first_loss": module.first_loss,
        "last_loss": module.last_loss,
        "supervised_preview": module.supervised_preview,
        "peak_vram_gb": round(
            torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
            2,
        ),
        "config": asdict(cfg),
        "host": platform.node(),
        "adapter_dir": str(adapter_dir),
    }
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    _append_burndown(record)
    return record


class _LoraTrainer:
    """Constructed lazily so importing this module needs no torch."""

    def __new__(cls, cfg: TrainingConfig, bundle, total_steps: int):
        import lightning as pl
        import torch

        class Module(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.cfg = cfg
                self.model = bundle.model
                self.processor = bundle.processor
                self.total_steps = total_steps
                self.first_loss = 0.0
                self.last_loss = 0.0
                self.supervised_preview = ""
                self.val_history: list[tuple[int, float]] = []

            def on_train_start(self):
                # Belt and braces: build_lora_model already called .train(), but
                # Lightning can restore eval mode after a validation loop and the
                # failure is silent in both directions.
                self.model.train()

            def training_step(self, batch, batch_idx):
                if batch_idx == 0 and not self.supervised_preview:
                    # What the loss is actually computed over. If this is not the
                    # answer text, the mask is broken and the run is worthless.
                    self.supervised_preview = supervised_span(self.processor, batch)[:200]
                    print(f"  [labels] supervised span: {self.supervised_preview!r}")
                loss = self.model(**batch).loss
                value = float(loss.detach())
                if not self.first_loss:
                    self.first_loss = value
                    if value > 9.0:
                        print(
                            f"  [warn] opening loss {value:.2f} is close to "
                            "ln(vocab)=11.93 -- the label mask is probably broken and "
                            "the model is being scored on padding."
                        )
                self.last_loss = value
                self.log("train_loss", loss, prog_bar=True)
                return loss

            def validation_step(self, batch, batch_idx):
                """Loss on held-out AOIs -- the only signal for over-fitting.

                Nothing else in this file can say when a run should stop. Train
                loss falls monotonically whether the model is learning the task
                or memorising 60 scenes, and the two look identical until the
                benchmark comes back flat. Val loss on AOIs the run has never
                seen separates them: while it tracks train loss the model is
                generalising, and the step where it turns upward is where the
                useful part of the run ended.
                """
                loss = self.model(**batch).loss
                self.log("val_loss", loss, prog_bar=True, sync_dist=True)
                return loss

            def on_validation_epoch_end(self):
                value = self.trainer.callback_metrics.get("val_loss")
                if value is None:
                    return
                value = float(value)
                self.val_history.append((int(self.global_step), round(value, 4)))
                best = min(v for _, v in self.val_history)
                marker = "  <- best" if value <= best else ""
                print(
                    f"  [val] step {self.global_step:>6}  val_loss {value:.4f}"
                    f"  train_loss {self.last_loss:.4f}{marker}",
                    flush=True,
                )
                # Rising val while train still falls is the definition of
                # memorising. Said once, loudly, rather than left in a curve
                # nobody opens.
                if len(self.val_history) >= 3:
                    recent = [v for _, v in self.val_history[-3:]]
                    if recent[0] < recent[1] < recent[2]:
                        print(
                            "  [warn] val_loss has risen for three consecutive "
                            f"checks ({recent[0]:.4f} -> {recent[2]:.4f}). The run is "
                            "past its useful length; the checkpoint at the best "
                            "val step is the one to keep.",
                            flush=True,
                        )

            def configure_optimizers(self):
                params = [p for p in self.model.parameters() if p.requires_grad]
                optimiser = torch.optim.AdamW(params, lr=cfg.init_lr, weight_decay=0.0)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimiser, lr_lambda_for(cfg, self.total_steps)
                )
                return {
                    "optimizer": optimiser,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
                }

        return Module()


def _wandb_logger(args, cfg: TrainingConfig):
    """W&B if asked for and installed. Never a silent no-op."""
    if not args.wandb:
        return None
    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError as error:
        raise SystemExit(
            "--wandb needs the wandb extra: pip install wandb. Refusing to run "
            "unlogged when logging was requested."
        ) from error
    return WandbLogger(
        project=args.wandb_project,
        name=f"{args.adapter}-{'smoke' if args.smoke else 'full'}",
        config=asdict(cfg),
    )


def _append_burndown(record: dict) -> None:
    """Guard rail 5: hours consumed, appended, never overwritten.

    Guard rail 4 says the budget table is re-derived from measurement. A row of
    ``{steps, wall_hours}`` cannot do that: seconds-per-step depends on the
    attention kernel, on whether checkpointing was actually active, and on the
    vision sequence length -- and the first smoke run on Modal differed from the
    plan's assumptions on the first two. A row that records only the elapsed
    time looks comparable to every other row while measuring something else, so
    the conditions travel with the number.
    """
    path = REPO_ROOT / "logs" / "hour_burndown.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "adapter": record["adapter"],
                    "smoke": record["smoke"],
                    "steps": record["steps"],
                    "wall_hours": record["wall_hours"],
                    "sec_per_step": record["sec_per_step"],
                    # The conditions the timing is only valid under.
                    "attention": record["attention"],
                    "checkpointing_active": record["checkpointing_active"],
                    "effective_batch": record["effective_batch"],
                    "peak_vram_gb": record["peak_vram_gb"],
                    "samples": record["samples"],
                    "host": record["host"],
                }
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one LoRA adapter. Guard rail 3: run rs_vqa first -- it is the "
            "cheapest adapter and exercises the whole data -> train -> serve path, "
            "so broken wiring surfaces at hour 4 rather than hour 16."
        )
    )
    parser.add_argument("--adapter", required=True, choices=ADAPTERS)
    parser.add_argument("--manifest", required=True, help="canonical JSONL corpus")
    parser.add_argument("--image-root", default=None, help="defaults to the manifest dir")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="checkpoints/adapter")
    parser.add_argument("--steps", type=int, default=0, help="0 = one epoch")
    parser.add_argument(
        "--snapshot-every", type=int, default=0,
        help="export a PEFT adapter every N steps; all are kept (~160 MB each)",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="skip the held-out check; only for timing runs, never for training",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="steps between held-out checks; 0 uses the checkpoint interval, so "
        "every val number has a checkpoint beside it",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=40,
        help="batches per validation check; enough for a stable number, not so "
        "many that checking costs more than the training between checks",
    )
    parser.add_argument("--smoke", action="store_true", help=f"{SMOKE_STEPS} steps, then stop")
    parser.add_argument(
        "--smoke-passed", default=None, help="path to the smoke run.json being trusted"
    )
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        help="override the frozen value. Recorded and warned about; section 5.5 "
        "says do not do this on a training run.",
    )
    parser.add_argument("--no-gsd-conditioning", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="satquery")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    record = train(args)
    print(
        f"\n{record['adapter']}: {record['steps']} steps in {record['wall_hours']:.2f} h "
        f"({record['sec_per_step']:.3f} s/step), loss "
        f"{record['first_loss']:.3f} -> {record['last_loss']:.3f}, "
        f"peak {record['peak_vram_gb']} GB"
    )
    print(f"Adapter written to {record['adapter_dir']}")


if __name__ == "__main__":
    main()
