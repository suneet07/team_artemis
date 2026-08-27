# Modal runbook — Phase 0 timing sweep on A100-80GB

Researched against Modal docs and the `modal-labs/modal-examples` repo (cloned and
read directly). Validated against `modal` 1.5.4.

## Why Modal

Serverless GPU, **per-second billing**, no idle cost. Better fit than a rented
Studio for a 15-minute sweep. §5.5 is written for an A100-**80GB** (C41), which
Modal exposes as `gpu="A100-80GB"`.

## Cost

| Item | Cost |
|---|---|
| A100-80GB | $0.000694/sec ≈ **$2.50/hr** |
| One ~15 min sweep | **~$0.60** |
| Starter plan | $0/mo + **$30/mo free credit** ≈ 12 A100-hours |
| §5.5's entire 50-hour budget | ~$125 of A100 time |

**Apply for Modal's academic grant (up to $10k).** An SIH/ISRO problem statement
with a public plan document is a strong application, and it would remove compute
as a constraint on this project entirely.

## Commands

```bash
pip install modal
modal setup                                        # one-time browser auth
modal run scripts/modal_timing_sweep.py --dry-run  # CPU, ~free
modal run scripts/modal_timing_sweep.py            # A100-80GB
```

Report lands at `../../logs/phase_0_timing_sweep_a100_80gb.md`.

CLI surface (verified via `modal run ... --help`): `--steps`, `--warmup-steps`,
`--micro-batch`, `--grad-accum`, `--max-pixels`, `--composites`,
`--dry-run/--no-dry-run`, `--out`.

## Gotchas that cost money if missed

1. **Default function timeout is 300 s.** A 12-cell sweep exceeds it and gets
   killed mid-run. Set `timeout=1 * HOURS`.
2. **PyTorch Lightning re-executes the process entrypoint** under multi-GPU.
   Modal's GPU guide: *"If the framework re-executes the entrypoint of the Python
   process (like PyTorch Lightning) you need to either set the strategy to
   `ddp_spawn` or `ddp_notebook`."* At `devices=1` it does not, but we launch via
   subprocess anyway (pattern from `reinforcement-learning/learn_math.py`), which
   keeps the harness Modal-agnostic.
3. **Cold starts re-download the 8 GB model** unless `HF_HOME` points at a
   persistent `modal.Volume`.
4. **`HF_HUB_ENABLE_HF_TRANSFER` is deprecated** in huggingface_hub 1.x. Use
   `HF_XET_HIGH_PERFORMANCE=1` instead — that is what the Studio run already used
   (8 GB at 1.46 GB/s).
5. **Do NOT add `modal.Retries` to a measurement function.** Retries suit long
   resumable training (`long-training.py`); on a timing run a silent retry bills
   twice and can mask a real failure.

## Idioms adopted from modal-examples

- `uv_pip_install` over `pip_install` — 147 call sites vs 10 in the repo.
- `extra_index_url="https://download.pytorch.org/whl/cu128"` to pin the exact
  torch build (per `06_gpu_and_ml/import_torch.py`). **One** install call, not
  two — lightning and transformers both depend on torch and a second pass can
  re-resolve it.
- `add_local_file(Path(__file__).parent / "...")` per
  `14_clusters/simple_torch_cluster.py`. Resolving relative to the file, not the
  working directory, means `modal run` works from anywhere. Verified.
- Last image layer, **not** `copy=True` — edits re-upload at container start
  instead of triggering an image rebuild.
- `MINUTES = 60` / `HOURS = 60 * MINUTES` constants.
- `--dry-run` routed to a **CPU** function: it only builds the processor and
  checks label masking, so it needs no GPU.

## Open at time of writing

Two fixes land together on the next run — the 80 GB card AND
`use_reentrant=False` for gradient checkpointing (the 40 GB sweep showed
activation memory scaling linearly with seq_len, meaning checkpointing was not
running). Do not attribute the improvement to the card alone. The report's
activation-scaling row now reports which one actually mattered.
