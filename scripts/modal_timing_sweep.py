"""
Modal runner for scripts/phase0_timing_sweep.py on an A100-80GB.

Master plan v3.8 §5.5 is titled "A100-80GB budget (C41)" and every config in it
is sized for 80 GB -- "a 4B base fits 80 GB trivially", "raise batch size until
utilisation saturates". The first sweep ran on a 40 GB card and hit an OOM wall
at seq_len ~570. This runs the same harness on the hardware the plan assumes.

Quick start
-----------
    pip install modal
    modal setup                                          # one-time browser auth
    modal run scripts/modal_timing_sweep.py --dry-run    # CPU, ~free, verifies masking
    modal run scripts/modal_timing_sweep.py              # A100-80GB sweep, ~$0.60

The report is written back to logs/phase_0_timing_sweep_a100_80gb.md locally.

Design notes (patterns taken from modal-labs/modal-examples)
------------------------------------------------------------
* `uv_pip_install` over `pip_install` -- what the examples use almost
  everywhere (147 call sites vs 10) and materially faster to build.
* torch pinned through `extra_index_url=.../cu128`, matching
  06_gpu_and_ml/import_torch.py, so we get exactly the torch 2.8.0+cu128 build
  the harness was validated against rather than whatever PyPI resolves to.
* `add_local_file` with `Path(__file__).parent`, as in
  14_clusters/simple_torch_cluster.py. Resolving relative to THIS file rather
  than the working directory means `modal run` works from any directory.
  It is the last image layer and is NOT `copy=True`, so editing the harness
  re-uploads the file at container start instead of rebuilding the image.
* Subprocess launch, as in 06_gpu_and_ml/reinforcement-learning/learn_math.py.
  Modal's GPU guide warns that PyTorch Lightning re-executes the process
  entrypoint under multi-GPU. At devices=1 it does not, but the subprocess keeps
  the harness Modal-agnostic -- the same file runs unchanged on any machine.
* NO `modal.Retries`. Retries are right for long resumable training
  (06_gpu_and_ml/long-training.py); they are wrong for a measurement, where a
  silent re-run would bill twice and could hide a real failure.
* `--dry-run` is routed to a CPU function. It only builds the processor and
  checks label masking -- no model, no GPU, so there is no reason to pay for one.

Cost
----
A100-80GB is $0.000694/sec (~$2.50/hour), billed per second. A ~15 min sweep is
about $0.60. The Starter plan includes $30/month free, roughly 12 A100 hours.
Modal also grants academic credits up to $10k -- worth applying for, since
§5.5's entire 50-hour budget is only ~$125 of A100 time.
"""

from pathlib import Path

import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

APP_NAME = "satquery-phase0-timing"
GPU = "A100-80GB"

CACHE_DIR = "/cache"
REMOTE_SCRIPT = "/root/phase0_timing_sweep.py"
REMOTE_OUT = "/root/sweep_report.md"

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "lightning==2.6.4",
        "transformers==5.16.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "pillow>=10.0",
        "numpy>=1.24",
        # Pins the cu128 build the harness was validated against. One install
        # call, not two -- lightning and transformers both depend on torch, and
        # a second pass could re-resolve it against a different index.
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            # Weights land on the Volume, so a cold start does not re-pull 8 GB.
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            # hf_transfer is deprecated in huggingface_hub 1.x; Xet is the
            # transport now (the 40 GB run pulled 8 GB at 1.46 GB/s on it).
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Reduces fragmentation-driven OOM on the long-sequence cells.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    # Last layer, and deliberately not copy=True: the harness is still being
    # edited, and this way a change re-uploads rather than rebuilding the image.
    .add_local_file(
        HERE / "phase0_timing_sweep.py",
        remote_path=REMOTE_SCRIPT,
    )
)

app = modal.App(APP_NAME, image=image)

# Persistent HF cache. Survives between runs, so the 8 GB Qwen3-VL download
# happens once rather than on every cold start.
cache_volume = modal.Volume.from_name("satquery-hf-cache", create_if_missing=True)


def _run_harness(args: list[str], write_report: bool) -> str:
    """Launch the harness as a subprocess. Returns the report markdown, if any."""
    import subprocess
    import sys

    cmd = [sys.executable, REMOTE_SCRIPT, *args]
    if write_report:
        cmd += ["--out", REMOTE_OUT]

    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=False)

    if not write_report:
        if proc.returncode != 0:
            raise RuntimeError(f"dry run failed with exit code {proc.returncode}")
        return ""

    out = Path(REMOTE_OUT)
    if not out.exists():
        raise RuntimeError(f"sweep exited with code {proc.returncode} and wrote no report")
    return out.read_text(encoding="utf-8")


@app.function(
    gpu=GPU,
    timeout=1 * HOURS,  # Modal's DEFAULT is 300 s -- a 12-cell sweep exceeds it
    volumes={CACHE_DIR: cache_volume},
)
def sweep(args: list[str]) -> str:
    import subprocess

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    report = _run_harness(args, write_report=True)
    cache_volume.commit()  # keep the weights for the next run
    return report


@app.function(
    timeout=20 * MINUTES,
    volumes={CACHE_DIR: cache_volume},
)
def check_shapes(args: list[str]) -> str:
    """Shape and label-masking check. No model is loaded, so no GPU is needed."""
    _run_harness(args, write_report=False)
    cache_volume.commit()
    return ""


@app.local_entrypoint()
def main(
    steps: int = 50,
    warmup_steps: int = 10,
    micro_batch: int = 4,
    grad_accum: int = 1,
    max_pixels: int = 0,
    composites: int = 0,
    source_size: int = 0,
    grad_checkpointing: bool = True,
    dry_run: bool = False,
    out: str = "logs/phase_0_timing_sweep_a100_80gb.md",
) -> None:
    """
    CLI flags use dashes: --steps, --warmup-steps, --micro-batch, --grad-accum,
    --max-pixels, --composites, --grad-checkpointing/--no-grad-checkpointing,
    --dry-run, --out.

    Pass --max-pixels / --composites to run a single cell instead of the
    full 12-cell sweep.
    """
    args = [
        "--steps",
        str(steps),
        "--warmup-steps",
        str(warmup_steps),
        "--micro-batch",
        str(micro_batch),
        "--grad-accum",
        str(grad_accum),
    ]
    if max_pixels:
        args += ["--max-pixels", str(max_pixels)]
    if composites:
        args += ["--composites", str(composites)]
    if source_size:
        args += ["--source-size", str(source_size)]
    if not grad_checkpointing:
        args += ["--no-grad-checkpointing"]

    if dry_run:
        print("Dry run (CPU) — shapes and label masking only...")
        check_shapes.remote([*args, "--dry-run"])
        return

    print(f"Running sweep on {GPU}...")
    report = sweep.remote(args)

    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report, encoding="utf-8")
    print(f"\nReport written to {dest}")
