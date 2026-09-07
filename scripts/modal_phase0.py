"""Modal runners for the Phase 0 ML jobs on an A100-80GB.

    pip install modal && modal setup                       # one-time

    modal run scripts/modal_phase0.py::smoke               # item 12, 200 steps
    modal run scripts/modal_phase0.py::train --adapter rs_vqa   # resumable
    modal run scripts/modal_phase0.py::bakeoff             # item 9
    modal run scripts/modal_phase0.py::d2                  # item 10
    modal run scripts/modal_phase0.py::router              # item 11
    modal run scripts/modal_phase0.py::ablation --experiment optical_composites
    modal run scripts/modal_phase0.py::eval_sweep          # all of them, one load
    modal run scripts/modal_phase0.py::vllm_smoke          # item 7

Companion to ``modal_timing_sweep.py``, which prices configs. This one runs the
jobs that produce Phase 0 *findings*: the smoke test that guard rail 1 requires
before any twelve-hour run, the base-model bake-off, the two ablations, the
router arms and the serving check.

Shared with the timing runner by design:

* the same ``satquery-hf-cache`` Volume, so the 8 GB Qwen3-VL download happens
  once across every job here;
* the same pinned cu128 torch build the harnesses were validated against;
* no ``modal.Retries``. A silent re-run of a measurement bills twice and can
  hide a real failure.

Different by necessity:

* the whole repo is mounted, not one file. These harnesses import
  ``satquery.training`` and ``satquery.agent`` -- that shared library is the
  point, and shipping a single script would recreate the island problem it was
  written to fix.
* a data Volume, ``satquery-data``, holds staged chips and manifests. Imagery
  does not belong in the image layer: it changes far more often than the
  dependencies and would invalidate the build on every restage.
* ``router`` runs on CPU. It is a text classifier over 300 short prompts and
  the rules arm needs no model at all.

Reports are written back into ``logs/`` locally, so a run leaves the same
artifact whether it happened here or on a workstation.
"""

import json
import shutil
from pathlib import Path

import modal

MINUTES = 60
HOURS = 60 * MINUTES

#: The base models the bake-off compares when --models is not given. Mirrors
#: training/eval/zero_shot.py; kept here only so the dump filenames can be
#: predicted without importing the harness locally.
CANDIDATE_BASES = ("Qwen/Qwen3-VL-4B-Instruct",)

APP_NAME = "satquery-phase0-ml"

#: How often a long training job flushes its checkpoints to the Volume. Ten
#: minutes bounds what a crash can cost to roughly one checkpoint interval,
#: against the six hours an uncommitted run would lose.
COMMIT_EVERY_SECONDS = 600
GPU = "A100-80GB"

CACHE_DIR = "/cache"
DATA_DIR = "/data"
REPO_DIR = "/root/satquery"

HERE = Path(__file__).parent.parent

base_image = (
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
        "pyyaml>=6.0",
        # Every runtime dependency `pyproject.toml` declares must be here, or a
        # container that imports satquery dies on the first request that
        # touches the missing one. Two have been found the slow way already --
        # scikit-image (below) and jsonschema, which only surfaced when
        # /meta/health imported the trace validator. `tests/test_modal_image.py`
        # now compares the two lists so the next one fails locally instead.
        "jsonschema>=4.21",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        # scikit-IMAGE, not scikit-learn -- they are different packages and
        # having one installed says nothing about the other. `pyproject.toml`
        # declares scikit-image>=0.22 and six satquery modules import it
        # (coreg, tools.catalog, tools.cv_utils, tools.thresholds, cd.dataset,
        # api.server), so leaving it out crash-looped every container that
        # imported a tool: the building-detector run died on it once, and the
        # deployed API died on it again.
        "scikit-image>=0.22",
        "rasterio>=1.3",
        "pandas>=2.0",
        "pyarrow>=15.0",
        # The reBEN reference classifiers are plain timm resnet50s under a
        # wrapper prefix, so timm alone loads them -- no configilm, no GitLab
        # clone at container start.
        "timm>=1.0",
        # The imagery is published only as Google Drive archives.
        "gdown>=5.2",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # FlashAttention-2, from the project's own prebuilt wheel rather than a
    # source build. `pip install flash-attn` compiles CUDA kernels and routinely
    # takes 30-40 minutes on a build container; this is a 244 MB download.
    #
    # The ABI variant is detected from torch, not guessed. Both cxx11abiTRUE and
    # cxx11abiFALSE wheels exist for every release, the wrong one installs
    # cleanly and then fails at *import*, and the fallback in lora.py:95 would
    # quietly drop back to sdpa -- which is exactly how the first smoke run
    # measured 0.836 s/step on sdpa while the §5.5 budget assumed FA2.
    .run_commands(
        'ABI=$(python -c "import torch; print(\'TRUE\' if torch._C._GLIBCXX_USE_CXX11_ABI'
        ' else \'FALSE\')") && '
        "pip install --no-cache-dir "
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
        "flash_attn-2.8.3.post1+cu12torch2.8cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl && "
        # Fail the build here rather than at run time. An image that ships a
        # broken flash-attn silently costs a GPU run to discover.
        'python -c "import flash_attn; print(\'flash-attn\', flash_attn.__version__)"'
    )
    # It ships as a zip containing a nested zip and a .rar, and Python's
    # stdlib reads neither RAR nor solid archives. p7zip reads both.
    .apt_install("p7zip-full", "unar")
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": REPO_DIR,
        }
    )
)

#: The repo, mounted last and without ``copy=True``: the harnesses are still
#: being edited, so a change re-uploads the tree at container start rather than
#: rebuilding the image.
#:
#: Modal forbids any build step *after* ``add_local_dir``, so this cannot be
#: folded into ``base_image`` -- ``vllm_image`` needs to install on top of the
#: same dependencies, and doing that after the mount fails every entrypoint in
#: this file with "an image tried to run a build step after using
#: image.add_local_*".
def _with_repo(built):
    return built.add_local_dir(
        HERE,
        remote_path=REPO_DIR,
        ignore=[
            "*.pyc",
            "__pycache__",
            ".git",
            ".env",
            ".env.*",
            # The whole console, not just its node_modules. The GPU container
            # serves an API; it never reads a .tsx file. Including it meant a
            # frontend edit -- or a `tsc -b` writing tsconfig.node.tsbuildinfo --
            # landed mid-upload and killed the deploy with "was modified during
            # build process". Three deploys died that way.
            "frontend",
            "data",
            "checkpoints",
            "graphify-out",
            "ben-micro-split/images",
            # 24 MB of run reports, over half the payload, and the container
            # writes logs rather than reading them.
            "logs",
            # Prose. `verify_routes.py` reads frontend/contracts/types.ts, but it
            # runs locally, never in the container.
            "docs",
            "sample_dataset",
            ".pytest_cache",
            ".ruff_cache",
        ],
    )


image = _with_repo(base_image)

# vLLM is a heavy, version-sensitive install and only item 7 needs it. Keeping
# it out of the shared image means a serving-stack change cannot invalidate the
# training image or vice versa.
vllm_image = _with_repo(base_image.uv_pip_install("vllm==0.11.0"))

#: Florence-2 ships its modelling code in the repo and loads it through
#: ``trust_remote_code``. That code was written against transformers 4.x and
#: does not import under the 5.16 the rest of this file pins, so the pin is
#: overridden here rather than downgraded everywhere -- a single baseline
#: measurement is not worth moving four adapters onto an older library.
#:
#: ``einops`` and ``timm`` are Florence's own undeclared imports; without them
#: the remote code fails at load with a bare ModuleNotFoundError.
florence_image = _with_repo(
    base_image.uv_pip_install(
        "transformers==4.51.3",
        "einops>=0.7",
        "timm>=0.9",
    )
)


app = modal.App(APP_NAME, image=image)

cache_volume = modal.Volume.from_name("satquery-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("satquery-data", create_if_missing=True)

VOLUMES = {CACHE_DIR: cache_volume, DATA_DIR: data_volume}


def _run(argv: list[str], *, reports: list[str]) -> dict[str, str]:
    """Run one harness in-repo and return whatever reports it wrote.

    Reports come back as strings rather than being left on the Volume so that a
    local ``modal run`` produces the same files a local run would. A finding
    that only exists inside a container is a finding nobody will read.
    """
    import subprocess
    import sys

    print("+ " + " ".join(argv), flush=True)
    proc = subprocess.run(argv, cwd=REPO_DIR, stdout=sys.stdout, stderr=sys.stderr, check=False)

    collected = {}
    for name in reports:
        path = Path(REPO_DIR) / name
        if path.exists():
            collected[name] = path.read_text(encoding="utf-8")
    if not collected:
        raise RuntimeError(
            f"exit code {proc.returncode} and none of {reports} was written. The job "
            "produced no artifact, so there is nothing to report and nothing to trust."
        )
    return collected


def _save(reports: dict[str, str]) -> None:
    for name, text in reports.items():
        target = HERE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")


@app.function(gpu=GPU, timeout=2 * HOURS, volumes=VOLUMES, image=florence_image)
def florence_job(argv: list[str], reports: list[str]) -> dict[str, str]:
    import subprocess

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    result = _run(argv, reports=reports)
    cache_volume.commit()
    data_volume.commit()
    return result


#: Evaluations, smokes and ramps. Deliberately NOT the 24 h maximum: everything
#: on this path finishes in minutes, so a long timeout here buys nothing and
#: lets a hung eval bill a day of A100 before anyone notices.
@app.function(gpu=GPU, timeout=4 * HOURS, volumes=VOLUMES)
def gpu_job(argv: list[str], reports: list[str]) -> dict[str, str]:
    import subprocess

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    result = _run(argv, reports=reports)
    cache_volume.commit()
    data_volume.commit()
    return result


#: Real training runs, which every adapter needs: the shortest measured adapter
#: is 4.0 h and the longest 15.7 h, so **not one of them fits the eval path's
#: timeout**. 24 h is Modal's maximum and covers the longest adapter with room
#: for a slower-than-measured run.
#:
#: No ``modal.Retries``. A retry restarts the timeout and silently re-bills a
#: half-finished run; resuming from the checkpoint is the deliberate version of
#: the same recovery, and it is a decision a person should make after looking at
#: why the run stopped.
@app.function(gpu=GPU, timeout=24 * HOURS, volumes=VOLUMES)
def train_job(
    argv: list[str], reports: list[str], out_dir: str, resume: bool
) -> dict[str, str]:
    """Run a training job, resuming from its own checkpoint when one exists.

    Resume is resolved **here**, not in the local entrypoint: the checkpoint
    lives on the Volume, which the caller's machine cannot see. Deciding
    locally would mean deciding from a path that is always absent and always
    starting from scratch.

    ``last.ckpt`` and not the newest ``step=N.ckpt``: it carries optimizer and
    scheduler state, so the LR schedule continues instead of restarting warm at
    step 0 -- which trains, looks entirely normal, and quietly ruins the run.
    """
    import subprocess
    from pathlib import Path as P

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    data_volume.reload()  # another invocation may have written the checkpoint
    checkpoint = P(out_dir) / "last.ckpt"
    if resume and checkpoint.exists():
        print(f"resuming from {checkpoint}", flush=True)
        argv = [*argv, "--resume", str(checkpoint)]
    elif resume:
        print(f"no checkpoint at {checkpoint}; starting fresh", flush=True)

    # Commit while the job runs, not only when it returns.
    #
    # Lightning writes last.ckpt to the mounted Volume every 500 steps, and the
    # resume path above depends on it being there. But a Volume write is only
    # guaranteed durable once committed, and committing after `_run` means a
    # crash, a preemption or a timeout at hour five persists nothing -- the
    # resume finds no checkpoint and the whole run starts again from zero. A
    # six-hour job is exactly the case that cannot afford that.
    import threading

    stop = threading.Event()

    def _commit_periodically():
        while not stop.wait(COMMIT_EVERY_SECONDS):
            try:
                data_volume.commit()
                print("  [volume] checkpoint state committed", flush=True)
            except Exception as error:  # noqa: BLE001 - never kill training
                print(f"  [volume] commit failed: {error!r}", flush=True)

    committer = threading.Thread(target=_commit_periodically, daemon=True)
    committer.start()
    try:
        result = _run(argv, reports=reports)
    finally:
        stop.set()
        committer.join(timeout=30)
        cache_volume.commit()
        data_volume.commit()
    return result


@app.function(timeout=30 * MINUTES, volumes=VOLUMES)
def cpu_job(argv: list[str], reports: list[str]) -> dict[str, str]:
    result = _run(argv, reports=reports)
    data_volume.commit()
    return result


#: Corpus generation, which ``cpu_job``'s 30 minutes does not cover. The caption
#: build reached 4,250 of 6,000 SpaceNet 2 tiles and was killed with a
#: ``FunctionTimeoutError`` -- and SpaceNet 6 had not started. Eight cores
#: because the work is opening GeoTIFF headers off a Volume, which is waiting,
#: not computing.
#:
#: Six hours rather than the 24 h maximum: nothing here should approach that,
#: and a job that does has a bug worth catching rather than a budget worth
#: extending.
@app.function(timeout=6 * HOURS, volumes=VOLUMES, cpu=8.0)
def corpus_job(argv: list[str], reports: list[str]) -> dict[str, str]:
    result = _run(argv, reports=reports)
    data_volume.commit()
    return result


@app.function(gpu=GPU, timeout=1 * HOURS, volumes=VOLUMES, image=vllm_image)
def vllm_job(argv: list[str], reports: list[str]) -> dict[str, str]:
    result = _run(argv, reports=reports)
    cache_volume.commit()
    return result


#: A second training function on H100, for measuring whether the faster card is
#: also the cheaper one. H100 is 1.58x the A100's price ($3.95 vs $2.50/h), so
#: it pays for itself only above a 1.58x speedup -- which is plausible for a
#: bf16 LoRA fine-tune but is an estimate until measured. Same code, same data,
#: one flag apart.
@app.function(gpu="H100", timeout=24 * HOURS, volumes=VOLUMES)
def train_job_h100(
    argv: list[str], reports: list[str], out_dir: str, resume: bool
) -> dict[str, str]:
    import subprocess
    from pathlib import Path as P

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    data_volume.reload()
    checkpoint = P(out_dir) / "last.ckpt"
    if resume and checkpoint.exists():
        print(f"resuming from {checkpoint}", flush=True)
        argv = [*argv, "--resume", str(checkpoint)]
    result = _run(argv, reports=reports)
    cache_volume.commit()
    data_volume.commit()
    return result


def _volume_path(value: str) -> str:
    """Undo Git Bash's MSYS path conversion for absolute Volume paths.

    Git Bash rewrites a leading ``/data`` argument into
    ``C:/Program Files/Git/data`` before the process ever sees it. That path
    does not exist in the container, so the job dies on a FileNotFoundError
    naming a Windows directory -- which reads like a bug in the job rather than
    in the shell that launched it. Undoing it here means the same command works
    from PowerShell and from Git Bash.
    """
    # Both "/data/manifests/x.jsonl" and a bare "/data" have to be recovered.
    # Matching only "/data/" with its trailing slash let `--image-root /data`
    # through untouched, so the manifest resolved and the imagery did not, and
    # the corpus checker reported all 400 sampled images missing -- a drift
    # error for a corpus that was in fact staged correctly.
    if ":" not in value:
        return value
    for marker in ("/data/", "/cache/"):
        if marker in value:
            return value[value.index(marker) :]
    for bare in ("/data", "/cache"):
        if value.endswith(bare):
            return bare
    return value


@app.local_entrypoint()
def smoke(
    adapter: str = "rs_vqa",
    manifest: str = "/data/manifests/rs_vqa_train.jsonl",
    image_root: str = "/data",
    micro_batch: int = 8,
    grad_accum: int = 1,
    gpu: str = "A100",
) -> None:
    """Phase 0 item 12 — the 200-step smoke on the exact training config.

    Guard rail 1, and the item that is still open: the existing sweep logs are
    50 steps and 5 steps, on synthetic tiles, with gradient checkpointing
    inert. This runs the real trainer on the real manifest and writes a
    ``run.json`` whose ``checkpointing_active`` field says whether the
    activation-memory numbers mean anything this time.
    """
    argv = [
        "python",
        "training/train_lora.py",
        "--adapter", adapter,
        "--manifest", _volume_path(manifest),
        "--image-root", _volume_path(image_root),
        "--out", f"/data/checkpoints/{adapter}_smoke",
        "--micro-batch", str(micro_batch),
        "--grad-accum", str(grad_accum),
        "--smoke",
        "--quiet",
    ]
    runner = train_job_h100 if gpu.upper().startswith("H100") else gpu_job
    if runner is gpu_job:
        _save(runner.remote(argv, ["logs/hour_burndown.jsonl"]))
    else:
        _save(
            runner.remote(
                argv, ["logs/hour_burndown.jsonl"], "/data/checkpoints/_smoke", False
            )
        )


@app.local_entrypoint()
def bakeoff(
    benchmarks: str = "rsvqa_lr=/data/eval/rsvqa_lr/rsvqa_lr_test.jsonl",
    image_root: str = "",
    batch_size: int = 8,
    adapter: str = "",
    composites: int = 0,
    models: str = "",
    limit: int = 0,
    ablate_images: str = "",
    dump: str = "",
    out: str = "logs/phase0_item9_bakeoff",
) -> None:
    """Phase 0 item 9 — Qwen3-VL-4B against Qwen3.5-2B, zero-shot.

    ``--benchmarks`` is a comma-separated list of ``name=path`` pairs; pass a
    grounding set named ``grounding=...`` to get acc@0.5 alongside AA, which is
    the number gate G3 turns on.

    ``image_root`` is empty by default and should usually stay that way. Each
    benchmark stages into its own directory and writes image paths relative to
    it, so one shared root cannot be right for two of them -- left unset, every
    manifest resolves against its own directory instead.
    """
    argv = ["python", "training/eval/zero_shot.py", "bakeoff", "--batch-size", str(batch_size)]
    if image_root:
        argv += ["--image-root", _volume_path(image_root)]
    if adapter:
        # Without this the entrypoint could only ever score the zero-shot
        # baseline -- the flag existed in the harness and was never plumbed
        # through, so the checkpoints we actually train were unscoreable here.
        argv += ["--adapter", _volume_path(adapter)]
    if composites:
        argv += ["--composites", str(composites)]
    if limit:
        argv += ["--limit", str(limit)]
    if models:
        argv += ["--models", *[m.strip() for m in models.split(",") if m.strip()]]
    if ablate_images:
        # Pair each question with the WRONG image. A score that survives this
        # was never coming from the picture -- it is the answer prior, and the
        # gap between the two runs is the only honest measure of how much the
        # model is actually seeing.
        argv += ["--ablate-images", ablate_images]
    if dump:
        # The raw generations, so an unparsable answer can be read rather than
        # counted. A low instruction-following rate is either the model not
        # answering or the formatter not recognising a real answer, and those
        # need opposite fixes.
        argv += ["--dump-predictions", dump]
    argv += ["--out", f"{out}.md"]
    for entry in benchmarks.split(","):
        # "name=path", so the path half needs the MSYS fix on its own. Passing
        # the pair through untouched let Git Bash rewrite /data into
        # C:/Program Files/Git/data inside the value, where _volume_path never
        # looked -- the fifth place this shell rewrite has surfaced.
        name, _, path = entry.strip().partition("=")
        argv += ["--benchmark", f"{name}={_volume_path(path)}" if path else name]
    reports = [f"{out}.md", f"{out}.json"]
    if dump:
        # --dump-predictions takes a DIRECTORY and writes one
        # "{model}_{benchmark}.jsonl" per pair inside it, so the report to
        # collect is the file, not the directory that was passed in.
        for entry in benchmarks.split(","):
            name = entry.strip().partition("=")[0]
            for model in (models.split(",") if models else CANDIDATE_BASES):
                slug = model.strip().replace("/", "_").replace(".", "-")
                reports.append(f"{dump}/{slug}_{name}.jsonl")
    _save(gpu_job.remote(argv, reports))


@app.local_entrypoint()
def d2(
    samples: str = "/data/eval/d2_200.jsonl",
    image_root: str = "/data/eval",
    limit: int = 200,
) -> None:
    """Phase 0 item 10 — centroid prior against no prior."""
    argv = [
        "python", "training/eval/zero_shot.py", "d2",
        "--samples", samples,
        "--image-root", _volume_path(image_root),
        "--limit", str(limit),
    ]
    _save(gpu_job.remote(argv, ["logs/phase0_item10_d2.md", "logs/phase0_item10_d2.json"]))


@app.local_entrypoint()
def router(arms: str = "rules", model: str = "") -> None:
    """Phase 0 item 11 — rules, llm and hybrid arms on the 300-query set.

    The rules arm needs no model and runs on CPU. Add ``--arms "rules llm
    hybrid" --model Qwen/Qwen3.5-2B`` to exercise the tie-breaker, which routes
    the job to a GPU.
    """
    argv = ["python", "training/eval/router_zero_shot.py", "--arms", *arms.split()]
    if model:
        argv += ["--model", model]
    reports = ["logs/phase0_item11_router.md", "logs/phase0_item11_router.json"]
    runner = gpu_job if model else cpu_job
    _save(runner.remote(argv, reports))


@app.local_entrypoint()
def ablation(
    experiment: str = "optical_composites",
    manifest: str = "/data/eval/india_holdout_v0.jsonl",
    image_root: str = "/data/eval",
    adapter: str = "",
    limit: int = 0,
) -> None:
    """One of the four ablations. ``--adapter`` attaches a trained LoRA."""
    argv = [
        "python", "training/eval/run_ablations.py",
        "--experiment", experiment,
        "--manifest", _volume_path(manifest),
        "--image-root", _volume_path(image_root),
    ]
    if adapter:
        argv += ["--adapter", adapter]
    if limit:
        argv += ["--limit", str(limit)]
    _save(
        gpu_job.remote(
            argv,
            [f"logs/ablation_{experiment}.md", f"logs/ablation_{experiment}.json"],
        )
    )


#: Benchmarks that can be pulled straight into the Volume. Each entry is the
#: files to fetch and whether to unzip them; nothing here is fetched to the
#: developer's laptop first.
#:
#: VRSBench is **val only, deliberately**. Its images are DOTA-derived and
#: training on them is forbidden (C20), so `Images_train.zip` -- 8 GB -- is not
#: listed. Fetching data you are barred from using is 8 GB of liability.
STAGEABLE = {
    "rsvqa_lr": {
        "base": "https://zenodo.org/records/6344334/files",
        "files": [
            ("Images_LR.zip", True),
            ("all_questions.json", False),
            ("all_answers.json", False),
            ("LR_split_test_questions.json", False),
            ("LR_split_train_questions.json", False),
        ],
        # Both splits, and they are staged differently on purpose.
        #
        # **train is capped at 24 questions per image.** RSVQA-LR annotates 572
        # images with 57,223 training questions -- about 101 each, against
        # BEN.txt's 6. Merged uncapped, half an rs_vqa corpus would come from
        # 572 pictures seen a hundred times over, and the plan asks for
        # RSVQA-LR as a "train minority".
        #
        # **test is uncapped.** An evaluation set with questions removed is no
        # longer comparable to anyone's published numbers, including the
        # RSVQA paper's own.
        "stage": [
            [
                "python", "scripts/stage_rsvqa_lr.py", "--root", "{root}",
                "--split", "train", "--max-per-image", "24",
            ],
            [
                "python", "scripts/stage_rsvqa_lr.py", "--root", "{root}",
                "--split", "test",
            ],
        ],
    },
    "ben_txt": {
        "base": (
            "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/"
            "BigEarthNet.txt/resolve/main"
        ),
        # Text only, 445 MB. The imagery is NOT downloaded: 464k patches is a
        # 59 GB archive and the corpus needs 20k of them, so those arrive as
        # windowed Copernicus reads at ~200 KB each. This is the plan's
        # "manifest-only downloads, full-corpus downloads banned" rule.
        "files": [("BigEarthNet.txt.parquet", False)],
        "stage": [
            [
                "python", "scripts/stage_ben_txt.py",
                "--parquet", "{root}/BigEarthNet.txt.parquet",
                "--split", "train",
                "--patches", "20000",
                "--rows-per-adapter", "60000",
                "--out", "/data/manifests/ben_txt_train.jsonl",
            ],
            [
                "python", "scripts/stage_ben_txt.py",
                "--parquet", "{root}/BigEarthNet.txt.parquet",
                "--split", "bench",
                "--patches", "5000",
                "--rows-per-adapter", "20000",
                "--out", "/data/manifests/ben_txt_bench.jsonl",
            ],
        ],
    },
    "vrsbench_val": {
        "base": "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main",
        "files": [
            ("Images_val.zip", True),
            ("VRSBench_EVAL_referring.json", False),
            ("VRSBench_EVAL_vqa.json", False),
            ("VRSBench_EVAL_Cap.json", False),
        ],
        "stage": None,  # no canonical converter yet
    },
}


def _remote_size(url: str) -> int:
    """Content-Length for a URL, or 0 when the server will not say."""
    import urllib.request

    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "satquery/0"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            return int(response.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001 - an unavailable HEAD is not fatal
        return 0


def _extract_parallel(archive_path, root, *, workers: int = 32) -> int:
    """Unzip into a Volume with concurrent writes, reporting progress.

    ``ZipFile.extractall`` writes members one at a time. That is fine on a local
    disk and slow against a Volume, which is a network filesystem: a 3.8 GB
    archive downloads in a couple of minutes and then spends far longer being
    written out as several thousand separate objects. The writes are latency-
    bound rather than bandwidth-bound, so they parallelise well.

    Each worker opens its own ``ZipFile`` handle -- a single handle is a shared
    cursor and is not thread-safe, and sharing one produces corrupt files rather
    than an error.

    Progress is printed as it goes. The first version of this printed nothing
    between "extracting" and "done", so a slow unzip was indistinguishable from
    a hung one for twelve minutes.
    """
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as P

    root = P(root)
    with zipfile.ZipFile(archive_path) as probe:
        members = [m for m in probe.infolist() if not m.is_dir()]
    total = len(members)
    print(f"extracting {total} entries with {workers} workers", flush=True)

    # Directories first, serially: creating them concurrently races.
    for member in members:
        (root / member.filename).parent.mkdir(parents=True, exist_ok=True)

    done = 0

    def _write(chunk):
        with zipfile.ZipFile(archive_path) as handle:
            for member in chunk:
                target = root / member.filename
                if target.exists() and target.stat().st_size == member.file_size:
                    continue
                with handle.open(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=1 << 20)
        return len(chunk)

    stride = max(1, total // (workers * 4))
    chunks = [members[i : i + stride] for i in range(0, total, stride)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for written in pool.map(_write, chunks):
            done += written
            print(f"  {done}/{total}", flush=True)
    return total


@app.function(timeout=2 * HOURS, volumes=VOLUMES)
def stage_job(name: str) -> str:
    """Download one benchmark into the data Volume, on Modal's network.

    The obvious route -- fetch to a laptop, then ``modal volume put`` -- moves
    every byte twice and needs local disk for data nobody opens locally. This
    runs on a CPU container inside Modal, so the bytes travel once.

    No GPU: this is I/O, and renting an A100 to run ``urlopen`` is the most
    expensive way to download a file.
    """
    import urllib.request
    from pathlib import Path as P

    spec = STAGEABLE[name]
    root = P(DATA_DIR) / "eval" / name
    root.mkdir(parents=True, exist_ok=True)

    for filename, unzip in spec["files"]:
        target = root / filename
        url = f"{spec['base']}/{filename}"

        # "Exists and is non-empty" is NOT proof of a complete download. A run
        # killed mid-transfer leaves a truncated file that passes that test, is
        # then skipped as already-present, and fails later as
        # `BadZipFile: File is not a zip file` -- a confusing error a long way
        # from its cause. The server's Content-Length is the check.
        expected = _remote_size(url)
        have = target.stat().st_size if target.exists() else 0
        if have and expected and have == expected:
            print(f"{filename}: already complete ({have / 1048576:.1f} MB)", flush=True)
        else:
            if have:
                print(
                    f"{filename}: {have / 1048576:.1f} MB on disk against "
                    f"{expected / 1048576:.1f} MB expected -- refetching",
                    flush=True,
                )
            print(f"{filename}: fetching", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "satquery/0"})
            with urllib.request.urlopen(request) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=8 << 20)
            got = target.stat().st_size
            if expected and got != expected:
                raise RuntimeError(
                    f"{filename}: downloaded {got} bytes, expected {expected}. A short "
                    "read written into the Volume becomes a corrupt archive that the "
                    "next run will happily skip."
                )
            print(f"{filename}: {got / 1048576:.1f} MB", flush=True)
            # A refetched archive invalidates whatever was unpacked from the old one.
            (root / f".{filename}.unzipped").unlink(missing_ok=True)

        if unzip:
            marker = root / f".{filename}.unzipped"
            if marker.exists():
                print(f"{filename}: already unzipped", flush=True)
            else:
                _extract_parallel(target, root)
                marker.write_text("done", encoding="utf-8")

    for command in spec["stage"] or []:
        import subprocess

        argv = [part.format(root=str(root)) for part in command]
        print("+ " + " ".join(argv), flush=True)
        subprocess.run(argv, cwd=REPO_DIR, check=True)

    # Volumes v1 want well under 50,000 files; a benchmark unzipped into one is
    # the realistic way to approach that, so the count is reported rather than
    # discovered later as a write failure.
    total = sum(1 for _ in root.rglob("*") if _.is_file())
    data_volume.commit()
    summary = f"{name}: {total} files under {root}"
    print(summary, flush=True)
    return summary


@app.local_entrypoint()
def stage(name: str = "rsvqa_lr") -> None:
    """Pull a benchmark straight into the Volume -- nothing lands locally.

    modal run scripts/modal_phase0.py::stage --name rsvqa_lr
    modal run scripts/modal_phase0.py::stage --name vrsbench_val
    """
    if name not in STAGEABLE:
        raise SystemExit(f"unknown dataset {name!r}; known: {sorted(STAGEABLE)}")
    print(stage_job.remote(name))


@app.local_entrypoint()
def train(
    val_every: int = 0,
    val_batches: int = -1,
    num_workers: int = 0,
    adapter: str = "rs_vqa",
    manifest: str = "/data/manifests/rs_vqa_train.jsonl",
    image_root: str = "/data",
    smoke_passed: str = "/data/checkpoints/rs_vqa_smoke/run.json",
    micro_batch: int = 8,
    grad_accum: int = 1,
    steps: int = 0,
    snapshot_every: int = 0,
    resume: bool = True,
) -> None:
    """A real training run, resumable across invocations.

    modal run scripts/modal_phase0.py::train --adapter rs_vqa
        --smoke-passed /data/checkpoints/rs_vqa_smoke/run.json

    **Re-running this command is how a run continues.** Every adapter is longer
    than a comfortable single invocation -- 4.0 h to 15.7 h measured -- and a
    container can be preempted or hit the 24 h ceiling. Lightning checkpoints
    every 500 steps onto the Volume, and this resumes from ``last.ckpt`` unless
    told otherwise, so the same command is both "start" and "continue".

    ``--no-resume`` forces a fresh run. Use it when the config changed:
    resuming onto a different corpus or batch size continues an optimizer state
    that belongs to a different experiment.

    ``micro_batch`` defaults to 8, which the saturation ramp measured as 92% of
    peak throughput at a third of the memory of the largest batch that fits.
    """
    argv = [
        "python",
        "training/train_lora.py",
        "--adapter", adapter,
        "--manifest", _volume_path(manifest),
        "--image-root", _volume_path(image_root),
        "--split", "train",
        "--out", f"/data/checkpoints/{adapter}",
        "--micro-batch", str(micro_batch),
        "--grad-accum", str(grad_accum),
    ]
    if steps:
        argv += ["--steps", str(steps)]
    if snapshot_every:
        argv += ["--snapshot-every", str(snapshot_every)]
    if val_every:
        argv += ["--val-every", str(val_every)]
    if num_workers:
        argv += ["--num-workers", str(num_workers)]
    if val_batches >= 0:
        argv += ["--val-batches", str(val_batches)]
    if smoke_passed:
        argv += ["--smoke-passed", smoke_passed]
    _save(
        train_job.remote(
            argv,
            ["logs/hour_burndown.jsonl"],
            f"/data/checkpoints/{adapter}",
            resume,
        )
    )


#: Fetching is network- and decode-bound, never GPU. It gets its own CPU
#: function with a long timeout: 20,000 windowed reads is hours of work, and
#: renting an A100 to decode JP2 tiles would be the most expensive way to do it.
@app.function(timeout=24 * HOURS, volumes=VOLUMES, cpu=8.0)
def fetch_job(argv: list[str]) -> str:
    import subprocess
    import sys as _sys

    print("+ " + " ".join(argv), flush=True)
    proc = subprocess.run(
        argv, cwd=REPO_DIR, stdout=_sys.stdout, stderr=_sys.stderr, check=False
    )
    data_volume.commit()
    # Report the job's OWN output directory, not a fixed one. Hardcoding
    # /data/chips made the RSVQA-HR fetch echo a stale summary from an
    # unrelated CDSE run -- a success that read like the wrong job.
    out_index = argv.index("--out") + 1 if "--out" in argv else None
    if out_index:
        for name in ("fetch_summary.json", "extract_summary.json"):
            summary = Path(argv[out_index]) / name
            if summary.exists():
                return summary.read_text(encoding="utf-8")
    return f"exit {proc.returncode}"


@app.local_entrypoint()
def fetch(
    manifest: str = "/data/manifests/ben_txt_train.jsonl",
    out: str = "/data/chips",
    workers: int = 16,
    limit: int = 0,
) -> None:
    """Pull the chosen patches from Copernicus into the Volume.

    Re-running is safe and is how an interrupted fetch continues: patches
    already written are refetched, so use --limit to bound a trial rather than
    to resume.

    Credentials come from Modal Secrets, never from the repo -- see
    satquery/credentials.py for why a credentials file must not live in a tree
    that gets uploaded.
    """
    argv = [
        "python", "scripts/fetch_copernicus_patches.py", "fetch",
        "--manifest", _volume_path(manifest),
        "--out", _volume_path(out),
        "--workers", str(workers),
    ]
    if limit:
        argv += ["--limit", str(limit)]
    print(fetch_job.remote(argv))


@app.function(timeout=10 * MINUTES)
def bandwidth_job(urls: list[str], seconds: int) -> dict:
    """Measure real download throughput from Modal to each source.

    A staging plan priced on a laptop's connection is priced on the wrong
    machine. Zenodo measured 3.67 MB/s from a home line -- 4.8 hours for the
    59 GB BigEarthNet archive -- and the only way to know whether Modal does
    better is to measure it from Modal.
    """
    import time
    import urllib.request

    results = {}
    for url in urls:
        request = urllib.request.Request(url, headers={"User-Agent": "satquery/0"})
        got = 0
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request) as response:
                while time.monotonic() - started < seconds:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    got += len(chunk)
        except Exception as error:  # noqa: BLE001 - a failed source is a result
            results[url] = {"error": repr(error)[:120]}
            continue
        elapsed = max(1e-6, time.monotonic() - started)
        mbps = got / elapsed / 1048576
        results[url] = {
            "mb_per_s": round(mbps, 2),
            "hours_for_59GiB": round(63251710377 / (mbps * 1048576) / 3600, 2),
        }
    return results


@app.local_entrypoint()
def bandwidth(seconds: int = 20) -> None:
    """Time the staging downloads from Modal rather than from a laptop."""
    urls = [
        "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1",
        "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip",
        "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN7_buildings/tarballs/"
        "SN7_buildings_train.tar.gz",
        "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN6_buildings/tarballs/"
        "SN6_buildings_AOI_11_Rotterdam_train.tar.gz",
        "https://rareplanes-public.s3-us-west-2.amazonaws.com/real/tarballs/train/"
        "RarePlanes_train_PS-RGB_tiled.tar.gz",
    ]
    for url, result in bandwidth_job.remote(urls, seconds).items():
        host = url.split("/")[2]
        print(f"{host:24} {result}")


#: Extraction is network-bound, never GPU. 24 h ceiling because the stream is
#: ~59 GB and a stall should not silently truncate the corpus.
@app.function(timeout=24 * HOURS, volumes=VOLUMES, cpu=4.0)
def extract_job(argv: list[str]) -> str:
    import subprocess
    import sys as _sys

    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, cwd=REPO_DIR, stdout=_sys.stdout, stderr=_sys.stderr, check=False)
    data_volume.commit()
    summary = Path(argv[argv.index("--out") + 1]) / "extract_summary.json"
    return summary.read_text(encoding="utf-8") if summary.exists() else "no summary written"


@app.local_entrypoint()
def extract(
    manifest: str = "/data/manifests/ben_txt_train.jsonl",
    out: str = "/data/chips",
    source: str = "hf",
) -> None:
    """Stream the BigEarthNet archive and keep only the chosen patches.

    Replaces the windowed Copernicus fetch for BEN: measured 26x faster, needs
    no credentials, and takes the patch tiles the dataset authors cut rather
    than reconstructing them. ``--source zenodo`` is the canonical mirror and
    about 5x slower.
    """
    print(
        extract_job.remote(
            [
                "python", "scripts/extract_ben_archive.py",
                "--manifest", _volume_path(manifest),
                "--out", _volume_path(out),
                "--source", source,
            ]
        )
    )


@app.local_entrypoint()
def validate(
    manifest: str = "/data/chips/canonical.jsonl",
    image_root: str = "/data/chips",
    sample_images: int = 400,
) -> None:
    """Run the corpus checks where the imagery actually lives.

    Locally the pixel check has to be skipped -- the chips are on the Volume --
    so "OK" from a laptop means "the JSON is consistent", not "the images are
    there and are not blank". This is the half that catches a manifest which
    has drifted from its imagery, which is the failure that produced 21
    all-black PNGs and a loss curve read off them.
    """
    _save(
        cpu_job.remote(
            [
                "python", "scripts/validate_corpus.py",
                "--manifest", _volume_path(manifest),
                "--image-root", _volume_path(image_root),
                "--sample-images", str(sample_images),
                "--json-out", "logs/corpus_validation.json",
            ],
            ["logs/corpus_validation.json"],
        )
    )


@app.local_entrypoint()
def sn7_plan(aois: int = 0, months: int = 0, out: str = "/data/eval/sn7") -> None:
    """Choose which SpaceNet 7 AOIs and months to stage.

    Defaults to everything: 60 AOIs x 25 monthly mosaics is ~6 GB from AWS Open
    Data at the 59-62 MB/s measured to this bucket, and taking all of it buys
    300 possible date pairs per AOI rather than a handful. Narrow with --aois
    or --months only when trialling.
    """
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_spacenet7.py", "plan",
                "--aois", str(aois),
                "--months", str(months),
                "--out", _volume_path(out),
            ]
        )
    )


@app.local_entrypoint()
def sn7_fetch(plan: str = "/data/eval/sn7/plan.json") -> None:
    """Pull the planned SpaceNet 7 files into the Volume, resumably.

    Re-running continues rather than restarting: the fetch skips whatever is
    already on disk, and writes through a .part file so an interrupted download
    cannot leave a truncated image that the resume check would then skip.
    """
    print(
        fetch_job.remote(
            ["python", "scripts/stage_spacenet7.py", "fetch", "--plan", _volume_path(plan)]
        )
    )


@app.function(timeout=2 * HOURS, volumes=VOLUMES, cpu=4.0)
def archive_job(paths: list[str], out: str) -> str:
    """Tar a set of Volume directories into one file on the same Volume.

    Modal Volumes do not cross accounts, and BEN's imagery is 60,063 files at
    33 KB each. Moving those individually means 60,063 round trips in each
    direction, where per-request overhead dwarfs the 1.9 GB of actual bytes.
    One archive is one download and one upload.
    """
    import subprocess
    from pathlib import Path as P

    target = P(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    argv = ["tar", "-cf", str(target), "-C", DATA_DIR, *paths]
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True)
    size = target.stat().st_size
    data_volume.commit()
    return f"{out} — {size / 1073741824:.2f} GB"


@app.function(timeout=2 * HOURS, volumes=VOLUMES, cpu=4.0)
def extract_archive_job(archive: str) -> str:
    """Unpack an archive that was uploaded to this account's Volume."""
    import subprocess
    from pathlib import Path as P

    source = P(archive)
    if not source.exists():
        raise SystemExit(f"{archive} is not on this Volume; upload it first")
    argv = ["tar", "-xf", str(source), "-C", DATA_DIR]
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True)
    data_volume.commit()
    return f"extracted {archive}"


@app.function(timeout=30 * MINUTES, volumes=VOLUMES)
def keep_job(source: str, target: str) -> str:
    """Copy a file within the Volume, so a rolling checkpoint survives.

    ``save_top_k=1`` keeps only the most recent step checkpoint, which means
    every 500 steps the previous one is deleted. That is the right default for
    disk -- each file is 3.81 GB because Lightning stores the frozen 4.5B base
    alongside the 40M LoRA weights -- but it also means a checkpoint worth
    keeping has to be copied out of the rotation before the next one lands.
    """
    import shutil
    from pathlib import Path as P

    data_volume.reload()
    src = P(source)
    if not src.exists():
        raise SystemExit(f"{source} is not on the Volume (already rotated out?)")
    dst = P(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    data_volume.commit()
    return f"kept {source} -> {target} ({dst.stat().st_size / 1073741824:.2f} GB)"


@app.function(gpu=GPU, timeout=1 * HOURS, volumes=VOLUMES)
def ckpt_to_adapter_job(checkpoint: str, out_dir: str, adapter: str) -> str:
    """Turn a Lightning checkpoint into a PEFT adapter directory.

    The trainer writes ``adapter/`` only when a run finishes, but the
    checkpoints worth comparing are mid-run: `best.ckpt` at the lowest
    validation loss and whichever step was preserved before rotation. Those are
    Lightning state dicts, which the evaluator cannot load. This rebuilds the
    LoRA model, loads the checkpoint's weights into it, and saves the adapter
    the normal way -- so a mid-run checkpoint becomes evaluable without waiting
    for the run to end.
    """
    import sys as _sys
    from pathlib import Path as P

    import torch

    _sys.path.insert(0, REPO_DIR)
    from satquery.training.config import TrainingConfig
    from satquery.training.lora import build_lora_model

    data_volume.reload()
    source = P(checkpoint)
    if not source.exists():
        raise SystemExit(f"{checkpoint} is not on the Volume")

    cfg = TrainingConfig.for_adapter(adapter)
    bundle = build_lora_model(cfg)

    state = torch.load(source, map_location="cpu", weights_only=False)
    weights = state.get("state_dict", state)
    # Lightning prefixes every key with the attribute the module was stored
    # under; the PEFT model underneath expects them without it.
    stripped = {
        key[len("model.") :] if key.startswith("model.") else key: value
        for key, value in weights.items()
    }
    missing, unexpected = bundle.model.load_state_dict(stripped, strict=False)
    trained = [k for k in stripped if "lora_" in k]
    if not trained:
        raise SystemExit(
            "no lora_ keys in the checkpoint. Nothing would be evaluated but "
            "the frozen base, which is not the experiment."
        )

    target = P(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(str(target))
    data_volume.commit()
    return (
        f"{checkpoint} -> {out_dir}  "
        f"({len(trained)} LoRA tensors, {len(missing)} missing, "
        f"{len(unexpected)} unexpected)"
    )


@app.local_entrypoint()
def eval_pipeline(
    benchmark: str = "/data/eval/change_vqa_val.jsonl",
    sn7_root: str = "/data/eval/sn7",
) -> None:
    """Score the measurement pipeline on the adapter's own benchmark.

    Same 2,012 rows, same answer contracts, so the number is directly
    comparable to the adapter's AA 43.5%. Runs on CPU: routing is regex and
    change_stats is set arithmetic, which is the point -- the questions the
    VLM spends a GPU guessing at are a subtraction.
    """
    _save(
        cpu_job.remote(
            [
                "python", "scripts/eval_pipeline.py",
                "--benchmark", _volume_path(benchmark),
                "--sn7-root", _volume_path(sn7_root),
                "--out", "logs/pipeline_eval.json",
            ],
            ["logs/pipeline_eval.json"],
        )
    )


@app.local_entrypoint()
def cd_train(
    steps: int = 4000,
    root: str = "/data/eval/sn7",
    batch_size: int = 16,
    crop: int = 256,
    crops_per_pair: int = 2,
    num_workers: int = 8,
    max_gap_months: int = 12,
    val_every: int = 250,
    lr: float = 3e-4,
    task: str = "change",
    upscale: int = 3,
) -> None:
    """Train the change-detection model for real, reporting F1 as it goes.

    ``cd_best.pt`` tracks the best held-out F1 and the report is rewritten at
    every validation, so a crash costs one interval rather than the run --
    both lessons from the change_vqa run, where the best checkpoint was step
    1,000 of 10,976 and the rolling one had long since been overwritten.
    """
    _save(
        gpu_job.remote(
            [
                "python", "training/train_cd.py",
                "--root", _volume_path(root),
                "--out", _volume_path("/data/checkpoints/change_map"),
                "--steps", str(steps),
                "--batch-size", str(batch_size),
                "--crop", str(crop),
                "--crops-per-pair", str(crops_per_pair),
                "--num-workers", str(num_workers),
                "--max-gap-months", str(max_gap_months),
                "--val-every", str(val_every),
                "--lr", str(lr),
                "--task", task,
                "--upscale", str(upscale),
                "--report", f"logs/cd_train_{task}.json",
            ],
            [f"logs/cd_train_{task}.json"],
        )
    )


@app.local_entrypoint()
def cd_smoke(
    root: str = "/data/eval/sn7",
    batch_size: int = 16,
    crop: int = 256,
    num_workers: int = 8,
    max_gap_months: int = 0,
) -> None:
    """200 steps of the Siamese change-detection model, for a real timing.

    The deterministic route for gate G4. The VQA adapter gets 1.2 points of its
    counting accuracy from actually reading the images; a change mask makes
    presence, direction, magnitude and location arithmetic instead of
    inference. This measures s/step and the F1 trajectory before anything long
    is committed -- the same guard rail that caught the VLM run's real cost.
    """
    _save(
        gpu_job.remote(
            [
                "python", "training/train_cd.py",
                "--root", _volume_path(root),
                "--out", _volume_path("/data/checkpoints/change_map"),
                "--batch-size", str(batch_size),
                "--crop", str(crop),
                "--num-workers", str(num_workers),
                "--max-gap-months", str(max_gap_months),
                "--val-every", "100",
                "--smoke",
                "--report", "logs/cd_smoke.json",
            ],
            ["logs/cd_smoke.json"],
        )
    )


@app.local_entrypoint()
def eval_split(
    manifest: str = "/data/manifests/change_vqa_train.jsonl",
    split: str = "val",
    out: str = "/data/eval/change_vqa_val.jsonl",
    per_type: int = 400,
) -> None:
    """Cut a scoreable benchmark file from a manifest's held-out split."""
    _save(
        cpu_job.remote(
            [
                "python", "scripts/make_eval_split.py",
                "--manifest", _volume_path(manifest),
                "--split", split,
                "--out", _volume_path(out),
                "--per-type", str(per_type),
            ],
            ["logs/eval_split_summary.json"],
        )
    )


@app.local_entrypoint()
def ckpt_to_adapter(
    checkpoint: str, out_dir: str, adapter: str = "change_vqa"
) -> None:
    """Make a mid-run Lightning checkpoint loadable by the evaluator."""
    print(ckpt_to_adapter_job.remote(
        _volume_path(checkpoint), _volume_path(out_dir), adapter
    ))


@app.local_entrypoint()
def keep(source: str, target: str) -> None:
    """Preserve a rolling checkpoint before it is overwritten."""
    print(keep_job.remote(_volume_path(source), _volume_path(target)))


@app.local_entrypoint()
def archive(paths: str = "chips/composites,manifests", out: str = "/data/transfer/ben.tar") -> None:
    """Pack Volume directories into one file, for moving between accounts.

    Run this on the account that HAS the data. Then `modal volume get` the
    archive, `modal volume put` it to the other account, and run `unarchive`
    there.
    """
    print(archive_job.remote([p.strip() for p in paths.split(",")], _volume_path(out)))


@app.local_entrypoint()
def unarchive(archive: str = "/data/transfer/ben.tar") -> None:
    """Unpack an archive uploaded to this account's Volume."""
    print(extract_archive_job.remote(_volume_path(archive)))


@app.local_entrypoint()
def gen_ground(
    rareplanes: str = "/data/eval/rareplanes",
    spacenet6: str = "",
    out: str = "/data/manifests/rs_ground_caption_train.jsonl",
    cap: int = 4000,
    limit: int = 0,
) -> None:
    """Phrase the staged object annotations as rs_ground_caption questions.

    Answers here are coordinates, so the blind ceiling that governs change_vqa
    does not apply -- a box cannot be guessed from the question text. What can
    go wrong instead is scale: BEN's regions cover a fifth of the image and
    SpaceNet 6's buildings a fifth of a percent, and a model that learns the
    first will fail the second. The summary reports box area per source for
    exactly that reason.
    """
    argv = [
        "python", "scripts/gen_ground.py",
        "--out", _volume_path(out),
        "--image-root", _volume_path("/data"),
        "--cap", str(cap),
        "--limit", str(limit),
        "--summary-out", "logs/rs_ground_caption_summary.json",
    ]
    if rareplanes:
        argv += ["--rareplanes", _volume_path(rareplanes)]
    if spacenet6:
        argv += ["--spacenet6", _volume_path(spacenet6)]
    _save(cpu_job.remote(argv, ["logs/rs_ground_caption_summary.json"]))


@app.local_entrypoint()
def sn6_plan(
    modalities: str = "PS-RGB", tiles: int = 0, out: str = "/data/eval/sn6"
) -> None:
    """Choose SpaceNet 6 tiles and which modalities to stage.

    ``PS-RGB`` alone (7.9 GB with labels) is the grounding half for G3; adding
    ``SAR-Intensity`` (41 GB) makes it the cross-modal source for G5. Volumes do
    not cross accounts, so which modalities you want depends on which account
    you are on.
    """
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_spacenet6.py", "plan",
                "--modalities", modalities,
                "--tiles", str(tiles),
                "--out", _volume_path(out),
            ]
        )
    )


@app.local_entrypoint()
def sn6_fetch(plan: str = "/data/eval/sn6/plan.json") -> None:
    """Pull the planned SpaceNet 6 files into the Volume, resumably."""
    print(
        fetch_job.remote(
            ["python", "scripts/stage_spacenet6.py", "fetch",
             "--plan", _volume_path(plan)]
        )
    )


@app.local_entrypoint()
def sn2_plan(
    modalities: str = "PS-RGB", tiles: int = 6000, out: str = "/data/eval/sn2"
) -> None:
    """Choose SpaceNet 2 images, drawn evenly across its four AOIs.

    SpaceNet 6 ran out: 3,401 Rotterdam tiles produced 2,627 and 2,459 rows
    against a 4,000 cap, so the building half of G3 cannot grow on it. This is
    0.3 m over Vegas, Paris, Shanghai and Khartoum -- four building
    vernaculars instead of one, which matters more than volume for a hidden set
    over India that resembles none of them.
    """
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_spacenet2.py", "plan",
                "--modalities", modalities,
                "--tiles", str(tiles),
                "--out", _volume_path(out),
            ]
        )
    )


@app.local_entrypoint()
def gen_captions(
    out: str = "/data/manifests/rs_ground_caption_captions.jsonl",
) -> None:
    """Build the captioning corpus from every staged source.

    Writes into ``/data/manifests`` so the ``browse`` endpoint discovers it
    automatically -- the whole corpus is then readable and filterable before a
    GPU is touched, which is how the grounding corpus caught a 273-row
    degenerate-box bug that a sampled review would have missed.
    """
    _save(
        corpus_job.remote(
            [
                "python", "scripts/gen_captions.py", "build",
                "--rareplanes", _volume_path("/data/eval/rareplanes"),
                "--sn2", _volume_path("/data/eval/sn2"),
                "--sn6", _volume_path("/data/eval/sn6"),
                "--rsvqa-hr", _volume_path("/data/eval/rsvqa_hr"),
                "--ben", _volume_path("/data/manifests/ben_txt_train.jsonl"),
                "--out", _volume_path(out),
                "--summary", "logs/caption_corpus.json",
            ],
            ["logs/caption_corpus.json"],
        )
    )


@app.local_entrypoint()
def captioning(
    samples: int = 300,
    style: str = "styled",
    adapter: str = "",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    tag: str = "base",
    dump: int = 0,
) -> None:
    """Score captioning on VRSBench, with the blind floor beside it.

    The half no detector reaches. Grounding is settled without training; this
    measures whether captioning is too, and reports what describing the *wrong*
    image earns so a mediocre score cannot read as a good one.
    """
    _save(
        gpu_job.remote(
            [
                "python", "scripts/eval_captioning.py",
                "--benchmark",
                _volume_path("/data/eval/vrsbench_val/VRSBench_EVAL_Cap.json"),
                "--images", _volume_path("/data/eval/vrsbench_val/Images_val"),
                "--model", model,
                *(["--adapter", _volume_path(adapter)] if adapter else []),
                "--samples", str(samples),
                "--style", style,
                "--dump", str(dump),
                *(["--dump-dir", _volume_path(f"/data/eval/cap_dump_{tag}")] if dump else []),
                "--out", f"logs/caption_{tag}.json",
            ],
            [f"logs/caption_{tag}.json"],
        )
    )


@app.function(timeout=4 * HOURS, volumes=VOLUMES)
def gdrive_job(file_ids: list[str], dest: str) -> str:
    """Pull an archive published only as a Google Drive share, and unpack it.

    Two ids because the authors publish two mirrors; the first that yields a
    readable archive wins. Drive serves large files behind an interstitial that
    a plain GET returns as HTML, so this uses gdown rather than urlretrieve --
    the failure mode otherwise is a few-kilobyte "file" that unpacks to nothing.
    """
    import subprocess
    import zipfile

    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    for file_id in file_ids:
        archive = target / f"{file_id}.zip"
        if not archive.exists():
            print(f"gdown {file_id}", flush=True)
            result = subprocess.run(
                # gdown 5.x removed --id; the identifier is positional now.
                ["gdown", file_id, "-O", str(archive), "--no-cookies"],
                capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                print(f"  failed: {result.stderr[-400:]}", flush=True)
                continue
        size = archive.stat().st_size
        print(f"  {archive.name}: {size / 2**20:.1f} MiB", flush=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
        except zipfile.BadZipFile:
            print("  not a zip -- Drive likely served an interstitial", flush=True)
            archive.unlink(missing_ok=True)
            continue
        archive.unlink(missing_ok=True)
        data_volume.commit()
        listing = sorted(p.name for p in target.iterdir())[:15]
        return f"extracted into {target}: {listing}"
    data_volume.commit()
    return f"FAILED: no mirror yielded a readable archive under {target}"


@app.local_entrypoint()
def second_fetch(dest: str = "/data/eval/second") -> None:
    """Stage the imagery CDVQA references.

    Published only as Google Drive shares from the authors' project page, with
    no licence terms of any kind. That is recorded in CREDITS rather than
    guessed at here.
    """
    print(
        gdrive_job.remote(
            [
                "1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u",
                "1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc",
            ],
            _volume_path(dest),
        )
    )


@app.function(timeout=2 * HOURS, volumes=VOLUMES)
def unpack_job(root: str) -> str:
    """Unpack nested archives in place, then report what the pair index will see.

    The Drive archive holds a nested ``.rar`` alongside a nested ``.zip``, and
    both must be unpacked. Reporting the ``im1``/``im2`` counts here rather
    than leaving it to the staging run means a half-extracted tree is visible
    now, not as a mystery "imagery missing" three commands later.
    """
    import subprocess

    target = Path(root)
    for _ in range(3):  # nested archives can be more than one deep
        archives = [
            p for p in target.rglob("*")
            if p.suffix.lower() in (".zip", ".rar", ".7z") and "__MACOSX" not in str(p)
        ]
        if not archives:
            break
        for archive in archives:
            # unar for RAR (p7zip silently mangles RAR5 -- it reported 9,871
            # sub-item errors and still left files on disk, which read as
            # present-but-undecodable and only surfaced at DataLoader time).
            if archive.suffix.lower() == ".rar":
                command = ["unar", "-force-overwrite", "-o", str(archive.parent), str(archive)]
            else:
                command = ["7z", "x", "-y", f"-o{archive.parent}", str(archive)]
            print(f"{command[0]} {archive.name}", flush=True)
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"  failed: {result.stdout[-300:]}{result.stderr[-300:]}", flush=True)
                continue
            archive.unlink(missing_ok=True)

    im1 = sum(1 for p in target.rglob("*.png") if p.parent.name.lower() == "im1")
    im2 = sum(1 for p in target.rglob("*.png") if p.parent.name.lower() == "im2")
    dirs = sorted({p.parent.name for p in target.rglob("*.png")})[:12]
    data_volume.commit()
    return f"im1={im1} im2={im2}; image directories seen: {dirs}"


@app.function(timeout=2 * HOURS, volumes=VOLUMES)
def verify_images_job(root: str, out: str) -> str:
    """Open every PNG and record the ones that will not decode.

    The archive's .rar leg failed with 9,871 sub-item errors, and a truncated
    PNG still occupies a directory entry -- so a file count says nothing about
    whether training can read it. A 40-pair sample missed this; the DataLoader
    found it at step zero instead. This checks all of them, once, so the
    manifest can exclude what is unreadable rather than discovering it per-epoch.
    """
    from PIL import Image

    target = Path(root)
    bad, total = [], 0
    for path in sorted(target.rglob("*.png")):
        if path.parent.name.lower() not in ("im1", "im2"):
            continue
        total += 1
        try:
            with Image.open(path) as handle:
                handle.verify()
        except Exception:  # noqa: BLE001 - any decode failure disqualifies it
            bad.append(str(path.relative_to(target)))

    report = Path(out)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"checked": total, "unreadable": len(bad), "files": bad}, indent=2),
        encoding="utf-8",
    )
    data_volume.commit()
    pct = 100.0 * len(bad) / total if total else 0.0
    return f"checked {total}, unreadable {len(bad)} ({pct:.1f}%) -> {report}"


@app.local_entrypoint()
def second_verify(
    root: str = "/data/eval/second",
    out: str = "/data/eval/second/unreadable.json",
) -> None:
    """Find every staged CDVQA image that will not decode."""
    print(verify_images_job.remote(_volume_path(root), _volume_path(out)))


@app.local_entrypoint()
def second_unpack(root: str = "/data/eval/second") -> None:
    """Unpack the nested imagery archives and report the im1/im2 counts."""
    print(unpack_job.remote(_volume_path(root)))


@app.local_entrypoint()
def cdvqa_stage(
    second: str = "/data/eval/second",
    split: str = "Train",
    out: str = "/data/manifests/change_vqa_cdvqa_train.jsonl",
    limit: int = 0,
    cap_per_type: int = 0,
) -> None:
    """Build change_vqa training rows from CDVQA's published annotations.

    CDVQA is Apache-2.0 and the QA is fetched from the authors' repository; the
    imagery must already be on the Volume under ``im1/`` and ``im2/``. Writes a
    separate manifest, so CDVQA's Test/Test2 splits stay eval-only and untouched.
    """
    _save(
        corpus_job.remote(
            [
                "python", "scripts/stage_cdvqa.py", "build",
                "--second", _volume_path(second),
                "--split", split,
                "--cache", _volume_path("/data/eval/cdvqa_qa"),
                "--out", _volume_path(out),
                "--limit", str(limit),
                "--cap-per-type", str(cap_per_type),
                "--summary", "logs/cdvqa_corpus.json",
            ],
            ["logs/cdvqa_corpus.json"],
        )
    )


@app.local_entrypoint()
def bifold(
    mode: str = "probe",
    arm: str = "sar",
    model: str = "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0",
    samples: int = 2000,
    s1_order: str = "vh_vv",
    s2_order: str = "configilm",
    tag: str = "",
) -> None:
    """Score a reBEN reference classifier on our own cross-modal questions.

    ``probe`` first: our staged rasters are a repack, so which band sits in
    which position is undocumented, and the model returns confident nonsense
    rather than an error when it is wrong. Probe resolves that on **train**
    rows so the val number stays a measurement.
    """
    name = tag or f"{mode}_{arm}"
    if mode == "duet":
        argv = [
            "python", "scripts/eval_bifold.py", "duet",
            "--corpus", _volume_path("/data/manifests/optsar_fusion.jsonl"),
            "--samples", str(samples),
            "--s1-order", s1_order,
            "--s2-order", s2_order,
            "--out", f"logs/bifold_{name}.json",
        ]
    else:
        argv = [
            "python", "scripts/eval_bifold.py", mode,
            "--corpus", _volume_path("/data/manifests/optsar_fusion.jsonl"),
            "--model", model,
            "--arm", arm,
            "--samples", str(samples),
            "--out", f"logs/bifold_{name}.json",
        ]
        if mode == "score":
            argv += ["--s1-order", s1_order, "--s2-order", s2_order]
    _save(gpu_job.remote(argv, [f"logs/bifold_{name}.json"]))


@app.local_entrypoint()
def gen_optsar(
    reben: str = "/data/eval/reben/patches/BEN_14k",
    out: str = "/data/manifests/optsar_fusion.jsonl",
    limit: int = 0,
) -> None:
    """Build the cross-modal corpus: the first caller gen_crossmodal_qa has had.

    Emits each question on the optical arm, the SAR arm and the fused arm -- but
    only where radar can actually answer it. Water and built-up have distinct
    scattering behaviour; arable land against pastures does not, and a SAR row
    asking that teaches the model to answer from the label prior instead of the
    image. Which classes earn radar arms is stated in ``SAR_ANSWERABLE``.
    """
    _save(
        corpus_job.remote(
            [
                "python", "scripts/gen_optsar.py", "build",
                "--reben", _volume_path(reben),
                "--out", _volume_path(out),
                "--limit", str(limit),
                "--summary", "logs/optsar_corpus.json",
            ],
            ["logs/optsar_corpus.json"],
        )
    )


@app.local_entrypoint()
def reben_inspect(out: str = "/data/eval/reben") -> None:
    """Report the paired-subset archive's layout without extracting it.

    The archive is a third-party upload with no documented structure. Writing an
    extractor against a guessed layout is how a staging run gets spent learning
    what it should have looked at first.
    """
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_reben_probe.py", "inspect",
                "--out", _volume_path(out),
            ]
        )
    )


@app.local_entrypoint()
def reben_verify(out: str = "/data/eval/reben", sample: int = 40) -> None:
    """Open the staged rasters and check values, bands and units.

    The subset is a *repack* -- one stacked GeoTIFF per patch where reBEN ships
    a directory of per-band files -- so byte-identity with the original cannot
    be established even in principle. What this establishes is whether the
    repack preserved the information: band count, dtype, geometry and value
    range. A subset re-quantised to 8-bit passes every check that reads only
    file names, and would then silently break every decibel threshold in D1.
    """
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_reben_probe.py", "verify",
                "--out", _volume_path(out),
                "--sample", str(sample),
            ]
        )
    )


@app.local_entrypoint()
def reben_stage(out: str = "/data/eval/reben") -> None:
    """Stage the paired S1/S2 subset, reconciled against the official index.

    Every patch identifier is resolved against reBEN's own ``metadata.parquet``,
    taken from a different mirror than the subset itself -- verifying a
    stranger's archive against that same stranger's metadata would verify
    nothing. Patches the authoritative index does not carry are refused rather
    than staged, because a corrupted corpus is invisible in every metric we have.
    """
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_reben_probe.py", "stage",
                "--out", _volume_path(out),
            ]
        )
    )


@app.local_entrypoint()
def florence(
    samples: int = 300,
    task: str = "<MORE_DETAILED_CAPTION>",
    model: str = "microsoft/Florence-2-large",
    tag: str = "florence",
    dump: int = 0,
) -> None:
    """Florence-2 captioning, for comparison against the prompted Qwen numbers.

    Florence takes a task token, not a sentence, so the mined VRSBench register
    cannot be given to it. The score therefore mixes sight with register, and
    ``task=all`` sweeps the three caption tokens because they differ mainly in
    output length -- which is what the brevity penalty grades.
    """
    _save(
        florence_job.remote(
            [
                "python", "scripts/eval_florence.py",
                "--benchmark",
                _volume_path("/data/eval/vrsbench_val/VRSBench_EVAL_Cap.json"),
                "--images", _volume_path("/data/eval/vrsbench_val/Images_val"),
                "--model", model,
                "--samples", str(samples),
                "--task", task,
                "--dump", str(dump),
                *(["--dump-dir", _volume_path(f"/data/eval/cap_dump_{tag}")] if dump else []),
                "--out", f"logs/caption_{tag}.json",
            ],
            [f"logs/caption_{tag}.json"],
        )
    )


@app.local_entrypoint()
def pipeline_bakeoff(
    pipeline: str = "qwen",
    samples: int = 200,
    max_marks: int = 12,
    tag: str = "",
) -> None:
    """One grounding pipeline, scored on the shared fixed row sample.

    Every pipeline reads the same ``pipeline_sample.json``, written once on the
    first run. Paired rows are the point: at 200 rows an unpaired five-point
    difference is noise, while the same five points on identical questions is a
    result. Earlier runs each drew their own sample, so the detector and the VLM
    were never actually compared on the same questions.
    """
    name = tag or pipeline
    _save(
        gpu_job.remote(
            [
                "python", "scripts/eval_grounding_pipelines.py",
                "--benchmark",
                _volume_path("/data/eval/vrsbench_val/VRSBench_EVAL_referring.json"),
                "--images", _volume_path("/data/eval/vrsbench_val/Images_val"),
                "--pipeline", pipeline,
                "--samples", str(samples),
                "--max-marks", str(max_marks),
                "--sample-file", _volume_path("/data/eval/pipeline_sample.json"),
                "--dump-dir", _volume_path(f"/data/eval/pipe_dump_{name}"),
                "--tag", name,
            ],
            [f"logs/pipeline_{name}.json"],
        )
    )


@app.local_entrypoint()
def vlm_grounding(
    samples: int = 600,
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    adapter: str = "",
    dump: int = 12,
    tag: str = "base",
    sampling: str = "random",
) -> None:
    """Score the base VLM on VRSBench referring with no fine-tuning.

    The control the G3 argument has been missing: "train our corpus" and "bolt
    on Grounding DINO" have only ever been compared to each other, never to the
    checkpoint we already load. Qwen-VL carries grounding in its pretraining, so
    this may be a shorter path than either.
    """
    _save(
        gpu_job.remote(
            [
                "python", "scripts/eval_vlm_grounding.py",
                "--benchmark",
                _volume_path("/data/eval/vrsbench_val/VRSBench_EVAL_referring.json"),
                "--images", _volume_path("/data/eval/vrsbench_val/Images_val"),
                "--model", model,
                *(["--adapter", _volume_path(adapter)] if adapter else []),
                "--samples", str(samples),
                "--sampling", sampling,
                "--dump", str(dump),
                *(["--dump-dir", _volume_path(f"/data/eval/vlm_dump_{tag}")] if dump else []),
                "--out", f"logs/vlm_grounding_{tag}.json",
            ],
            [f"logs/vlm_grounding_{tag}.json"],
        )
    )


@app.local_entrypoint()
def grounding_recall(
    samples: int = 600,
    topk: int = 10,
    tiles: int = 1,
    box_threshold: float = 0.15,
    no_synonyms: bool = False,
    model: str = "IDEA-Research/grounding-dino-base",
    tag: str = "base",
    dump: int = 0,
) -> None:
    """Ceiling test for the detector-proposes / selector-chooses G3 design.

    Measures **recall**, not accuracy: how often the correct VRSBench box is
    among Grounding DINO's top-N candidates. The selection stage that would
    follow can only lose boxes, so this is the design's upper bound. Needs no
    VLM, no training and no selection logic, so nothing can confound it.

    Grounding DINO is Apache-2.0 and reaches us through ``transformers``, which
    needs no custom CUDA extension -- the dependency-rot risk that made this
    look expensive does not exist on this route.
    """
    _save(
        gpu_job.remote(
            [
                "python", "scripts/eval_grounding_recall.py",
                "--benchmark",
                _volume_path("/data/eval/vrsbench_val/VRSBench_EVAL_referring.json"),
                "--images", _volume_path("/data/eval/vrsbench_val/Images_val"),
                "--model", model,
                "--samples", str(samples),
                "--topk", str(topk),
                "--tiles", str(tiles),
                "--box-threshold", str(box_threshold),
                *(["--no-synonyms"] if no_synonyms else []),
                *(["--dump-dir", _volume_path(f"/data/eval/gdino_dump_{tag}"),
                   "--dump-count", str(dump)] if dump else []),
                "--out", f"logs/grounding_recall_{tag}.json",
            ],
            [f"logs/grounding_recall_{tag}.json"],
        )
    )


@app.local_entrypoint()
def g3_fetch_all(
    rareplanes: str = "/data/eval/rareplanes/plan.json",
    sn2: str = "/data/eval/sn2/plan.json",
) -> None:
    """Fetch every remaining G3 source in ONE container, sequentially.

    **Two ``modal run`` invocations of this app cannot coexist.** Both
    entrypoints share one ``app`` object, so the second run displaces the first
    -- observed as "Webhook label stolen: a newer app acquired ..." and then as
    a silently stopped fetch. Running both stagers inside a single job is the
    fix, and it also removes the ``.part`` race: two processes fetching the same
    key write the same temporary name, and the truncated result still satisfies
    the ``exists()`` resume check, so it would never be re-fetched.

    Sequential rather than parallel on purpose. These are S3-bound, one
    container already saturates the useful bandwidth, and a shared failure mode
    is easier to read in one log than two.

    Launch with ``--detach`` so the fetch survives the local machine going away:
    without it the app stops when the client does, which cost us a SpaceNet 6
    run tonight.
    """
    print(
        fetch_job.remote(
            [
                "bash", "-lc",
                f"python scripts/stage_rareplanes.py fetch --plan {_volume_path(rareplanes)} "
                f"&& python scripts/stage_spacenet2.py fetch --plan {_volume_path(sn2)}",
            ]
        )
    )


@app.local_entrypoint()
def sn2_fetch(plan: str = "/data/eval/sn2/plan.json") -> None:
    """Pull the planned SpaceNet 2 images into the Volume, resumably."""
    print(
        fetch_job.remote(
            ["python", "scripts/stage_spacenet2.py", "fetch",
             "--plan", _volume_path(plan)]
        )
    )


@app.local_entrypoint()
def rareplanes_plan(
    out: str = "/data/eval/rareplanes", tiles: int = 0, negative_ratio: float = 0.5
) -> None:
    """Choose which RarePlanes tiles to stage, and pull the COCO boxes.

    The object half of `rs_ground_caption`. BEN.txt supplies land-cover regions
    -- median box 19.8% of the image, none under 1% -- so a model trained on it
    alone learns to draw large boxes. RarePlanes supplies 30 cm aircraft, which
    is both the small-object case and the closest GSD match to Cartosat-2S in
    the inventory.
    """
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_rareplanes.py", "plan",
                "--out", _volume_path(out),
                "--tiles", str(tiles),
                "--negative-ratio", str(negative_ratio),
            ]
        )
    )


@app.local_entrypoint()
def rareplanes_fetch(plan: str = "/data/eval/rareplanes/plan.json") -> None:
    """Pull the planned RarePlanes tiles into the Volume, resumably."""
    print(
        fetch_job.remote(
            ["python", "scripts/stage_rareplanes.py", "fetch",
             "--plan", _volume_path(plan)]
        )
    )


@app.local_entrypoint()
def india_plan(out: str = "/data/eval/india") -> None:
    """Choose the Indian AOIs and dates for the self-generated Sentinel pairs.

    24 boxes across 9 land-cover strata -- irrigated and rain-fed agriculture,
    forest, coastal, arid, water, mountain, northeast and urban -- each with a
    season fixed *within* its pair and varied *across* AOIs, so the corpus does
    not learn that "change" means "monsoon". 21 survive the requirement that a
    single Sentinel-2 granule fully contain the box; the rest are reported and
    skipped, because a scene that merely intersects the box came back 87%
    nodata the first time this ran.
    """
    print(
        fetch_job.remote(
            ["python", "scripts/stage_india_sentinel.py", "plan",
             "--out", _volume_path(out)]
        )
    )


@app.local_entrypoint()
def india_fetch(plan: str = "/data/eval/india/plan.json") -> None:
    """Pull the planned Indian tiles from the public Sentinel-2 COGs.

    No credentials: this reads windowed COGs from the ``sentinel-cogs`` open
    bucket via the Earth Search STAC API. Each AOI is tiled rather than taken
    whole, and each tile is written as six composites -- true-colour,
    false-colour and short-wave at each of two dates -- alongside the NDVI,
    NDWI and NDBI deltas that become its labels.
    """
    print(
        fetch_job.remote(
            ["python", "scripts/stage_india_sentinel.py", "fetch",
             "--plan", _volume_path(plan)]
        )
    )


@app.local_entrypoint()
def gen_india(
    root: str = "/data/eval/india",
    out: str = "/data/manifests/change_vqa_india.jsonl",
    cap: int = 400,
    limit: int = 0,
) -> None:
    """Phrase the Indian tile pairs as change_vqa questions.

    Weaker labels than SpaceNet 7 by construction -- thresholded index deltas,
    not annotations -- so this is the domain-match and land-cover supplement,
    never the source of record. Deltas inside the dead band are skipped rather
    than rounded to "unchanged", which costs rows and buys labels that mean
    something.
    """
    _save(
        cpu_job.remote(
            [
                "python", "scripts/gen_india.py",
                "--root", _volume_path(root),
                "--out", _volume_path(out),
                "--image-root", _volume_path("/data"),
                "--cap", str(cap),
                "--limit", str(limit),
                "--summary-out", "logs/change_vqa_india_summary.json",
            ],
            ["logs/change_vqa_india_summary.json"],
        )
    )


@app.local_entrypoint()
def hr_plan(
    split: str = "test_phili",
    images: int = 600,
    max_per_image: int = 6,
) -> None:
    """Choose which RSVQA-HR images to stage for a split.

    ``test_phili`` is the dataset's own domain-shift holdout: Philadelphia,
    unseen during training, and captured by a sensor that is also unseen. Same
    question generator and answer format as test set 1, so a score difference
    between the two isolates the domain shift from everything else -- which is
    the closest public rehearsal of the Cartosat problem available.
    """
    root = "/data/eval/rsvqa_hr"
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_rsvqa_hr.py", "plan",
                "--root", root,
                "--split", split,
                "--images", str(images),
                "--max-per-image", str(max_per_image),
                # Explicit, because the default is root/plan.json and that file
                # is the training split's plan -- writing over it would silently
                # destroy the record of what the adapter was trained on.
                "--out", f"{root}/plan_{split}.json",
            ]
        )
    )


@app.local_entrypoint()
def hr_fetch(
    plan: str = "/data/eval/rsvqa_hr/plan.json",
    out: str = "/data/eval/rsvqa_hr",
) -> None:
    """Stream RSVQA-HR's Images.tar into the Volume, unattended.

    Run on a laptop this died overnight when the machine slept, having got
    1,160 of 3,000 images. The tar is sequential and Zenodo is slow and erratic
    (0.95-5.28 MB/s measured), so the job is an hour or more of uninterrupted
    streaming -- which is exactly the kind of thing that should not depend on a
    laptop staying awake.
    """
    print(
        fetch_job.remote(
            [
                "python", "scripts/stage_rsvqa_hr.py", "fetch",
                "--plan", _volume_path(plan),
                "--out", _volume_path(out),
            ]
        )
    )


@app.local_entrypoint()
def bucket_area(
    manifests: str = (
        "/data/eval/rsvqa_hr/rsvqa_hr_train.jsonl,"
        "/data/eval/rsvqa_hr/rsvqa_hr_test.jsonl"
    ),
    question_type: str = "area",
) -> None:
    """Quantise RSVQA area answers into the paper's five classes, in place.

    Train and test together and never one alone: a model trained on exact
    values and scored on classes is being measured on a format it was never
    shown, and the reverse silently flatters it.
    """
    for manifest in filter(None, (m.strip() for m in manifests.split(","))):
        print(
            cpu_job.remote(
                [
                    "python", "scripts/bucket_rsvqa_area.py",
                    "--manifest", _volume_path(manifest),
                    "--question-type", question_type,
                ],
                ["logs/bucket_area.json"],
            )
        )


@app.local_entrypoint()
def merge(adapter: str = "rs_vqa", out: str = "/data/manifests/rs_vqa_train.jsonl") -> None:
    """Build one adapter corpus from the staged sources, at the declared ratio.

    The ratio is not arbitrary and it is stated in CREDITS: BigEarthNet.txt is
    the largest single source, ahead of RSVQA-HR and RSVQA-LR. If this command
    and that file ever disagree, the file is wrong -- which is worse than not
    having written it.
    """
    argv = [
        "python", "scripts/merge_corpus.py",
        "--adapter", adapter,
        "--source", "ben=/data/chips/canonical.jsonl:chips:40000",
        "--source",
        "rsvqa_hr=/data/eval/rsvqa_hr/rsvqa_hr_train.jsonl:eval/rsvqa_hr:24000",
        "--source",
        "rsvqa_lr=/data/eval/rsvqa_lr/rsvqa_lr_train.jsonl:eval/rsvqa_lr:10000",
        # RSVQA-HR's area rows were excluded while their answers were exact
        # integers: 167 distinct values across 518 test rows meant a model that
        # read the image perfectly still scored near zero. The dataset's own
        # protocol scores area as five classes (Lobry et al. 2020), and
        # scripts/bucket_rsvqa_area.py now applies them to train and test
        # alike. Under those classes 66% "0m2" is a legitimate, visually
        # decidable answer rather than noise, so the rows are back in.
        # NOTE: RSVQA's size words are NOT excluded. The two scales define
        # them 30x apart (paper Table I), but the fix is to state the rule in
        # the prompt rather than delete the rows -- see gsd_prompt_prefix, which
        # attaches the correct threshold per source. Dropping them would have
        # cost 3,067 rows and a capability the PS plausibly tests.
        "--out", out,
    ]
    _save(cpu_job.remote(argv, [out.replace(".jsonl", ".summary.json")]))


@app.local_entrypoint()
def check_resize(root: str = "/data/eval/sn7", sample: int = 12) -> None:
    """Does a pre-resized mosaic give the processor the same tensor?

    Pre-resizing is the only large speed-up left that does not touch the frozen
    max_pixels, and it is only free if the pixels reaching the model are
    unchanged. This measures that instead of assuming it.
    """
    _save(
        gpu_job.remote(
            [
                "python", "scripts/check_resize_equivalence.py",
                "--root", _volume_path(root),
                "--sample", str(sample),
                "--out", "logs/resize_equivalence.json",
            ],
            ["logs/resize_equivalence.json"],
        )
    )


@app.local_entrypoint()
def verify_rows(
    manifest: str = "/data/manifests/change_vqa_train.jsonl",
    root: str = "/data/eval/sn7",
    sample: int = 3000,
) -> None:
    """Re-derive each sampled answer from the labels and check the manifest.

    The corpus checker proves internal consistency. It cannot prove that the
    answer on a row is the answer for *those two images in that order* -- a
    corpus whose answers all belong to the wrong pair passes every other check
    and trains to a smooth, meaningless loss curve. This is the check that
    would have caught it.
    """
    _save(
        cpu_job.remote(
            [
                "python", "scripts/verify_change_rows.py",
                "--manifest", _volume_path(manifest),
                "--root", _volume_path(root),
                "--sample", str(sample),
                "--out", "logs/row_verification.json",
            ],
            ["logs/row_verification.json"],
        )
    )


@app.local_entrypoint()
def audit_masking(root: str = "/data/eval/sn7", limit: int = 0) -> None:
    """How much of each SpaceNet 7 mosaic is masked out.

    ``images_masked`` blanks cloud-covered regions to pure black. A fully black
    month is already rejected at generation; a partial mask is not, and it
    leaves a legible tile whose labels count buildings the model cannot see.
    """
    _save(
        cpu_job.remote(
            [
                "python", "scripts/audit_masking.py",
                "--root", _volume_path(root),
                "--limit", str(limit),
                "--out", "logs/masking.json",
            ],
            ["logs/masking.json"],
        )
    )


@app.local_entrypoint()
def gen_change(
    root: str = "/data/eval/sn7",
    out: str = "/data/manifests/change_vqa_train.jsonl",
    cap: int = 400,
    task_cap: int = 2500,
    limit: int = 0,
    aois: int = 0,
    balance: bool = True,
) -> None:
    """Turn the staged SpaceNet 7 date pairs into change_vqa questions.

    Nothing is invented here: the persistent building IDs in ``labels_match``
    give an exact appeared/demolished inventory per pair, and the script only
    phrases it. ``cap`` is the per-answer ceiling -- the raw pair distribution
    is dominated by adjacent stable months, so without it the corpus would be
    mostly "no" and a model could score well without reading either image.
    """
    _save(
        cpu_job.remote(
            [
                "python", "scripts/gen_change.py",
                "--root", _volume_path(root),
                "--out", _volume_path(out),
                "--image-root", _volume_path("/data"),
                "--cap", str(cap),
                "--task-cap", str(task_cap),
                "--limit", str(limit),
                "--aois", str(aois),
                *(["--balance"] if balance else []),
                "--summary-out", "logs/change_vqa_summary.json",
            ],
            ["logs/change_vqa_summary.json"],
        )
    )


@app.local_entrypoint()
def sample(
    manifest: str = "/data/manifests/rs_vqa_train.jsonl",
    image_root: str = "/data",
    per_type: int = 3,
    group_by: str = "",
    thumb_px: int = 320,
    jpeg_quality: int = 0,
) -> None:
    """Bring back a browsable slice of the corpus: images with their questions.

    Validation says the corpus is internally consistent. It cannot say whether a
    question makes sense for the picture beside it -- that needs a person to
    look, and this is what they look at.
    """
    _save(
        cpu_job.remote(
            [
                "python", "scripts/sample_corpus.py",
                "--manifest", _volume_path(manifest),
                "--image-root", _volume_path(image_root),
                "--per-type", str(per_type),
                "--group-by", group_by,
                "--thumb-px", str(thumb_px),
                "--jpeg-quality", str(jpeg_quality),
                "--out", "logs/corpus_sample.json",
            ],
            ["logs/corpus_sample.json"],
        )
    )


#: The browser needs a web server; nothing else in this file does. Its own image
#: layer so a FastAPI bump cannot invalidate the training image.
browser_image = _with_repo(base_image.uv_pip_install("fastapi[standard]==0.115.6"))


ask_image = _with_repo(
    base_image.uv_pip_install("fastapi[standard]==0.115.6", "python-multipart==0.0.20")
)


@app.cls(
    gpu="L4",
    image=ask_image,
    volumes=VOLUMES,
    timeout=60 * MINUTES,
    # Five minutes of idle before the container goes away. Long enough that a
    # demo does not pay the ~90 s weight load between questions, short enough
    # that forgetting the tab open does not bill overnight.
    scaledown_window=5 * MINUTES,
    # The web endpoint is public and unauthenticated, which is what makes it a
    # demo anyone can open. Uncapped, that also means anonymous traffic can
    # spawn L4 containers without limit and bill for every one. `Api` below was
    # already capped for a state-sharing reason; this cap is purely about
    # spend. One container serves a demo; a queue is the correct response to
    # more load than that, not a bigger bill.
    max_containers=1,
)
class Ask:
    """The adapter behind a web page, for asking it about your own imagery.

    An L4 rather than the A100 the training used: the model is 4B in bf16, so
    inference fits in 24 GB with room to spare, and it costs a third as much per
    hour. Nothing here trains, so the memory that gradients needed is free.
    """

    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, REPO_DIR)
        from satquery.training.generate import VLMRunner
        from scripts.ask_server import ADAPTER, BASE_MODEL

        data_volume.reload()
        self.runner = VLMRunner(BASE_MODEL, adapter_path=ADAPTER)
        print(f"loaded {BASE_MODEL} + {ADAPTER}", flush=True)

    def _answer(self, items, use_adapter: bool):
        model = self.runner.model
        if use_adapter or not hasattr(model, "disable_adapter"):
            return self.runner.answer_all(items, batch_size=1, max_new_tokens=64)
        # The base model's behaviour without a second 8 GB of weights: PEFT
        # switches the LoRA off in place for the duration of the call.
        with model.disable_adapter():
            return self.runner.answer_all(items, batch_size=1, max_new_tokens=64)

    @modal.asgi_app()
    def web(self):
        import sys

        sys.path.insert(0, REPO_DIR)
        from scripts.ask_server import build_app

        return build_app(self._answer)


#: The frontend contract, served with the trained adapters loaded. Same image as
#: ``Ask`` -- it needs torch, PEFT and the model -- plus multipart for uploads.
api_image = _with_repo(
    base_image.uv_pip_install(
        "fastapi[standard]==0.115.6", "python-multipart==0.0.20"
    )
)


@app.function(image=api_image, volumes=VOLUMES, timeout=10 * MINUTES)
def preflight_job() -> dict:
    """Import the served app inside the real image and report what breaks.

    A deploy takes three minutes and a crash-looping container reports itself
    by email, so an import error costs a round trip through the inbox to find.
    This does the one thing that fails -- build the ASGI app, with the same
    image and the same Volume -- and hands back the traceback directly.

    Run it before `modal deploy`, not after: it has caught the same class of
    bug twice now (`scikit-image` absent while `scikit-learn` is present, which
    reads as installed to a skim).
    """
    import sys
    import traceback
    from pathlib import Path as P

    sys.path.insert(0, REPO_DIR)
    data_volume.reload()

    out: dict = {"ok": False, "adapters": {}, "error": None}
    candidates = {
        "rs_vqa": "/data/checkpoints/rs_vqa/adapter",
        "change_vqa": "/data/checkpoints/change_vqa/adapter",
    }
    out["adapters"] = {n: P(pth).is_dir() for n, pth in candidates.items()}
    try:
        from satquery.api.server import build_app

        application = build_app(
            upload_dir="/data/uploads",
            adapters={n: p for n, p in candidates.items() if P(p).is_dir()},
        )
        out["routes"] = len(getattr(application, "routes", []))
        out["ok"] = True
    except Exception:
        out["error"] = traceback.format_exc()
    return out


@app.local_entrypoint()
def preflight() -> None:
    """`modal run scripts/modal_phase0.py::preflight` -- check before deploying."""
    import json as _json

    result = preflight_job.remote()
    print(_json.dumps({k: v for k, v in result.items() if k != "error"}, indent=2))
    if result.get("error"):
        print(chr(10) + "IMPORT FAILED inside the image:" + chr(10))
        print(result["error"])
        raise SystemExit(1)
    print("preflight OK - the API builds inside its image")


@app.cls(
    gpu="L4",
    image=api_image,
    volumes=VOLUMES,
    timeout=60 * MINUTES,
    # Long enough that a demo does not pay the weight load between questions,
    # short enough that a forgotten tab does not bill overnight.
    scaledown_window=5 * MINUTES,
    # One container, because the server keeps scenes, bundles and queries in
    # memory. Left to autoscale, Modal round-robins requests across replicas
    # that do not share state, so a bundle created on one is a 404 on the next
    # -- which showed up as `GET /queries/{id}` returning 404 while
    # `/queries/{id}/trace` returned 200 for the same id, and as bundles
    # vanishing mid-demo.
    #
    # A cap rather than shared storage because this is a single-operator demo
    # API and one L4 already holds the 4B base plus both adapters. If it ever
    # needs to serve concurrent users, the fix is to move those three dicts onto
    # the Volume or a real store, not to raise this number.
    max_containers=1,
)
class Api:
    """``satquery.api.server`` behind a GPU, which is what makes it real.

    On CPU the executor records every learned tool as unavailable and answers
    from deterministic evidence alone -- correct, honest, and not what the
    problem statement asks for: *"A generic LLM or VLM without remote-sensing
    adaptation will not satisfy the requirements."*

    An L4 rather than the A100 the training used: the model is 4B in bf16, so
    inference fits in 24 GB with room to spare and costs a third as much. No
    gradients here, so the memory they needed is free.
    """

    @modal.enter()
    def load(self):
        import sys as _sys

        _sys.path.insert(0, REPO_DIR)
        data_volume.reload()

        from pathlib import Path as P

        # Only adapters that are actually on the Volume. Registering a path that
        # does not exist would have the tool fail at generation time instead of
        # being honestly reported as absent, and ``/meta/health`` would claim a
        # capability the system does not have.
        candidates = {
            "rs_vqa": "/data/checkpoints/rs_vqa/adapter",
            "change_vqa": "/data/checkpoints/change_vqa/adapter",
        }
        self.adapters = {
            name: path for name, path in candidates.items() if P(path).is_dir()
        }
        missing = sorted(set(candidates) - set(self.adapters))
        if missing:
            print(f"adapters not on the Volume, serving without: {missing}", flush=True)

        # Grounding and captioning ship the BASE model, deliberately -- there is
        # no rs_ground_caption adapter and there is not meant to be one. Base
        # Qwen scores 62.7% acc@0.5 on referring grounding, above published
        # fine-tuned GeoChat, so training could only cost accuracy; captioning
        # stays on the prompted baseline for the same reason.
        #
        # `None` is how register_learned_tools spells "serve this manifest from
        # the base model". Leaving the entry out entirely, which is what this
        # did before, made the tool report itself unavailable and the router
        # plan a step nothing could execute -- so a decision to ship untrained
        # read, from outside, as a missing component.
        self.adapters["rs_ground_caption"] = None
        print(f"adapters found: {sorted(self.adapters)}", flush=True)

    @modal.asgi_app()
    def web(self):
        import sys as _sys

        _sys.path.insert(0, REPO_DIR)
        from satquery.api.server import build_app

        return build_app(upload_dir="/data/uploads", adapters=self.adapters)


@app.function(image=browser_image, volumes=VOLUMES, timeout=60 * MINUTES)
@modal.asgi_app()
def browse():
    """A public page for reading the whole corpus. Modal serves it; no tunnel.

    The sampled review page shows 51 rows -- enough to prove the pipeline, not
    enough to judge a dataset. This serves every row with filtering, so a
    suspicion about one question type can be checked against hundreds of
    examples instead of two.

    Read-only: it opens the manifest and the images and has no write path, so
    browsing cannot damage what trains. The image handler refuses any path
    resolving outside the image root.
    """
    import sys

    sys.path.insert(0, REPO_DIR)
    from scripts.corpus_server import build_app

    return build_app()


@app.local_entrypoint()
def saturation(
    source_px: int = 512,
    composites: int = 3,
    ladder: str = "1,2,4,8,16,32",
    adapter: str = "rs_vqa",
) -> None:
    """Item 12b — the batch ceiling at the sequence length training will see.

    The 200-step smoke measured 13.65 GB at micro-batch 4, but on 120 px BEN
    chips: 42 vision tokens a sample. A 512 px benchmark chip is 768. Until this
    runs, nobody knows whether micro-batch 4 even *fits* on real benchmark
    imagery -- and finding out twenty minutes into a twelve-hour run is the
    expensive way to learn it.
    """
    argv = [
        "python", "scripts/batch_saturation_ramp.py",
        "--adapter", adapter,
        "--source-px", str(source_px),
        "--composites", str(composites),
        "--ladder", ladder,
    ]
    _save(
        gpu_job.remote(
            argv,
            ["logs/phase0_batch_saturation.md", "logs/phase0_batch_saturation.json"],
        )
    )


@app.local_entrypoint()
def gates(
    adapter: str = "/data/checkpoints/rs_vqa/adapter",
    batch_size: int = 16,
    composites: int = 3,
    max_new_tokens: int = 16,
    label: str = "",
    limit: int = 0,
    show: int = 0,
    ablate_images: str = "none",
    only: str = "",
) -> None:
    """Score a trained rs_vqa adapter against its four section 6.1 gates.

    modal run scripts/modal_phase0.py::gates

    The bake-off already knew how to run these benchmarks; what it could not do
    was run them against anything but a bare base model, so "the adapter scores
    X" had no way of being measured. ``--adapter ""`` reproduces the zero-shot
    baseline through the identical path, which is the only honest thing to
    compare a fine-tuned number against.

    Three manifests, four numbers: BEN.txt carries binary and MCQ in one file
    and they are separate gates, so the scorer's per-question-type breakdown is
    what gets read, not the headline average.
    """
    # The baseline and the adapter must not share an output path: the second
    # run would overwrite the first, and the comparison this exists to produce
    # would be gone with no error to say so.
    stem = label or ("rs_vqa_gates" if adapter else "rs_vqa_gates_zeroshot")
    if limit:
        stem += f"_limit{limit}"
    if ablate_images != "none":
        stem += f"_{ablate_images}"
    if only:
        stem += "_" + only.replace(",", "-")

    benchmarks = [
        "ben=/data/chips_bench/canonical.jsonl",
        "rsvqa_lr=/data/eval/rsvqa_lr/rsvqa_lr_test.jsonl",
        "rsvqa_hr=/data/eval/rsvqa_hr/rsvqa_hr_test.jsonl",
    ]
    # ``--only rsvqa_lr`` when a change touches one benchmark. Re-scoring all
    # three to learn one number regenerates values already on disk and bills
    # for it: a fix to the LR answer space cannot move BEN or HR by a point.
    if only:
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        benchmarks = [b for b in benchmarks if b.split("=", 1)[0] in wanted]
        if not benchmarks:
            raise SystemExit(f"--only {only!r} matched no benchmark")
    argv = [
        "python", "training/eval/zero_shot.py", "bakeoff",
        "--batch-size", str(batch_size),
        # rs_vqa trains at three composite views per sample. RSVQA rows carry
        # one image and are repeated up to three during training, so scoring
        # them at their natural width would measure an input shape the adapter
        # never saw. Keep in step with adapter_composites("rs_vqa").
        "--composites", str(composites),
        # Only binary_* and mcq_* feed the two BEN gates. Its captioning and
        # grounding rows belong to rs_ground_caption's gates, and leaving them
        # in costs the longest decode in the corpus -- a 200-word caption drags
        # every batch it lands in out to max_new_tokens -- for answers this
        # report then discards.
        "--keep-types", "ben=binary_,mcq_",
        # The longest answer any gate scores is "between 101m2 and 1000m2".
        # 64 was the harness default, sized for captioning.
        "--max-new-tokens", str(max_new_tokens),
        "--out", f"logs/{stem}.md",
        # The raw prediction beside the formatted one, per run, so a suspect
        # score can be read back without paying for the generation again.
        "--dump-predictions", f"logs/{stem}_predictions",
    ]
    for entry in benchmarks:
        argv += ["--benchmark", entry]
    if adapter:
        argv += ["--adapter", _volume_path(adapter)]
    if limit:
        argv += ["--limit", str(limit)]
    if show:
        argv += ["--show", str(show)]
    if ablate_images != "none":
        argv += ["--ablate-images", ablate_images]
    _save(
        gpu_job.remote(argv, [f"logs/{stem}.md", f"logs/{stem}.json"])
    )


@app.local_entrypoint()
def eval_sweep(
    manifest: str = "/data/chips/canonical.jsonl",
    image_root: str = "/data/chips",
    adapter: str = "",
    limit: int = 0,
    batch_size: int = 8,
) -> None:
    """Every ablation plus D2, in one container on one model load.

    The four ablations and item 10 all run ``VLMRunner`` on the same base at the
    same ``max_pixels``; ``point_prior`` is literally D2 under another name.
    Run separately that is five container starts and five 8 GB loads off the
    Volume for work that shares a model -- roughly fifteen minutes of A100 time
    spent on nothing.

    An experiment whose data is missing is reported as ``not_run`` rather than
    killing the sweep, and the job still exits non-zero so a partial sweep
    cannot be mistaken for a complete one.
    """
    argv = [
        "python",
        "training/eval/run_ablations.py",
        "--experiment", "all",
        "--manifest", _volume_path(manifest),
        "--image-root", _volume_path(image_root),
        "--out", "logs/ablation_all.md",
    ]
    if adapter:
        argv += ["--adapter", adapter]
    if limit:
        argv += ["--limit", str(limit)]
    _save(
        gpu_job.remote(
            argv,
            [
                "logs/ablation_all.md",
                "logs/ablation_all.json",
                # D2 writes its own full report from inside point_prior; the
                # sweep report only carries its verdict line.
                "logs/phase0_item10_d2.md",
                "logs/phase0_item10_d2.json",
            ],
        )
    )


@app.local_entrypoint()
def vllm_smoke(adapters: str = "", base_only: bool = False) -> None:
    """Phase 0 item 7 — one base, two LoRAs, hot-swapped.

    ``--adapters "rs_vqa=/data/checkpoints/rs_vqa/adapter,change_vqa=..."``.
    With none, pass ``--base-only``; that is a partial result and the report
    labels it as one rather than ticking the checklist item.
    """
    argv = ["python", "-m", "satquery.serving.vllm_smoke"]
    for entry in filter(None, (a.strip() for a in adapters.split(","))):
        argv += ["--adapter", entry]
    if base_only or not adapters:
        argv.append("--base-only")
    _save(
        vllm_job.remote(
            argv, ["logs/phase0_item7_vllm_smoke.md", "logs/phase0_item7_vllm_smoke.json"]
        )
    )
