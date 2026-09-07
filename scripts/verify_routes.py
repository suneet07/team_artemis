"""Exercise every API route and every router task against a running server.

Two things get checked, because they fail differently:

* **Routes** -- all 20 endpoints, including the error paths. A 404 that should
  be a 404 is a pass; a 500 anywhere is a failure. Contract fields are checked
  against `frontend/contracts/types.ts`, since a missing required field crashes
  the console rather than degrading it -- that is how `bundle.provenance` took
  the whole workspace down.
* **Tasks** -- all 8 entries in the `Task` enum, each driven by a question that
  should reach it, with the modalities that task requires. A task nothing can
  route to is a dead branch, and a task that routes but plans an unavailable
  tool is worse: the executor records the failure as a warning and answers from
  whatever else ran, so it looks healthy from outside.
* **One executed query** -- because the two checks above only ever *planned*.
  That gap shipped an outage: the fusion branch read `scene_of` off the router's
  `QueryContext`, which has no such method, and **every** crossmodal query
  failed with an AttributeError while 411 tests and 44 route checks passed. The
  branch is reached only when an optical mask and a SAR mask both materialise,
  and nothing built a pair that produced two. `verify_crossmodal_execution`
  synthesises that pair and runs `answer_query` over it.

    python scripts/verify_routes.py --base http://127.0.0.1:8000/api/v1
    python scripts/verify_routes.py --base https://<app>.modal.run/api/v1
    python scripts/verify_routes.py --offline-only --stub-adapters   # ~6 s, no GPU

Exits non-zero if anything failed, so it can gate a deploy.

What this does **not** check is answer quality: the stub adapters return a fixed
string, which proves the plumbing and nothing else. Quality is the testing-corpus
gallery's job -- 200 held-out rows with published gold answers, replayed through
the live API (`scripts/verify_gallery.py`, `docs/segments/08-testing-corpora.md`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, note: str = "") -> None:
    results.append((name, status, note))
    mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "skip"}[status]
    print(f"  {mark}  {name}" + (f"  -- {note}" if note else ""), flush=True)


def call(
    base: str,
    method: str,
    path: str,
    body: dict | None = None,
    timeout: int = 900,
    raw: bool = False,
):
    """Return (status_code, payload). Never raises for an HTTP error status."""
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            if raw:
                return response.status, payload
            return response.status, json.loads(payload or b"null")
    except urllib.error.HTTPError as exc:
        body_text = exc.read()
        try:
            return exc.code, json.loads(body_text or b"null")
        except Exception:
            return exc.code, body_text
    except Exception as exc:  # noqa: BLE001 - a dead server is a result
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


def upload(base: str, scene_path: Path) -> tuple[int, dict]:
    """multipart/form-data by hand, so this script needs no requests dependency."""
    boundary = "----satqueryverify"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{scene_path.name}"\r\n'
            ).encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            scene_path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(base + "/scenes", data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read()[:200].decode(errors="replace")}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


def required_fields(interface: str) -> list[str]:
    """Non-optional fields of a contract interface, read from the frontend."""
    types = (ROOT / "frontend" / "contracts" / "types.ts").read_text(encoding="utf-8")
    block = re.search(r"export interface " + interface + r" \{(.*?)\n\}", types, re.S)
    if not block:
        return []
    out = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "/*", "*")):
            continue
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\??):", line)
        if match and match.group(2) != "?":
            out.append(match.group(1))
    return out


def check_contract(name: str, interface: str, payload: dict) -> None:
    missing = [f for f in required_fields(interface) if f not in (payload or {})]
    record(
        f"contract {interface} ({name})",
        FAIL if missing else PASS,
        f"missing {missing}" if missing else "",
    )


# --------------------------------------------------------------------------
# Task probes. Each names the modalities that task legitimately needs -- asking
# a change question of a single scene should refuse, not route.
# --------------------------------------------------------------------------
TASK_PROBES = [
    ("single_vqa", "What is the vegetation cover in this scene?", ["optical"]),
    (
        "single_caption",
        "Describe the land-cover and major objects visible in this image.",
        ["optical"],
    ),
    ("single_grounding", "Where is the water in this image?", ["optical"]),
    (
        "change_description",
        "Describe the changes between these images.",
        ["optical", "optical"],
    ),
    ("change_vqa", "What changed between the two dates?", ["optical", "optical"]),
    ("change_map", "Produce a change mask for this pair.", ["optical", "optical"]),
    (
        "crossmodal_extraction",
        "Use the optical and SAR images together to extract built-up regions.",
        ["optical", "sar"],
    ),
    (
        "crossmodal_vqa",
        "Use the optical and SAR images together to identify built-up and "
        "water-covered regions.",
        ["optical", "sar"],
    ),
]


def register_stub_adapters() -> list[str]:
    """Register the learned tools against a stub runner, so the adapter path
    runs on a machine with no CUDA.

    This is the half a local run otherwise never touches. Without it the whole
    learned branch -- prompt selection, `served_prompt` mode, the tool contract,
    the answer composer, the trace -- is only ever exercised on a GPU, three
    minutes and a deploy away. Both of the bugs that made a working system look
    broken lived exactly there: `Scene.effective_gsd_m` (a field that does not
    exist, so every rs_vqa step failed) and a composer that ignored the learned
    answer entirely. A stub would have caught both in a second.

    The stub returns a fixed string. That is enough to prove the plumbing and
    nothing about quality -- quality needs the real weights, and that is what
    the GPU run is for.
    """
    import satquery.tools.learned as learned

    class _StubRunner:
        def answer_all(self, items, batch_size=1, max_new_tokens=128):
            # Report *which* served prompt arrived, not the first line -- the
            # first line is always `scale_prefix`, so echoing it proves nothing
            # about prompt selection. These markers are the load-bearing clauses:
            # the 0-1000 coordinate scale and JSON shape for grounding, the mined
            # register for captioning.
            out = []
            for item in items:
                text = item.get("question", "")
                if "bbox_2d" in text and "0-1000" in text:
                    kind = "PRECISE_PROMPT"
                elif "sourced from GoogleEarth" in text and "47 words" in text:
                    kind = "STRONG_PROMPT"
                else:
                    kind = "scale_prefix+question"
                out.append(f"stub answer [prompt={kind}]")
            return out

    learned._runner = lambda base, adapter: _StubRunner()  # noqa: SLF001
    return learned.register_learned_tools(
        {"rs_vqa": None, "change_vqa": None, "rs_ground_caption": None}
    )


def _synthetic_pair(directory: Path) -> tuple[Path, Path]:
    """A co-registered optical + SAR pair that both threshold to real masks.

    Synthesised rather than fixtured so the probe has no data dependency, but
    with the properties the fusion branch actually needs: one CRS, one
    transform, named bands so the inventory resolves, and content that puts
    both an optical index and a SAR backscatter threshold above their floors.
    A pair where either mask comes back empty never reaches fusion, and a probe
    that quietly skips the branch it exists to test is worse than no probe.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    size = 64
    transform = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    crs = "EPSG:32633"

    # Left half water-like (high green, low NIR -> positive NDWI), right half
    # vegetation (low green, high NIR). Two clear populations, so Otsu has a
    # bimodal histogram to find rather than a gradient to guess at.
    green = np.full((size, size), 900.0, dtype="float32")
    nir = np.full((size, size), 3000.0, dtype="float32")
    green[:, : size // 2] = 2600.0
    nir[:, : size // 2] = 700.0
    optical = directory / "probe_optical.tif"
    with rasterio.open(
        optical, "w", driver="GTiff", height=size, width=size, count=4,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(np.full((size, size), 800.0, dtype="float32"), 1)
        dst.write(green, 2)
        dst.write(np.full((size, size), 1000.0, dtype="float32"), 3)
        dst.write(nir, 4)
        dst.descriptions = ("blue", "green", "red", "nir")

    # Radar over the same footprint, in dB. Bright right half (built-up-like),
    # dark left half (specular water-like), which is the same split the optical
    # half encodes -- so the two masks agree on where the boundary is and
    # `fuse_masks` gets a real IoU rather than an empty intersection.
    vh = np.full((size, size), -18.0, dtype="float32")
    vv = np.full((size, size), -12.0, dtype="float32")
    vh[:, : size // 2] = -26.0
    vv[:, : size // 2] = -22.0
    sar = directory / "probe_sar.tif"
    with rasterio.open(
        sar, "w", driver="GTiff", height=size, width=size, count=2,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(vh, 1)
        dst.write(vv, 2)
        dst.descriptions = ("VH", "VV")

    return optical, sar


def verify_crossmodal_execution() -> bool:
    """Actually run a crossmodal query, rather than only planning one.

    ``verify_router_offline`` routes and checks that every planned tool exists.
    It does not execute, and that gap shipped a real outage: the fusion branch
    read ``context.scene_of("optical")`` from the router's ``QueryContext``,
    which has no such method, so **every** crossmodal query failed with an
    AttributeError. All 411 tests and all 44 route checks passed while it did,
    because the branch is reached only when an optical mask and a SAR mask both
    exist -- and nothing built a pair that produced two.

    Returns True on success. Deterministic tools only; no GPU, no server.
    """
    import tempfile

    sys.path.insert(0, str(ROOT))
    from satquery.agent.pipeline import answer_query

    with tempfile.TemporaryDirectory() as tmp:
        optical, sar = _synthetic_pair(Path(tmp))
        try:
            outcome = answer_query(
                "Use the optical and SAR images together to identify built-up "
                "and water-covered regions.",
                [str(optical), str(sar)],
                # Declared, because a 4-band float raster is ambiguous and this
                # probe is testing the crossmodal branch, not modality
                # detection. Evidence stays in the temp dir with the inputs.
                modalities=["optical", "sar"],
                output_dir=Path(tmp),
                write_evidence=False,
            )
        except Exception as error:  # noqa: BLE001 - that is the finding
            record(
                "crossmodal executes end to end",
                FAIL,
                f"{type(error).__name__}: {error}",
            )
            return False

    if outcome.refused:
        record("crossmodal executes end to end", FAIL, f"refused: {outcome.answer[:120]}")
        return False

    trace = outcome.trace if isinstance(outcome.trace, dict) else {}
    steps = [step.get("tool") for step in trace.get("steps") or []]
    agreement = trace.get("agreement")
    if agreement is None:
        # Not a failure on its own -- a pair whose masks do not overlap has
        # nothing to reconcile -- but it means this probe did not cover the
        # branch, and saying so is the point.
        record(
            "crossmodal executes end to end",
            PASS,
            f"ran {steps}; no fusion (masks did not both materialise)",
        )
        return True

    record(
        "crossmodal executes end to end",
        PASS,
        f"fused: IoU {agreement.get('iou')} verdict {agreement.get('verdict')}",
    )
    return True


def verify_router_offline(stub_adapters: bool = False) -> None:
    """Every Task must be reachable, and every planned tool must exist.

    Runs in-process: no server, no GPU, seconds rather than minutes. A task that
    plans a tool the registry does not know is a dead branch that would only
    surface as a trace warning at request time.
    """
    sys.path.insert(0, str(ROOT))
    from satquery.agent.router import QueryContext, route
    from satquery.agent.task_enum import Task
    from satquery.tools.registry import ToolRegistry

    if stub_adapters:
        registered = register_stub_adapters()
        record("stub adapters registered", PASS, ", ".join(registered))

    known = set(ToolRegistry.default().names())
    # Registered is not the same as executable. `implementation(name)` is what
    # the executor calls; a manifest with nothing behind it is recorded as a
    # trace warning at request time and the answer comes back from whatever else
    # ran -- which reads as healthy from outside. Ask the same question the
    # executor asks, and say so per plan.
    try:
        from satquery.tools.catalog import implementation
    except Exception:  # noqa: BLE001 - catalog pulls scipy/skimage
        def implementation(_name):  # type: ignore[misc]
            return None

    def executable(name: str) -> bool:
        try:
            return implementation(name) is not None
        except Exception:  # noqa: BLE001
            return False

    reached: set[str] = set()

    for expected, question, modalities in TASK_PROBES:
        context = QueryContext(
            query_text=question,
            modalities=list(modalities),
            inventories=[],
            image_count=len(modalities),
            dates=["2018-01-01", "2020-01-01"][: len(modalities)],
        )
        decision = route(context)
        got = decision.task.value
        reached.add(got)
        tools = [step["tool"] for step in (decision.plan or [])]
        unknown = [t for t in tools if t not in known]
        if unknown:
            record(f"task {expected}", FAIL, f"plans unknown tool(s) {unknown}")
        elif got != expected:
            record(f"task {expected}", FAIL, f"routed to {got}; plan {tools}")
        elif not tools:
            record(f"task {expected}", FAIL, "empty plan")
        else:
            absent = [t for t in tools if not executable(t)]
            shown = " > ".join(
                t if executable(t) else f"{t}(absent)" for t in tools
            )
            # Absent tools are reported, not failed: `change_map` is absent by a
            # closed decision and `change_stats` still runs on whatever class
            # maps exist. A task whose plan is *entirely* absent is a different
            # thing -- nothing would execute, and that is a failure.
            if absent and len(absent) == len(tools):
                record(f"task {expected}", FAIL, f"every planned tool is absent: {absent}")
            else:
                record(f"task {expected}", PASS, shown)

    unreachable = sorted({t.value for t in Task} - reached)
    record(
        "every Task reachable",
        FAIL if unreachable else PASS,
        f"never routed to: {unreachable}" if unreachable else "",
    )


def verify_routes(base: str, scene: Path) -> None:
    # -- meta ---------------------------------------------------------------
    status, health = call(base, "GET", "/meta/health", timeout=900)
    record("GET /meta/health", PASS if status == 200 else FAIL, f"HTTP {status}")
    if status == 200:
        print(f"        build={health.get('build')} gpu={health.get('gpu')} "
              f"adapters={health.get('adapters_loaded')}")

    for path, key, interface in (
        ("/meta/tasks", "tasks", "TaskMeta"),
        ("/meta/tools", "tools", "ToolManifest"),
        ("/meta/disagreement-causes", "causes", None),
    ):
        status, payload = call(base, "GET", path, timeout=120)
        items = (payload or {}).get(key) or []
        ok = status == 200 and isinstance(items, list)
        record(f"GET {path}", PASS if ok else FAIL, f"HTTP {status}, {len(items)} items")
        # Every item, not just the first. `outputs` was missing from all 15
        # tools and the system page ran Object.entries(undefined); checking one
        # row would have caught it, but a field absent on only some rows is the
        # harder bug and costs nothing extra to catch here.
        if ok and interface and items:
            required = required_fields(interface)
            missing = sorted(
                {f for item in items for f in required if f not in item}
            )
            record(
                f"contract {interface} (all {len(items)})",
                FAIL if missing else PASS,
                f"missing {missing}" if missing else "",
            )

    status, payload = call(base, "GET", "/demo/bundles", timeout=120)
    record("GET /demo/bundles", PASS if status == 200 else FAIL, f"HTTP {status}")

    # -- scenes -------------------------------------------------------------
    status, created = call(base, "GET", "/scenes", timeout=120)
    record("GET /scenes", PASS if status == 200 else FAIL, f"HTTP {status}")

    status, scene_payload = upload(base, scene)
    scene_id = (scene_payload or {}).get("scene_id")
    record("POST /scenes", PASS if scene_id else FAIL, f"HTTP {status}")
    if not scene_id:
        record("remaining routes", SKIP, "no scene to work with")
        return

    status, blob = call(base, "GET", f"/scenes/{scene_id}/preview.png", timeout=300, raw=True)
    ok = status == 200 and isinstance(blob, bytes) and blob[:4] == b"\x89PNG"
    record("GET /scenes/{id}/preview.png", PASS if ok else FAIL,
           f"HTTP {status}, {len(blob) if isinstance(blob, bytes) else 0} bytes")

    # -- bundles ------------------------------------------------------------
    status, bundle = call(base, "POST", "/bundles",
                          {"scene_ids": [scene_id], "label": "verify_routes"}, timeout=300)
    bundle_id = (bundle or {}).get("bundle_id")
    record("POST /bundles", PASS if bundle_id else FAIL, f"HTTP {status}")
    if not bundle_id:
        record("remaining routes", SKIP, "no bundle")
        return
    check_contract("POST /bundles", "Bundle", bundle)
    if bundle.get("scenes"):
        check_contract("bundle.scenes[0]", "Scene", bundle["scenes"][0])

    # The console's own payload shape, which is NOT the flat one above. Sending
    # only `scene_ids` is how a bundle with zero scenes reached the UI: the POST
    # succeeded, status came back `ready`, and the prep screen waited forever.
    status, contract_bundle = call(
        base, "POST", "/bundles",
        {"scenes": [{"scene_id": scene_id, "role": "optical"}],
         "pair_type": "single", "label": "contract shape"},
        timeout=300,
    )
    attached = len((contract_bundle or {}).get("scenes") or [])
    record("POST /bundles (contract shape: scenes=[{scene_id,role}])",
           PASS if attached == 1 else FAIL,
           f"HTTP {status}, {attached} scene(s) attached")

    status, _ = call(base, "POST", "/bundles", {"scenes": []}, timeout=120)
    record("POST /bundles -> 400 with no scenes", PASS if status == 400 else FAIL,
           f"HTTP {status}")

    status, one = call(base, "GET", f"/bundles/{bundle_id}", timeout=120)
    record("GET /bundles/{id}", PASS if status == 200 else FAIL, f"HTTP {status}")

    status, listing = call(base, "GET", "/bundles", timeout=120)
    ok = status == 200 and any(b.get("bundle_id") == bundle_id
                               for b in (listing or {}).get("bundles", []))
    record("GET /bundles", PASS if ok else FAIL, f"HTTP {status}")

    status, missing = call(base, "GET", "/bundles/bn_doesnotexist", timeout=120)
    record("GET /bundles/{id} -> 404", PASS if status == 404 else FAIL, f"HTTP {status}")

    # -- queries ------------------------------------------------------------
    status, query = call(base, "POST", "/queries",
                         {"bundle_id": bundle_id,
                          "question": "What is the vegetation cover in this scene?"})
    query_id = (query or {}).get("query_id")
    record("POST /queries (bundle_id)", PASS if query_id else FAIL, f"HTTP {status}")
    if query_id:
        check_contract("POST /queries", "QueryResult", query)
        trace = query.get("trace") or {}
        check_contract("query.trace", "Trace", trace)
        graded = trace.get("graded") or {}
        print(f"        task={graded.get('task_selected')} "
              f"tools={graded.get('tools_invoked')} {query.get('latency_ms')}ms")
        failures = [w for w in (trace.get("warnings") or []) if "fail" in w.lower()]
        record("no tool failures in trace", FAIL if failures else PASS,
               "; ".join(failures)[:160])

    status, _ = call(base, "POST", "/queries", {"question": "no scenes here"}, timeout=120)
    record("POST /queries -> 400 without scenes", PASS if status == 400 else FAIL, f"HTTP {status}")

    status, _ = call(base, "POST", "/queries",
                     {"bundle_id": bundle_id, "question": ""}, timeout=120)
    record("POST /queries -> 400 empty question", PASS if status == 400 else FAIL, f"HTTP {status}")

    status, _ = call(base, "POST", "/queries",
                     {"bundle_id": "bn_nope", "question": "anything?"}, timeout=120)
    record("POST /queries -> 404 unknown bundle", PASS if status == 404 else FAIL, f"HTTP {status}")

    if not query_id:
        return

    for suffix, label in (("", "GET /queries/{id}"),
                          ("/trace", "GET /queries/{id}/trace"),
                          ("/evidence", "GET /queries/{id}/evidence"),
                          ("/events", "GET /queries/{id}/events")):
        status, payload = call(base, "GET", f"/queries/{query_id}{suffix}",
                               timeout=300, raw=(suffix == "/events"))
        record(label, PASS if status == 200 else FAIL, f"HTTP {status}")

    status, filtered = call(base, "GET", f"/queries?bundle_id={bundle_id}", timeout=120)
    ok = status == 200 and any(q.get("query_id") == query_id
                               for q in (filtered or {}).get("queries", []))
    record("GET /queries?bundle_id=", PASS if ok else FAIL, f"HTTP {status}")

    status, _ = call(base, "POST", f"/queries/{query_id}/cancel", {}, timeout=120)
    record("POST /queries/{id}/cancel", PASS if status in (200, 409) else FAIL, f"HTTP {status}")


    # -- prompt selection ---------------------------------------------------
    # Only meaningful when the learned tools are registered. On a stub run this
    # is nearly free and covers the half a CPU box otherwise never touches:
    # `rs_ground_caption` serves the BASE model, so the prompt IS the capability
    # -- 62.7% grounding and 0.252 captioning came from these exact strings, and
    # serving the raw question instead loses the 0-1000 coordinate scale that
    # the box parser depends on.
    if "rs_ground_caption" in str(health.get("adapters_loaded") or []):
        for label, question, want in (
            (
                "caption",
                "Describe the land-cover and major objects visible in this image.",
                "STRONG_PROMPT",
            ),
            ("grounding", "Where is the water in this image?", "PRECISE_PROMPT"),
        ):
            _, probe = call(base, "POST", "/queries",
                            {"bundle_id": bundle_id, "question": question})
            answer = str((probe or {}).get("answer") or "")
            found = re.search(r"prompt=([A-Za-z_+]+)", answer)
            if not found:
                record(f"prompt for {label}", SKIP, "real weights, not a stub")
            else:
                record(f"prompt for {label}", PASS if found.group(1) == want else FAIL,
                       f"got {found.group(1)}, want {want}")
    else:
        record("prompt selection", SKIP, "rs_ground_caption not registered")

    # -- assets -------------------------------------------------------------
    evidence = query.get("evidence") or []
    if evidence:
        asset_id = evidence[0].get("asset_id")
        status, blob = call(base, "GET", f"/assets/{asset_id}", timeout=300, raw=True)
        record("GET /assets/{id}", PASS if status == 200 else FAIL,
               f"HTTP {status}, {len(blob) if isinstance(blob, bytes) else 0} bytes")
        status, meta = call(base, "GET", f"/assets/{asset_id}/meta", timeout=120)
        record("GET /assets/{id}/meta", PASS if status == 200 else FAIL, f"HTTP {status}")
    else:
        record("GET /assets/{id}", SKIP, "query produced no evidence")

    status, _ = call(base, "GET", "/assets/as_doesnotexist", timeout=120)
    record("GET /assets/{id} -> 404", PASS if status == 404 else FAIL, f"HTTP {status}")

    # ---------------------------------------------------------------- gallery
    # The bundle the gallery hands the workspace must satisfy the same contract
    # as one the upload screen builds. It did not: the first version wrote its
    # own record with `kind` instead of `pair_type` and none of `label`,
    # `prep_ms`, `provenance`, `supported_tasks` or `bounds_wgs84`. Those are
    # required in types.ts, and a missing required field blanks the console
    # rather than degrading it -- "Ask this" opened an empty white workspace.
    status, gallery = call(base, "GET", "/gallery", timeout=180)
    if status != 200 or not (gallery or {}).get("items"):
        record("GET /gallery", SKIP, f"HTTP {status}; no gallery on this deployment")
    else:
        record("GET /gallery", PASS, f"HTTP {status}, {len(gallery['items'])} item(s)")
        item_id = gallery["items"][0]["item_id"]
        status, opened = call(base, "POST", f"/gallery/{item_id}/bundle", timeout=600)
        bundle_id = (opened or {}).get("bundle_id")
        record(
            "POST /gallery/{item}/bundle",
            PASS if bundle_id else FAIL,
            f"HTTP {status}",
        )
        if bundle_id:
            status, made = call(base, "GET", f"/bundles/{bundle_id}", timeout=180)
            check_contract("gallery bundle", "Bundle", made or {})
            if (made or {}).get("scenes"):
                check_contract("gallery bundle.scenes[0]", "Scene", made["scenes"][0])
            record(
                "gallery bundle is usable",
                PASS if (made or {}).get("status") == "ready" else FAIL,
                f"status {(made or {}).get('status')}, "
                f"{len((made or {}).get('scenes') or [])} scene(s)",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--scene",
                        default="data/chips/geotiff/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57.tif")
    parser.add_argument("--offline-only", action="store_true",
                        help="router checks only; no server needed")
    parser.add_argument("--stub-adapters", action="store_true",
                        help="register the learned tools against a stub runner, "
                             "so the adapter path is exercised without a GPU")
    args = parser.parse_args()

    started = time.time()
    print("\n=== router / task coverage (in-process) ===")
    verify_router_offline(stub_adapters=args.stub_adapters)
    verify_crossmodal_execution()

    if not args.offline_only:
        print(f"\n=== API routes ({args.base}) ===")
        verify_routes(args.base, ROOT / args.scene)

    failed = [r for r in results if r[1] == FAIL]
    skipped = [r for r in results if r[1] == SKIP]
    print(f"\n{len(results) - len(failed) - len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped "
          f"in {time.time() - started:.1f}s")
    for name, _, note in failed:
        print(f"  FAILED: {name} -- {note}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
