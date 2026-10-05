"""The frozen frontend contract, served from this repo's agent.

``frontend/contracts/openapi.yaml`` is frozen: the frontend builds to it and
mocks it with MSW, and the plan's own instruction was "swapping to the live API
is one env var". This makes that true for the endpoints the frontend actually
calls, which is a smaller set than the full contract -- ``frontend/mocks/
handlers.ts`` names ten, and the graded path is four of them.

**Why a new module rather than a merge.** A second backend exists on a teammate's
branch (``satquery/api/app.py``) built on ``agent/bundle.py``, ``agent/probe.py``
and ``fusion/reconcile.py``, none of which are in this tree, while this tree has
``agent/pipeline.py``, ``fusion/fusion.py`` and the whole ``ingest/`` package
that theirs lacks. Reconciling two diverged trees is a deliberate merge for
their owners; standing up the contract over ``answer_query`` is additive and
destroys nothing.

**Three endpoints are pure projections of things this repo already holds**, and
that is the point -- they cannot drift from the system's behaviour because they
are not restatements of it:

* ``/meta/tools`` reads the tool manifests the parameter gate validates against;
* ``/meta/disagreement-causes`` reads ``DISAGREEMENT_RULES``, the same table D1
  reconciles with;
* ``/meta/tasks`` reads the ``Task`` enum the router routes to.

A hand-written list would be a second source of truth that looks right until the
day it disagrees.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Imported at module level, not inside build_app: ``from __future__ import
# annotations`` turns every annotation into a string, and FastAPI resolves those
# ForwardRefs against module globals. A function-local import leaves
# ``UploadFile`` unresolvable and the app refuses to build.
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

__all__ = ["build_app"]

#: What the frontend's ``VITE_API_BASE`` defaults to.
API_PREFIX = "/api/v1"

#: Ceiling on a single upload. Defaults to the 4 GB the frontend advertises in
#: `client.ts`; the point is that the ceiling exists on the side that can
#: enforce it.
MAX_UPLOAD_BYTES = int(os.environ.get("SATQUERY_MAX_UPLOAD_BYTES", 4 * 1024**3))
_UPLOAD_CHUNK_BYTES = 1024 * 1024

#: Raster image extensions an upload may carry. Mirrors `IMAGE_EXTENSIONS` in
#: the frontend's `UploadScreen.tsx`, which is only a courtesy to the user --
#: this is the check that holds.
UPLOAD_IMAGE_EXTENSIONS = frozenset(
    {
        ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jfif", ".jp2", ".j2k",
        ".webp", ".bmp", ".gif", ".avif", ".heic", ".heif",
    }
)  # fmt: skip


def _is_image_upload(filename: str | None, content_type: str | None) -> bool:
    """Whether an upload claims to be a raster image.

    The extension is the main signal: browsers send GeoTIFF and JPEG 2000 with
    an empty or generic content type. The content type is the fallback for an
    image whose name carries no known extension.
    """
    if Path(filename or "").suffix.lower() in UPLOAD_IMAGE_EXTENSIONS:
        return True
    kind = (content_type or "").lower()
    return kind.startswith("image/") and kind != "image/svg+xml"

#: Browser origins allowed to call this API. `*` keeps the public demo working
#: for anyone opening the link, which is what it is for. Set this to the
#: deployed frontend's origin to narrow it -- comma-separated, no spaces.
ALLOWED_ORIGINS = [
    o for o in os.environ.get("SATQUERY_ALLOWED_ORIGINS", "*").split(",") if o
]

#: Optional shared key. Unset -- the default -- leaves the API open, which is
#: what a public demo needs. Set it and every route except the health checks
#: requires a matching `X-Api-Key`, which is the header the frontend has always
#: sent from `VITE_API_KEY`.
#:
#: A key in `VITE_API_KEY` is baked into the built bundle and is therefore
#: public: it stops a stranger who has only the API URL, not someone who has
#: opened the site. That is the whole of what it buys.
API_KEY = os.environ.get("SATQUERY_API_KEY") or None
_UNAUTHENTICATED_PATHS = frozenset({"/health", f"{API_PREFIX}/meta/health"})

#: Human labels for the task enum. The enum is the source of which tasks exist;
#: this only supplies wording, and a task missing here still appears.
TASK_LABELS: dict[str, tuple[str, str, list[str]]] = {
    "single_vqa": ("Ask about one image", "Free-form question over a single scene.",
                   ["single", "crossmodal", "bitemporal"]),
    "single_caption": ("Describe the scene", "Caption of a single scene.",
                       ["single", "crossmodal", "bitemporal"]),
    "single_grounding": ("Locate something", "Find and box a referred object or region.",
                         ["single", "crossmodal", "bitemporal"]),
    "change_description": ("Describe the changes",
                           "Narrative of what changed between two dates.", ["bitemporal"]),
    "change_vqa": ("Ask about the change",
                   "Question about what changed between two dates.", ["bitemporal"]),
    "change_map": ("Map the change", "Pixel map of what changed.", ["bitemporal"]),
    "crossmodal_extraction": ("Combine optical and SAR",
                              "Extract complementary information from a co-registered pair.",
                              ["crossmodal"]),
    "crossmodal_vqa": ("Ask across modalities",
                       "Question answered from optical and SAR together.", ["crossmodal"]),
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _tasks() -> dict[str, Any]:
    from satquery.agent.task_enum import Task

    out = []
    for task in Task:
        label, description, requires = TASK_LABELS.get(
            task.value, (task.value.replace("_", " ").title(), "", ["single"])
        )
        out.append(
            {
                "task": task.value,
                "label": label,
                "description": description,
                "requires": requires,
            }
        )
    return {"tasks": out}


def _tools() -> dict[str, Any]:
    """The manifests themselves, so the UI cannot show a tool the gate rejects."""
    from satquery.tools.registry import ToolRegistry

    def _param(spec) -> dict[str, Any]:
        """One ``ParamSpec`` as the contract's JSON shape.

        Only the fields that are set: the frontend renders a parameter row per
        key, and emitting ``range: null`` for an enum would draw an empty
        control the tool does not have.
        """
        out: dict[str, Any] = {"type": getattr(spec, "type", "string")}
        for field in ("values", "range", "requires_bands"):
            value = getattr(spec, field, None)
            if value:
                out[field] = list(value)
        if getattr(spec, "has_default", False):
            out["default"] = spec.default
        if getattr(spec, "optional", False):
            out["optional"] = True
        return out

    # Whether a manifest has an executable behind it, asked of the same catalog
    # the executor calls. A manifest is a promise; an implementation is the
    # thing that can keep it. Listing all fifteen with no distinction told the
    # UI that `change_vqa` and `rs_vqa` were ready on a host with no adapters
    # loaded, which is the one claim this API must never make.
    try:
        from satquery.tools.catalog import implementation
    except Exception:  # noqa: BLE001 - catalog pulls scipy/skimage; manifests do not
        def implementation(_name):  # type: ignore[misc]
            return None

    registry = ToolRegistry.default()
    tools = []
    for name in sorted(registry.names()):
        manifest = registry.get(name)
        try:
            available = implementation(name) is not None
        except Exception:  # noqa: BLE001 - an unloadable tool is an absent one
            available = False
        tools.append(
            {
                "available": available,
                "name": manifest.name,
                "description": manifest.description,
                "version": manifest.version,
                "required_modalities": list(manifest.required_modalities),
                "confidence_source": manifest.confidence_source,
                "expected_latency_ms": manifest.expected_latency_ms,
                "permitted_parameters": {
                    key: _param(spec)
                    for key, spec in (manifest.permitted_parameters or {}).items()
                },
                # Required by the contract and never sent, so the system page
                # ran Object.entries(undefined) and blanked -- the same failure
                # as the missing `bundle.provenance`, and equally invisible from
                # the API side because nothing errored here.
                #
                # Read from the manifest rather than restated: `outputs` is what
                # the tool promises to produce, and the console renders it as
                # that tool's contract. A hand-written copy would drift from the
                # YAML the parameter gate validates against.
                "outputs": {
                    key: dict(spec) if isinstance(spec, dict) else {"type": str(spec)}
                    for key, spec in (manifest.outputs or {}).items()
                },
            }
        )
    return {"tools": tools}


def _disagreement_causes() -> dict[str, Any]:
    """Section 4.7.2's table, read from the rules D1 actually reconciles with."""
    from satquery.fusion.fusion import DISAGREEMENT_RULES

    return {
        "causes": [
            {
                "code": rule.cause,
                "label": rule.cause.replace("_", " ").capitalize(),
                "trusted_modality": rule.winner,
                "explanation": rule.explanation,
            }
            for rule in DISAGREEMENT_RULES
        ]
    }


def _health(adapters: list[str]) -> dict[str, Any]:
    import importlib.util

    from satquery.agent.trace import SCHEMA_VERSION

    return {
        "status": "ok",
        "version": "0.4.1",
        "trace_schema_version": SCHEMA_VERSION,
        "serving": "local",
        # Reported, never assumed: the frontend shows a degraded-mode banner and
        # it should reflect the machine it is actually talking to.
        "gpu": bool(importlib.util.find_spec("torch"))
        and _cuda_available(),
        "offline_mode": False,
        "adapters_loaded": adapters,
        "demo_bundles_warm": 0,
        # Which copy of the code is answering. Modal keeps warm containers
        # serving the previous build until they scale down, so a deploy followed
        # by a request can silently exercise the code you just replaced -- three
        # separate "the fix didn't work" investigations tonight were that.
        # Polling this until it changes replaces a conservative sleep with a
        # measurement, and turns a 7-minute wait into however long it actually
        # takes.
        "build": _build_id(),
    }


#: Computed once per process. The tree cannot change under a running container.
_BUILD_ID: str | None = None


def build_id() -> str:
    """Short digest of the served source tree, from file **contents**.

    Content and nothing else, so the same code always produces the same id
    wherever it is computed. That is what lets a deploy check compare the id a
    container reports against one computed from the local tree and conclude
    "the right code is live" -- rather than only "the id changed", which cannot
    tell a correct no-op deploy from a container that never restarted.

    It used to mix in size and mtime, which made it cheap but strictly local:
    copying the tree into an image rewrites every mtime, so the served id could
    never match a local one. The symptom was a deploy that touched only
    ``scripts/`` and ``tests/`` reporting "WARN still serving <old id>" -- true,
    correct, and indistinguishable from a failure.

    Reading the bodies costs a few hundred KB, paid once per process.
    """
    global _BUILD_ID
    if _BUILD_ID is not None:
        return _BUILD_ID

    import hashlib
    from pathlib import Path as P

    root = P(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    try:
        for path in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    except OSError:
        return "unknown"
    _BUILD_ID = digest.hexdigest()[:12]
    return _BUILD_ID


def _build_id() -> str:
    """Backwards-compatible alias; :func:`build_id` is the public name."""
    return build_id()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - absence of torch is a valid answer
        return False


#: Extension -> what the evidence actually is. A GeoTIFF mask is downloadable
#: and georeferenced; a PNG overlay is for looking at. The frontend renders them
#: differently, so the distinction is carried rather than inferred from a URL.
_ASSET_KINDS = {
    ".tif": ("mask_geotiff", "image/tiff"),
    ".tiff": ("mask_geotiff", "image/tiff"),
    ".png": ("overlay_png", "image/png"),
    ".json": ("stats_json", "application/json"),
}


def _replay_events(emit, trace: dict[str, Any]) -> None:
    """Turn a finished trace into the named events the UI subscribes to.

    ``satquery.agent.events`` already defines these names and ``graph.py``
    emits them live; ``answer_query`` does not take an emitter, so rather than
    thread one through a synchronous call the same frames are derived from what
    the trace recorded. The payloads are the trace's own values, so the stream
    and the trace cannot disagree.
    """
    graded = trace.get("graded", {})
    if trace.get("router_path"):
        emit(
            "router",
            {
                "router_path": trace["router_path"],
                "task_selected": graded.get("task_selected"),
                "notes": trace.get("routing_notes", []),
            },
        )
    check = graded.get("parameter_check") or {}
    if check:
        emit(
            "validator",
            {"passed": check.get("passed"), "rejections": check.get("rejected", [])},
        )
    if graded.get("permitted_parameters"):
        emit("plan", {"plan": graded["permitted_parameters"]})
    for step in trace.get("steps", []):
        emit("step_started", {"tool": step.get("tool")})
        emit(
            "step_completed",
            {"tool": step.get("tool"), "result": step.get("outputs", {})},
        )
    agreement = trace.get("agreement")
    if agreement:
        emit(
            "agreement",
            {
                # The verdict, not the winner, used to be sent here. `verdict`
                # is "consistent" | "partial" | "conflict" | "single_modality"
                # while the contract types this field `Modality` -- "optical" |
                # "sar" | "unknown" -- so the console rendered "CONFLICT
                # TRUSTED" and, because neither column matched "conflict",
                # marked *both* sensors SUPERSEDED on the same panel.
                #
                # The trace's Agreement block carries no winner (its schema is
                # additionalProperties:false), so this is null unless one is
                # ever added. Null is what the console wants: it falls back to
                # the cause's `trusted_modality` from /meta/disagreement-causes,
                # which is the same 5-rule table this would otherwise duplicate.
                "winning_modality": agreement.get("winning_modality"),
                "explanation": agreement.get("disagreement_cause") or "",
            },
        )
    fusion = trace.get("fusion")
    if fusion:
        emit(
            "fusion",
            {
                "fused_answer": fusion.get("answer"),
                "confidence": fusion.get("confidence"),
            },
        )


def _write_preview(scene_path: Path, out: Path) -> bool:
    """Render a scene to an RGB PNG the browser can show.

    True colour when the file says which band is which, file order when it does
    not. A BigEarthNet chip stores its bands in wavelength order -- B02, B03,
    B04, so blue, green, red -- and taking the first three in file order mapped
    blue to the red channel, which is why vegetation rendered blue and the
    scenes looked like false-colour infrared. The band designations are already
    resolved for the index tools; this asks the same resolver rather than
    guessing.

    Where red/green/blue cannot all be resolved -- SAR, panchromatic, a band set
    with no visible bands at all -- it falls back to the first three in file
    order, which is honest about being an orientation aid and not a photograph.

    One stretch across all three channels, not one per channel. Per-channel was
    tried and is wrong here: a uniform-vegetation chip spans barely 400 DN
    (red p2 1164, p98 1574), and stretching each channel independently to full
    range turns that narrow spread into neon. A shared window keeps the ratio
    between bands, which is the thing that makes the colours read as colours.

    Orientation, not analysis: every measured claim comes from a tool, with its
    own overlay.
    """
    try:
        import numpy as np
        import rasterio
        from PIL import Image

        from satquery.ingest.bands import _match_descriptions

        with rasterio.open(scene_path) as handle:
            matched = _match_descriptions(handle.descriptions)
            rgb = [matched.get(name) for name in ("red", "green", "blue")]
            if all(band is not None for band in rgb):
                indexes, true_colour = rgb, True
            else:
                count = min(3, handle.count)
                indexes, true_colour = list(range(1, count + 1)), False
            data = handle.read(indexes).astype("float32")

        while data.shape[0] < 3:
            data = np.concatenate([data, data[-1:]], axis=0)
        stack = np.transpose(data[:3], (1, 2, 0))

        finite = stack[np.isfinite(stack)]
        if finite.size:
            low, high = np.percentile(finite, (2, 98))
            if high > low:
                stack = np.clip((stack - low) / (high - low), 0, 1)
                if true_colour:
                    # A linear stretch of surface reflectance renders dark,
                    # because reflectance is concentrated at the bottom of the
                    # window. 1/2.2 is the usual display gamma, and it is what
                    # separates field boundaries from the fields.
                    stack = np.power(stack, 1 / 2.2)
        image = Image.fromarray((np.nan_to_num(stack) * 255).astype("uint8"), "RGB")
        image.thumbnail((512, 512))
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out, "PNG", optimize=True)
        return True
    except Exception:  # noqa: BLE001 - a scene without a preview is still a scene
        return False


def _probe(path: Path) -> dict[str, Any]:
    """What ingest can tell us about an uploaded scene, or an honest blank.

    A file that cannot be opened is still a scene the user uploaded: it gets a
    record carrying the reason rather than a 500, so the compatibility panel can
    say what is wrong instead of the upload vanishing.
    """
    try:
        from satquery.ingest.scene import load_scene

        loaded = load_scene(path)
        scene, report = loaded.scene, loaded.ingest.compatibility
        height, width = scene.shape
        return {
            "role": loaded.ingest.modality,
            "modality": loaded.ingest.modality,
            "modality_source": loaded.ingest.modality_source,
            "width_px": int(width),
            "height_px": int(height),
            "crs": str(scene.crs) if scene.crs else None,
            "bands": sorted(scene.inventory.bands),
            "computable_indices": list(scene.inventory.computable_indices),
            "pixel_size_m": report.pixel_size_m,
            "native_gsd_m": report.native_gsd_m,
            "nodata_frac": round(report.nodata_frac, 6),
            "bit_depth": report.bit_depth,
            "warnings": list(loaded.warnings) + list(report.warnings),
            # The contract types these as required-but-nullable. The console
            # reads them straight off the scene, and an absent key is not the
            # same as an explicit null to JavaScript.
            #
            # `acquired_at` comes from the file's own tags when it carries one.
            # Nothing here invents an acquisition date from the upload time --
            # a wrong date on a bi-temporal pair silently reverses which scene
            # is "before".
            "acquired_at": _acquired_at(loaded),
            "bounds_wgs84": _bounds_wgs84(scene),
            # No tile pyramid and no vector footprint are produced in this
            # build, so both are null rather than a URL that would 404.
            "tile_url_template": None,
            "footprint_url": None,
            "compatibility": {
                "format_ok": report.format_ok,
                "crs_valid": report.crs_valid,
                # ScenePane reads the modality from here, not from the top
                # level, and uses crs_valid to decide whether to stamp "NO CRS".
                "modality": loaded.ingest.modality,
                "modality_source": loaded.ingest.modality_source,
                "nodata_frac": round(report.nodata_frac, 6),
            },
        }
    except Exception as exc:  # noqa: BLE001 - an unreadable upload is a state
        return {
            "role": None,
            "modality": None,
            "acquired_at": None,
            "bounds_wgs84": None,
            "tile_url_template": None,
            "footprint_url": None,
            "warnings": [f"could not read this file: {exc}"],
            "compatibility": {
                "format_ok": False,
                "crs_valid": False,
                "modality": None,
            },
        }


def _provenance(members: list[dict]) -> list[dict[str, Any]]:
    """The preparation steps that actually ran, one per scene.

    Read back from what ingest recorded at upload rather than written as the
    bundle is built: the point of a provenance panel is to say what was done to
    the pixels, and a list assembled here from intentions would agree with
    itself no matter what the pipeline did.
    """
    steps: list[dict[str, Any]] = []
    for member in members:
        params: dict[str, Any] = {}
        for key in ("modality", "modality_source", "crs", "bands", "pixel_size_m"):
            value = member.get(key)
            if value is not None:
                params[key] = value
        steps.append(
            {
                "stage": "p1_ingest",
                "op": "read_scene",
                "params": params,
                "at": member.get("uploaded_at") or _now(),
                "scene_id": member.get("scene_id"),
            }
        )
    return steps


def _bounds_wgs84(scene) -> list[float] | None:
    """Scene extent as [west, south, east, north] in degrees, or None.

    Reprojected from the scene's own CRS rather than assumed: the console
    places imagery on a lat/lon map, and handing it raw UTM metres puts a
    Sentinel tile somewhere off the coast of Africa. A scene with no CRS gets
    None, which the map already renders as "not georeferenced".
    """
    transform = getattr(scene, "transform", None)
    crs = getattr(scene, "crs", None)
    if transform is None or not crs:
        return None
    try:
        from rasterio.coords import BoundingBox
        from rasterio.warp import transform_bounds

        height, width = scene.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        box = BoundingBox(
            left=min(left, right),
            bottom=min(top, bottom),
            right=max(left, right),
            top=max(top, bottom),
        )
        west, south, east, north = transform_bounds(crs, "EPSG:4326", *box)
        if not all(map(_finite, (west, south, east, north))):
            return None
        return [round(west, 6), round(south, 6), round(east, 6), round(north, 6)]
    except Exception:  # noqa: BLE001 - an unprojectable scene is still a scene
        return None


def _finite(value: float) -> bool:
    import math

    return isinstance(value, (int, float)) and math.isfinite(value)


def _acquired_at(loaded) -> str | None:
    """Acquisition time from the file's own tags, never from the upload clock."""
    for holder in (getattr(loaded, "ingest", None), loaded):
        for attr in ("acquired_at", "datetime", "acquisition_date"):
            value = getattr(holder, attr, None)
            if value:
                return str(value)
    return None


def _task_support(pair_type: str) -> tuple[list[str], list[dict]]:
    """Which tasks a bundle supports, from the same table ``/meta/tasks`` uses.

    Derived rather than listed. ``TASK_LABELS`` already records what each task
    requires, so reading it here means the task list the UI shows and the
    supported set a bundle advertises cannot drift apart. Blocked tasks carry a
    reason: compatibility checking is a named deliverable, and an option that is
    silently missing teaches the user nothing.
    """
    from satquery.agent.task_enum import Task

    supported: list[str] = []
    blocked: list[dict] = []
    for task in Task:
        _label, _description, requires = TASK_LABELS.get(task.value, ("", "", ["single"]))
        if pair_type in requires:
            supported.append(task.value)
            continue
        if "bitemporal" in requires:
            reason = "needs two scenes of the same modality at different dates"
        elif "crossmodal" in requires:
            reason = "needs a co-registered optical and SAR pair"
        else:
            reason = f"not available for a {pair_type} bundle"
        blocked.append({"task": task.value, "reason": reason})
    return supported, blocked


def _pair_compatibility(members: list[dict]) -> dict[str, Any]:
    """Cross-scene checks. One scene is trivially consistent with itself."""
    crs_values = [m.get("crs") for m in members if m.get("crs")]
    common = crs_values[0] if crs_values and len(set(crs_values)) == 1 else None
    checks: list[str] = []
    if common and len(members) > 1:
        checks.append("crs_match")
    if members and all((m.get("compatibility") or {}).get("format_ok") for m in members):
        checks.append("format_ok")
    return {
        "coregistered": bool(common) if len(members) > 1 else True,
        "rmse_px": None,
        "correction_applied": False,
        "method": None,
        "common_crs": common,
        "checks_passed": checks,
    }


def _asset_georef(path: Path) -> dict[str, Any]:
    """``crs`` and ``bounds_wgs84`` for an evidence file, or explicit nulls.

    Both are required-but-nullable in the contract, and null is meaningful
    here: it tells the console to place the asset by pixel box rather than on
    the map. Only rasters are opened; a PNG overlay written without a world
    file has no georeferencing to report and says so.
    """
    blank: dict[str, Any] = {"crs": None, "bounds_wgs84": None}
    if path.suffix.lower() not in (".tif", ".tiff"):
        return blank
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(path) as handle:
            if not handle.crs:
                return blank
            west, south, east, north = transform_bounds(
                handle.crs, "EPSG:4326", *handle.bounds
            )
            if not all(map(_finite, (west, south, east, north))):
                return {"crs": str(handle.crs), "bounds_wgs84": None}
            return {
                "crs": str(handle.crs),
                "bounds_wgs84": [
                    round(west, 6),
                    round(south, 6),
                    round(east, 6),
                    round(north, 6),
                ],
            }
    except Exception:  # noqa: BLE001 - an unreadable asset is still an asset
        return blank


def _box_assets(
    assets: dict[str, dict[str, Any]],
    query_id: str,
    trace: dict[str, Any],
    members: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Grounding boxes as drawable assets -- one per producing tool.

    One asset per tool, not one per query, because the two producers mean
    different things and the answer text already says so:

      rs_ground_caption    the model's answer. Usually one box.
      object_box_fallback  connected-component candidates from classical
                           vision -- "not learned detection", typically 8-9.

    Merging them produced a single "9 box proposals" layer in which the one box
    the model actually chose was indistinguishable from eight blobs it had
    nothing to do with. Separate layers let the panel label and toggle them
    apart, which is the whole point of showing evidence rather than asserting a
    count.

    Boxes are emitted in the SOURCE raster's pixels. The console composites in
    pixel space for imagery that may carry no CRS at all -- a screenshot or a
    plain JPEG -- and the conversion from the tools' normalised 0-1 happens
    here, where the scene's dimensions are known.
    """
    scene = members[0] if members else {}
    width = int(scene.get("width_px") or 0)
    height = int(scene.get("height_px") or 0)
    if width <= 0 or height <= 0:
        return []

    #: The model's box is the answer, so it is drawn in the accent colour and
    #: listed first; proposals sit behind it in a muted one.
    palette = {"rs_ground_caption": "#dc4a1e", "object_box_fallback": "#8a8f7d"}
    out: list[dict[str, Any]] = []

    for step in trace.get("steps") or []:
        tool = step.get("tool") or "unknown"
        raw = (step.get("outputs") or {}).get("boxes") or []
        boxes: list[list[float]] = []
        for box in raw:
            if isinstance(box, dict):
                # `bbox_xyxy_normalised` first: it is the canonical 0-1 xyxy the
                # box module emits. `bbox_pixels` is **yxyx**, and reading that
                # as xyxy transposes every box -- on a 480x640 raster it pushed
                # x to 638, outside the image entirely.
                box = (
                    box.get("bbox_xyxy_normalised")
                    or box.get("bbox")
                    or []
                )
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            # Already-pixel boxes are left alone; normalised ones are scaled.
            # Mis-scaling here is the failure that put every VRSBench reference
            # in the top-left corner, so the test is on the values themselves
            # rather than on which tool produced them.
            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
            boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
        if not boxes:
            continue

        learned = tool == "rs_ground_caption"
        asset_id = f"as_{uuid.uuid4().hex[:8]}"
        record = {
            "asset_id": asset_id,
            "kind": "boxes",
            "label": (
                f"{len(boxes)} grounded box{'es' if len(boxes) != 1 else ''}"
                if learned
                else f"{len(boxes)} candidate{'s' if len(boxes) != 1 else ''} (classical vision)"
            ),
            "produced_by": tool,
            "media_type": "application/json",
            "bytes": 0,
            "query_id": query_id,
            "crs": scene.get("crs"),
            "bounds_wgs84": scene.get("bounds_wgs84"),
            "bbox_px": boxes,
            "colour": palette.get(tool, "#8a8f7d"),
            "tile_url_template": None,
            "overlay_url": None,
            "download_url": f"{API_PREFIX}/assets/{asset_id}?download=1",
            "boxes": boxes,
        }
        assets[asset_id] = record
        out.append(record if learned else dict(record))

    # The model's answer first, so it draws on top of the proposals.
    out.sort(key=lambda a: a["produced_by"] != "rs_ground_caption")
    return out


def _register_assets(
    assets: dict[str, dict[str, Any]],
    query_id: str,
    items: list[str],
    query_dir: Path,
) -> list[dict[str, Any]]:
    """Give each written evidence file an id and a URL the frontend can fetch.

    The pipeline records evidence as filesystem paths, which a browser cannot
    open. Serving the path verbatim would also hand out the server's directory
    layout, so each file gets an opaque id and is reachable only through the
    asset endpoints.
    """
    try:
        from satquery.tools.registry import ToolRegistry

        known_tools = set(ToolRegistry.default().names())
    except Exception:  # noqa: BLE001 - a registry that will not load is not fatal here
        known_tools = set()

    def _producing_tool(stem: str) -> str:
        """The tool that wrote this evidence file, from its name.

        The contract types this field as the **tool name**, and consumers match
        on it exactly: the disagreement panel picks the optical and radar masks
        by asking which tool produced them.

        This used to be ``stem.split("_")[0]`` -- the first token of the
        filename -- so ``texture_seg_mask.tif`` reported ``"texture"`` and
        ``sar_backscatter_overlay.png`` reported ``"sar"``. No registered tool
        has either name, so a panel matching ``spectral_index`` or
        ``sar_backscatter`` matched nothing at all, and the one view whose job
        is to show a cross-modal disagreement rendered "NO MASK" on both halves
        of a real conflict.

        Matched against the registry rather than parsed, so a tool whose name
        gains an underscore does not quietly break this again. ``fused`` has no
        tool behind it -- it is the fusion product -- and keeps its own name.
        """
        for suffix in ("_mask", "_overlay", "_boxes", "_stats"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem in known_tools:
            return stem
        # Longest registered name that prefixes the stem, so `spectral_index_ndvi`
        # still resolves to `spectral_index`.
        matches = [name for name in known_tools if stem.startswith(name)]
        return max(matches, key=len) if matches else stem

    out: list[dict[str, Any]] = []
    for item in items:
        path = Path(item)
        if not path.is_absolute():
            path = query_dir / path.name
        if not path.exists():
            continue
        kind, media_type = _ASSET_KINDS.get(
            path.suffix.lower(), ("evidence", "application/octet-stream")
        )
        asset_id = f"as_{uuid.uuid4().hex[:8]}"
        record = {
            "asset_id": asset_id,
            "kind": kind,
            "label": path.stem.replace("_", " "),
            "produced_by": _producing_tool(path.stem),
            "media_type": media_type,
            "bytes": path.stat().st_size,
            "query_id": query_id,
            "download_url": f"{API_PREFIX}/assets/{asset_id}?download=1",
            "overlay_url": (
                f"{API_PREFIX}/assets/{asset_id}" if kind == "overlay_png" else None
            ),
            # Georeferencing read from the written file, not copied from the
            # scene: a mask is only placeable on a map if the tool actually
            # wrote a CRS into it, and claiming the scene's CRS for a file that
            # has none would put the overlay somewhere confident and wrong.
            **_asset_georef(path),
            # No tile pyramid is built for evidence in this build. The contract
            # types this as required-and-nullable, and the console calls
            # resolveUrl on it -- an absent key threw
            # "Cannot read properties of undefined (reading 'startsWith')"
            # and took the whole workspace down with it.
            "tile_url_template": None,
            "path": str(path),
        }
        assets[asset_id] = record
        out.append({k: v for k, v in record.items() if k != "path"})
    return out


def build_app(
    upload_dir: Path | str | None = None,
    adapters: dict[str, str | None] | None = None,
):
    """A FastAPI app implementing the contract's meta and query endpoints.

    ``adapters`` maps a manifest tool name to an adapter directory -- for
    example ``{"rs_vqa": "/data/checkpoints/rs_vqa/adapter"}``. Anything passed
    here is registered as a runnable implementation, which is what stops the
    executor recording the tool as unavailable and answering from deterministic
    evidence alone.

    Passing nothing is a valid deployment: the CPU-only path runs the
    deterministic tools and says so. What it must not do is claim otherwise, so
    ``/meta/health`` reports exactly what was registered.
    """
    uploads = Path(upload_dir or "/tmp/satquery-uploads")
    uploads.mkdir(parents=True, exist_ok=True)

    loaded_adapters: list[str] = []
    if adapters:
        from satquery.tools.learned import register_learned_tools

        loaded_adapters = register_learned_tools(adapters)

    app = FastAPI(title="SatQuery AI", version="0.4.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if API_KEY is not None:

        @app.middleware("http")
        async def require_api_key(request, call_next):
            # Preflight carries no custom headers by definition, so demanding
            # the key here would block every cross-origin call before the real
            # request was ever made.
            if request.method == "OPTIONS" or request.url.path in _UNAUTHENTICATED_PATHS:
                return await call_next(request)
            if request.headers.get("X-Api-Key") != API_KEY:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "UNAUTHENTICATED",
                            "message": "This deployment requires a valid X-Api-Key header.",
                        }
                    },
                )
            return await call_next(request)

    #: Queries are held in memory. The contract has the frontend poll
    #: ``/queries/{id}`` after posting, so the result has to outlive the request
    #: -- but persisting them is a deployment decision, not a contract one.
    queries: dict[str, dict[str, Any]] = {}
    scenes: dict[str, dict[str, Any]] = {}
    bundles_store: dict[str, dict[str, Any]] = {}
    assets: dict[str, dict[str, Any]] = {}
    #: Per-query event log. ``answer_query`` is synchronous, so events are
    #: collected during the call and replayed when the client subscribes rather
    #: than pushed live. The frontend cannot tell the difference -- it reads
    #: named SSE frames either way -- and pretending to stream a computation
    #: that already finished would put fake timings in a graded trace.
    events: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    def _infer_kind(scene_ids: list[str]) -> str:
        """single / bitemporal / crossmodal, from what the bundle holds."""
        if len(scene_ids) < 2:
            return "single"
        modalities = {scenes[s].get("modality") for s in scene_ids if s in scenes}
        return "crossmodal" if len(modalities - {None}) > 1 else "bitemporal"

    @app.get(f"{API_PREFIX}/meta/health")
    def meta_health() -> dict[str, Any]:
        return _health(loaded_adapters)

    # The teammate branch exposes this at /health; both are served so neither
    # frontend build has to change.
    @app.get("/health")
    def health() -> dict[str, Any]:
        return _health(loaded_adapters)

    @app.get(f"{API_PREFIX}/meta/tasks")
    def meta_tasks() -> dict[str, Any]:
        return _tasks()

    @app.get(f"{API_PREFIX}/meta/tools")
    def meta_tools() -> dict[str, Any]:
        return _tools()

    @app.get(f"{API_PREFIX}/meta/disagreement-causes")
    def meta_causes() -> dict[str, Any]:
        return _disagreement_causes()

    def _register_scene(target: Path, filename: str) -> dict[str, Any]:
        """Probe a file already on disk into a scene record.

        Shared with the gallery, deliberately. A gallery item that took a
        different path into the store could differ from an uploaded one in
        modality, band inventory or pixel size -- and the entire point of the
        gallery is that it exercises the ordinary route.
        """
        scene_id = f"sc_{uuid.uuid4().hex[:8]}"
        record = {
            "scene_id": scene_id,
            "filename": Path(filename or target.name).name,
            "file": target.name,
            "path": str(target),
            "bytes": target.stat().st_size,
            "uploaded_at": _now(),
            "created_at": _now(),
            "status": "ready",
            "is_demo": False,
        }
        # Probed, not declared. Modality, band inventory and pixel size decide
        # which tasks this scene can support, so taking them from the file means
        # an upload cannot claim a capability it does not have.
        record.update(_probe(target))
        preview = uploads / "previews" / f"{scene_id}.png"
        if _write_preview(target, preview):
            record["preview_path"] = str(preview)
            record["preview_url"] = f"{API_PREFIX}/scenes/{scene_id}/preview.png"
        else:
            # A scene whose pixels will not render is still listed, with the
            # reason. Omitting the key entirely is what crashed the console.
            record["preview_url"] = None
            record.setdefault("warnings", []).append("no preview could be rendered")
        scenes[scene_id] = record
        return record

    @app.post(f"{API_PREFIX}/scenes")
    async def create_scene(file: UploadFile) -> dict[str, Any]:
        if not _is_image_upload(file.filename, file.content_type):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"{file.filename or 'the upload'} is not a raster image; "
                    "accepted formats: "
                    + ", ".join(sorted(UPLOAD_IMAGE_EXTENSIONS))
                ),
            )
        scene_id_hint = Path(file.filename or "scene.tif").name
        target = uploads / f"up_{uuid.uuid4().hex[:8]}_{scene_id_hint}"
        # Streamed in chunks against a ceiling rather than read whole. The
        # endpoint is unauthenticated, so `await file.read()` put an entire
        # upload of any size into the container's memory before anything
        # looked at it -- one large POST was enough to take the process down.
        # The frontend's own 4 GB limit is advisory; a limit that only exists
        # in the client is not a limit.
        written = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"upload exceeds the {MAX_UPLOAD_BYTES} byte limit; "
                                "raise SATQUERY_MAX_UPLOAD_BYTES to accept more"
                            ),
                        )
                    handle.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        record = _register_scene(target, file.filename or target.name)
        return {k: v for k, v in record.items() if k not in ("path", "preview_path")}

    @app.get(f"{API_PREFIX}/scenes/{{scene_id}}/preview.png")
    def scene_preview(scene_id: str):
        from fastapi.responses import FileResponse

        record = scenes.get(scene_id)
        preview = (record or {}).get("preview_path")
        if not preview or not Path(preview).exists():
            raise HTTPException(status_code=404, detail="no preview for this scene")
        return FileResponse(preview, media_type="image/png")

    @app.get(f"{API_PREFIX}/scenes")
    def list_scenes() -> dict[str, Any]:
        return {
            "scenes": [
                {k: v for k, v in s.items() if k not in ("path", "preview_path")}
                for s in scenes.values()
            ]
        }

    # ------------------------------------------------------------- gallery
    # A fixed set of items drawn from the *test* splits of the benchmarks the
    # adapters were scored on, so anyone can pick one and see the system answer
    # a question whose gold answer is published. It lives on the Volume rather
    # than in the image: 53 MB of imagery in the payload would slow every
    # deploy, and a gallery that can be replaced without redeploying is the
    # point.
    gallery_root = Path(os.environ.get("SATQUERY_GALLERY", "/data/gallery"))

    def _reload_gallery_volume() -> None:
        """Pick up writes made to the Volume from outside this container.

        A Modal Volume is snapshotted when the container starts; anything
        written to it afterwards -- by `modal volume put`, which is how the
        gallery is published -- stays invisible until the mount is reloaded.
        The symptom is quietly wrong rather than broken: the endpoint keeps
        serving the manifest that existed at boot, so a freshly verified
        gallery reports the counts it had hours ago and nothing errors.

        Best-effort. Off Modal there is no volume to reload, and a failure here
        should degrade to serving what is already mounted rather than take the
        endpoint down.
        """
        try:
            import modal

            modal.Volume.from_name("satquery-data").reload()
        except Exception:  # noqa: BLE001 - not on Modal, or no permission
            pass

    def _gallery_manifest() -> dict[str, Any]:
        path = gallery_root / "manifest.json"
        if not path.exists():
            _reload_gallery_volume()
        if not path.exists():
            return {"items": [], "available": False, "reason": f"no manifest at {path}"}
        data = json.loads(path.read_text(encoding="utf-8"))
        data["available"] = True
        return data

    def _gallery_item(item_id: str) -> dict[str, Any]:
        for item in _gallery_manifest().get("items", []):
            if item.get("item_id") == item_id:
                return item
        raise HTTPException(status_code=404, detail=f"no gallery item {item_id!r}")

    def _public_item(item: dict[str, Any]) -> dict[str, Any]:
        """One item as the console sees it: local paths swapped for URLs.

        ``gold`` and ``benchmark_answer`` stay in. The gallery's claim is that
        the served answer matches a published one, and a viewer who cannot see
        what it is being matched against has to take that on trust.
        """
        # `account` is dropped: it names the Modal workspace the imagery was
        # staged in, which identifies the operator's own accounts and has no
        # bearing on whether the answer is right. Older manifests still carry
        # it, so this strips rather than assuming it is absent.
        public = {k: v for k, v in item.items() if k not in ("images", "account")}
        public["images"] = [
            {
                "role": image.get("role"),
                "preview_url": f"{API_PREFIX}/gallery/previews/{Path(image['preview']).name}",
            }
            for image in item.get("images", [])
            if image.get("preview")
        ]
        return public

    @app.get(f"{API_PREFIX}/gallery")
    def gallery_index(refresh: bool = False) -> dict[str, Any]:
        # `?refresh=1` after publishing a new manifest, rather than reloading on
        # every read: the reload is a network round trip to the volume service
        # and the gallery is otherwise static between publishes.
        if refresh:
            _reload_gallery_volume()
        manifest = _gallery_manifest()
        return {
            "available": manifest.get("available", False),
            "counts": manifest.get("counts", {}),
            "statuses": manifest.get("statuses", {}),
            "verification": manifest.get("verification"),
            "items": [_public_item(item) for item in manifest.get("items", [])],
        }

    @app.get(f"{API_PREFIX}/gallery/previews/{{name}}")
    def gallery_preview(name: str):
        from fastapi.responses import FileResponse

        # Resolved and re-rooted before use: `name` reaches this from the URL,
        # and a `..` in it would otherwise read any file the container can see.
        target = (gallery_root / "previews" / Path(name).name).resolve()
        if not target.is_file() or gallery_root.resolve() not in target.parents:
            raise HTTPException(status_code=404, detail="no such preview")
        return FileResponse(target, media_type="image/png")

    @app.post(f"{API_PREFIX}/gallery/{{item_id}}/bundle")
    def gallery_bundle(item_id: str) -> dict[str, Any]:
        """Load one gallery item and hand back a bundle ready to be asked.

        Server-side, because the imagery is already on the Volume: making the
        browser download half a megabyte of GeoTIFF only to upload it again
        would be slower and would prove nothing extra. The scenes are registered
        through ``_register_scene``, the same probe an upload goes through.
        """
        item = _gallery_item(item_id)
        prepared: list[dict[str, str]] = []
        for image in item.get("images", []):
            source = gallery_root / "images" / item["segment"] / Path(image["path"]).name
            if not source.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"gallery imagery missing on the volume: {source.name}",
                )
            record = _register_scene(source, source.name)
            prepared.append(
                {"scene_id": record["scene_id"], "role": image.get("role") or "primary"}
            )

        # Built by the ordinary endpoint, not assembled here.
        #
        # The first version of this wrote its own record and got the shape
        # wrong: it invented `kind` where the contract says `pair_type`, and
        # omitted `label`, `prep_ms`, `provenance`, `supported_tasks`,
        # `blocked_tasks` and `bounds_wgs84` entirely. Every one of those is
        # required in `frontend/contracts/types.ts`, and a missing required
        # field does not degrade the console -- it blanks it. Opening a gallery
        # item landed on an empty white workspace.
        #
        # Reusing the endpoint means the gallery cannot drift from the contract
        # again, and it is the same reasoning as `_register_scene`: a second
        # path into the store is a second thing to keep correct.
        bundle = create_bundle({"scenes": prepared})
        bundle["gallery_item"] = item_id
        bundles_store[bundle["bundle_id"]]["gallery_item"] = item_id
        return {
            "bundle_id": bundle["bundle_id"],
            "question": item["question"],
            "item": _public_item(item),
        }

    @app.get(f"{API_PREFIX}/bundles")
    def list_bundles() -> dict[str, Any]:
        return {"bundles": list(bundles_store.values())}

    @app.post(f"{API_PREFIX}/bundles")
    def create_bundle(body: dict[str, Any]) -> dict[str, Any]:
        """Group prepared scenes so queries can name one id instead of paths.

        Preparation is synchronous here -- ``answer_query`` loads and validates
        the scenes itself -- so a bundle is a named set rather than a job. The
        contract's ``/jobs`` polling exists for the async path; reporting
        ``ready`` immediately is honest about which one this is.
        """
        # The contract shape is `scenes: [{scene_id, role}]` -- the console
        # assigns a role per scene (t0/t1, optical/sar) as part of preparing a
        # bundle, so the role travels with the id. `scene_ids: [...]` is the
        # flat form and is accepted too, because scripts and curl use it.
        #
        # Reading only the flat form is what produced a bundle with zero scenes:
        # the POST succeeded, `status` came back `ready`, and the console sat on
        # PREPARING forever waiting for scenes that were never attached. Nothing
        # errored, which is why the route verifier missed it -- it sent its own
        # shape rather than the frontend's.
        roles: dict[str, str] = {}
        scene_ids: list[str] = []
        for entry in body.get("scenes") or []:
            if isinstance(entry, dict) and entry.get("scene_id"):
                scene_ids.append(entry["scene_id"])
                if entry.get("role"):
                    roles[entry["scene_id"]] = entry["role"]
            elif isinstance(entry, str):
                scene_ids.append(entry)
        scene_ids.extend(s for s in (body.get("scene_ids") or []) if s not in scene_ids)

        if not scene_ids:
            raise HTTPException(
                status_code=400,
                detail="a bundle needs at least one scene; send "
                "scenes=[{scene_id, role}] or scene_ids=[...]",
            )
        unknown = [s for s in scene_ids if s not in scenes]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown scenes: {unknown}")
        bundle_id = f"bn_{uuid.uuid4().hex[:6]}"
        members = []
        for scene_id in scene_ids:
            member = dict(scenes[scene_id])
            # The caller's role wins over the probed modality: the console knows
            # which of a pair is t0 and which is t1, and ingest cannot.
            if scene_id in roles:
                member["role"] = roles[scene_id]
            members.append(member)
        pair_type = body.get("pair_type") or _infer_kind(scene_ids)
        supported, blocked = _task_support(pair_type)
        record = {
            "bundle_id": bundle_id,
            "label": body.get("label") or bundle_id,
            "created_at": _now(),
            "pair_type": pair_type,
            "status": "ready",
            # Preparation is synchronous here, so there is no elapsed time to
            # report. Zero rather than a plausible number: the UI shows this
            # beside a measured query latency, and a fabricated figure sitting
            # next to a real one is worse than an obvious zero.
            "prep_ms": 0,
            "scenes": [
                {k: v for k, v in m.items() if k not in ("path", "preview_path")}
                for m in members
            ],
            "scene_ids": scene_ids,
            "pair_compatibility": _pair_compatibility(members),
            "supported_tasks": supported,
            "blocked_tasks": blocked,
            # P1-P4 as they actually ran. The contract types these as required
            # arrays, and the console reads `bundle.provenance.length` without a
            # guard, so omitting them blanked the workspace on mount.
            #
            # Only stages that really executed are listed. This build ingests at
            # upload and does no SAR normalisation, co-registration or tiling,
            # so P2-P4 are absent rather than reported as no-ops -- a
            # provenance panel showing a step that never ran is worse than one
            # showing three steps honestly.
            "provenance": _provenance(members),
            # Null, not a fabricated count: nothing tiles a scene in this build,
            # and `tile_count: 0` would read as "tiled into zero tiles".
            "tiles": None,
            "warnings": sorted({w for m in members for w in m.get("warnings", [])}),
            "bounds_wgs84": next(
                (m["bounds_wgs84"] for m in members if m.get("bounds_wgs84")), None
            ),
        }
        bundles_store[bundle_id] = record
        return record

    @app.get(f"{API_PREFIX}/bundles/{{bundle_id}}")
    def get_bundle(bundle_id: str) -> dict[str, Any]:
        if bundle_id not in bundles_store:
            raise HTTPException(status_code=404, detail="unknown bundle")
        return bundles_store[bundle_id]

    @app.get(f"{API_PREFIX}/demo/bundles")
    def demo_bundles() -> dict[str, Any]:
        """Pre-cached scenes for the offline demo (plan section 4.11).

        Empty rather than absent. A 404 here reads to the frontend as a broken
        server and it retries in a loop; an empty list reads as "no demo bundles
        prepared", which is both true and what the landing screen already
        renders.
        """
        return {"bundles": [b for b in bundles_store.values() if b.get("demo")]}

    @app.post(f"{API_PREFIX}/queries")
    def create_query(body: dict[str, Any]) -> dict[str, Any]:
        from satquery.agent.pipeline import answer_query

        question = (body.get("question") or body.get("query_text") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        # A bundle is how the frontend addresses scenes -- it uploads, bundles,
        # then asks questions of the bundle. Resolving it here rather than
        # making the client unpack it keeps the pairing decisions (which scene
        # is t0, which modality each carries) on the server that made them.
        bundle_id = body.get("bundle_id")
        scene_ids = list(body.get("scene_ids") or body.get("images") or [])
        if bundle_id and not scene_ids:
            bundle = bundles_store.get(bundle_id)
            if bundle is None:
                raise HTTPException(status_code=404, detail=f"no bundle {bundle_id!r}")
            scene_ids = list(bundle.get("scene_ids") or [])

        paths = [scenes[s]["path"] if s in scenes else s for s in scene_ids]
        if not paths:
            raise HTTPException(status_code=400, detail="at least one scene is required")

        # Modality and date come from what ingest probed at upload, not from
        # the client. A caller that mislabels a SAR scene as optical would
        # otherwise route the query to tools that cannot read it, and the
        # refusal would name the wrong cause.
        modalities = body.get("modalities")
        if modalities is None:
            probed = [scenes[s].get("modality") for s in scene_ids if s in scenes]
            if probed and all(m for m in probed):
                modalities = probed

        query_id = f"qr_{uuid.uuid4().hex[:8]}"
        # Each query writes its masks and overlays into its own directory. The
        # pipeline only exports evidence when it has somewhere to put it, so
        # omitting this is why earlier traces carried no evidence at all -- the
        # UI's evidence panel had nothing to render and the PS's "return visual
        # evidence" requirement went unmet by omission rather than by design.
        query_dir = uploads / "queries" / query_id
        query_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        collected: list[tuple[str, dict[str, Any]]] = []

        def emit(name: str, payload: dict[str, Any]) -> None:
            collected.append((name, payload))

        emit("accepted", {"query_id": query_id, "queued_ms": 0})
        try:
            outcome = answer_query(
                question,
                paths,
                modalities=modalities,
                dates=body.get("dates"),
                output_dir=query_dir,
                write_evidence=True,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a query state
            # A failure is a state the frontend renders, not a 500 it cannot
            # show: "a refusal is a feature, not an error" applies to breakage
            # as well as to refusals.
            record = {
                "query_id": query_id,
                # The SSE endpoint for this query. The console opens an
                # EventSource only when the create response carries one
                # (`enabled: Boolean(streamUrl)`), so without this the chat panel
                # sat on "validating inputs" forever -- while the query itself had
                # already finished in 52 ms. Answering synchronously does not
                # remove the need for the URL: the events are replayed from the
                # recorded trace, and that replay is what draws the step timeline.
                "stream_url": f"{API_PREFIX}/queries/{query_id}/events",
                "bundle_id": bundle_id,
                "question": question,
                "created_at": _now(),
                "state": "failed",
                "answer": None,
                "confidence": None,
                "latency_ms": int((time.time() - started) * 1000),
                "refusal": {"reason": str(exc), "fix": "check the inputs and retry"},
                "evidence": [],
                "warnings": [],
                "trace": None,
            }
            emit("error", {"query_id": query_id, "reason": str(exc)})
            events[query_id] = collected
            queries[query_id] = record
            return record

        trace = getattr(outcome, "trace", None) or {}
        record_answer = getattr(outcome, "answer", None)
        record_confidence = getattr(outcome, "confidence", None)
        _replay_events(emit, trace)
        evidence = _register_assets(
            assets, query_id, trace.get("evidence", []), query_dir
        )
        # The scenes this query actually ran on, for the pixel dimensions the
        # box conversion needs. `members` in create_bundle is a different local.
        queried = [scenes[s] for s in scene_ids if s in scenes]
        evidence.extend(_box_assets(assets, query_id, trace, queried))
        trace["evidence"] = [item["asset_id"] for item in evidence]
        record = {
            "query_id": query_id,
            # The SSE endpoint for this query. The console opens an
            # EventSource only when the create response carries one
            # (`enabled: Boolean(streamUrl)`), so without this the chat panel
            # sat on "validating inputs" forever -- while the query itself had
            # already finished in 52 ms. Answering synchronously does not
            # remove the need for the URL: the events are replayed from the
            # recorded trace, and that replay is what draws the step timeline.
            "stream_url": f"{API_PREFIX}/queries/{query_id}/events",
            "bundle_id": bundle_id,
            "question": question,
            "created_at": _now(),
            "state": "succeeded",
            "answer": record_answer,
            "confidence": record_confidence,
            "confidence_basis": getattr(outcome, "confidence_basis", "heuristic"),
            "latency_ms": int((time.time() - started) * 1000),
            "refusal": getattr(outcome, "refusal", None),
            "evidence": evidence,
            "warnings": trace.get("warnings", []),
            "trace": trace,
        }
        # The answer itself. The console builds a turn's text from the stream,
        # not from the POST body: `fusion` sets `answer`, `confidence` and
        # `confidence_basis`, and without it the turn rendered its latency and
        # trace link above an empty answer.
        #
        # `fusion` and not a run of `token` frames: tokens are streaming
        # deltas, and this pipeline answers synchronously in ~40 ms. Emitting
        # them would draw a typing animation for text that was already
        # complete -- theatre the trace would then contradict.
        emit(
            "fusion",
            {
                "answer": record_answer,
                "confidence": record_confidence,
                "confidence_basis": getattr(outcome, "confidence_basis", "heuristic"),
            },
        )
        # Full asset records, not ids. The console feeds this array straight
        # into its layer store and then reads `asset.download_url` off each
        # entry -- handing it strings threw "Cannot read properties of
        # undefined (reading 'startsWith')" inside resolveUrl and blanked the
        # workspace the moment a query produced its first piece of evidence.
        emit("evidence", {"assets": evidence})
        emit("done", {"query_id": query_id, "state": "succeeded"})
        events[query_id] = collected
        queries[query_id] = record
        return record

    @app.get(f"{API_PREFIX}/queries")
    def list_queries(bundle_id: str | None = None) -> dict[str, Any]:
        """Past queries, newest first, optionally for one bundle.

        The frontend asks for ``/queries?bundle_id=...`` after every successful
        ask, to refresh the history panel. Without this route that call fell
        through to the ``/queries/{query_id}`` matcher and came back as a query
        that does not exist, so the panel stayed empty however many questions
        were asked.
        """
        rows = list(queries.values())
        if bundle_id:
            rows = [r for r in rows if r.get("bundle_id") == bundle_id]
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return {"queries": rows}

    @app.get(f"{API_PREFIX}/queries/{{query_id}}/events")
    def query_events(query_id: str):
        """Replay a query's events as SSE, in the frame shape EventSource wants.

        Named frames, ``id:`` on each so ``Last-Event-ID`` resumption works, and
        a terminal ``done`` or ``error`` so the client stops listening instead
        of holding a connection open. Section 3 has the frontend fall back to
        polling after two failed attempts, which is why every field this streams
        is also readable from ``GET /queries/{id}``.
        """
        from fastapi.responses import StreamingResponse

        if query_id not in events:
            raise HTTPException(status_code=404, detail="unknown query")

        def frames():
            # Built from parts rather than one multi-line f-string: SSE frames
            # are newline-delimited, and embedding those literally is how the
            # separator gets mangled by whatever writes this file next.
            for index, (name, payload) in enumerate(events[query_id]):
                yield (
                    f"id: {index}\n"
                    f"event: {name}\n"
                    f"data: {json.dumps(payload)}\n"
                    "\n"
                )

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(f"{API_PREFIX}/assets/{{asset_id}}")
    def get_asset(asset_id: str):
        from fastapi.responses import FileResponse

        record = assets.get(asset_id)
        if record is None or not Path(record["path"]).exists():
            raise HTTPException(status_code=404, detail="unknown asset")
        return FileResponse(record["path"], media_type=record["media_type"])

    @app.get(f"{API_PREFIX}/assets/{{asset_id}}/meta")
    def get_asset_meta(asset_id: str) -> dict[str, Any]:
        record = assets.get(asset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown asset")
        return {k: v for k, v in record.items() if k != "path"}

    @app.get(f"{API_PREFIX}/queries/{{query_id}}")
    def get_query(query_id: str) -> dict[str, Any]:
        if query_id not in queries:
            raise HTTPException(status_code=404, detail="unknown query")
        return queries[query_id]

    @app.get(f"{API_PREFIX}/queries/{{query_id}}/trace")
    def get_trace(query_id: str) -> dict[str, Any]:
        if query_id not in queries:
            raise HTTPException(status_code=404, detail="unknown query")
        trace = queries[query_id].get("trace")
        if trace is None:
            raise HTTPException(status_code=404, detail="no trace for this query")
        return trace

    @app.get(f"{API_PREFIX}/queries/{{query_id}}/evidence")
    def get_evidence(query_id: str) -> dict[str, Any]:
        if query_id not in queries:
            raise HTTPException(status_code=404, detail="unknown query")
        return {"evidence": queries[query_id].get("evidence", [])}

    @app.post(f"{API_PREFIX}/queries/{{query_id}}/cancel")
    def cancel_query(query_id: str) -> dict[str, Any]:
        if query_id not in queries:
            raise HTTPException(status_code=404, detail="unknown query")
        # Answering is synchronous here, so by the time a client can cancel the
        # query has already finished. Reported honestly rather than pretended.
        return {"query_id": query_id, "state": queries[query_id]["state"],
                "cancelled": False,
                "detail": "queries complete synchronously; nothing to cancel"}

    return app
