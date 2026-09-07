import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from satquery.agent.bundle import ImageBundle, ImageRef
from satquery.agent.graph import QueryResult, run_query
from satquery.agent.probe import probe_bundle
from satquery.fusion.reconcile import DISAGREEMENT_CAUSES
from satquery.ingest.band_inventory import BandInventory
from satquery.tools.registry import ToolRegistry

app = FastAPI(title="SatQuery AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores for bundles and queries
_BUNDLE_STORE: dict[str, ImageBundle] = {}
_QUERY_STORE: dict[str, QueryResult] = {}
# §24: set while a query runs so the executor can stop at a wave boundary.
# Present here rather than in the graph because cancellation is a transport
# concern; the graph only consults the flag.
_CANCEL_FLAGS: dict[str, threading.Event] = {}


def register_bundle(bundle: ImageBundle) -> None:
    _BUNDLE_STORE[bundle.bundle_id] = bundle


def _get_or_create_bundle(bundle_id: str) -> ImageBundle:
    if bundle_id in _BUNDLE_STORE:
        return _BUNDLE_STORE[bundle_id]

    # Default fallback demo bundle
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
        scene_id="scene_default",
        path="default.tif",
        modality="optical",
        crs="EPSG:32644",
        computable_indices=["NDVI", "NDWI"],
    )
    bundle = ImageBundle(
        bundle_id=bundle_id,
        images=[img],
        band_inventory=inv,
        pair_type="single",
    )
    _BUNDLE_STORE[bundle_id] = bundle
    return bundle


class QuerySubmitRequest(BaseModel):
    question: str
    allow_llm_router: bool = True


class CreateQueryRequest(BaseModel):
    bundle_id: str
    question: str
    allow_llm_router: bool = True


@app.get("/health")
def health_endpoint() -> dict[str, Any]:
    from satquery.serving.client import get_serving_client

    return get_serving_client().get_health()


@app.get("/meta/tools")
def get_meta_tools() -> list[dict[str, Any]]:
    reg = ToolRegistry.default()
    return [asdict(reg.get(name)) for name in reg.names()]


@app.get("/meta/disagreement-causes")
def get_meta_disagreement_causes() -> dict[str, Any]:
    fixture_path = (
        Path(__file__).parent.parent.parent
        / "frontend"
        / "mocks"
        / "fixtures"
        / "disagreement_causes.json"
    )
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    causes = [
        {"code": k, "trusted_modality": v["trusted_modality"], "explanation": v["explanation"]}
        for k, v in DISAGREEMENT_CAUSES.items()
    ]
    return {"causes": causes}


@app.get("/bundles/{bundle_id}/probe")
def probe_bundle_endpoint(bundle_id: str) -> dict[str, Any]:
    bundle = _get_or_create_bundle(bundle_id)
    supported, blocked = probe_bundle(bundle)
    return {
        "supported_tasks": supported,
        "blocked_tasks": blocked,
        "active_modalities": [img.modality for img in bundle.images],
        "warnings": bundle.warnings or [],
    }


@app.get("/queries/{query_id}")
def get_query_endpoint(query_id: str) -> dict[str, Any]:
    if query_id not in _QUERY_STORE:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found")
    res = _QUERY_STORE[query_id]
    return asdict(res)


@app.post("/queries/{query_id}/cancel")
def cancel_query_endpoint(query_id: str) -> dict[str, Any]:
    """Request cancellation (§24). Cooperative: takes effect at the next wave."""
    flag = _CANCEL_FLAGS.get(query_id)
    if flag is not None:
        flag.set()
        return {"status": "cancelling", "query_id": query_id}
    if query_id in _QUERY_STORE:
        # Already finished; nothing left to stop.
        return {"status": _QUERY_STORE[query_id].state, "query_id": query_id}
    raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found")



def _find_asset(asset_id: str) -> Any | None:
    """Locate a produced asset across completed queries."""
    for result in _QUERY_STORE.values():
        for asset in result.evidence or []:
            if getattr(asset, "asset_id", None) == asset_id:
                return asset
    return None


@app.get("/assets/{asset_id}")
def get_asset_endpoint(asset_id: str) -> Response:
    """Serve a produced asset by id (§18: download_url points here)."""
    asset = _find_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")
    source = (asset.stats or {}).get("source_path") if asset.stats else None
    if not source or not Path(source).is_file():
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' has no stored file")
    return Response(
        content=Path(source).read_bytes(),
        media_type=asset.media_type or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{Path(source).name}"'},
    )


@app.get("/assets/{asset_id}/meta")
def get_asset_meta_endpoint(asset_id: str) -> dict[str, Any]:
    asset = _find_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")
    return asdict(asset)


@app.get("/queries/{query_id}/trace")
def get_query_trace_endpoint(query_id: str) -> dict[str, Any]:
    if query_id in _QUERY_STORE and _QUERY_STORE[query_id].trace:
        return _QUERY_STORE[query_id].trace
    import os
    trace_dir = Path(os.environ.get("SATQUERY_TRACES_DIR", "traces"))
    trace_file = trace_dir / f"{query_id}.json"
    if trace_file.exists():
        return json.loads(trace_file.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail=f"Trace for query '{query_id}' not found")


async def _process_query_submission(
    bundle_id: str,
    question: str,
    allow_llm_router: bool,
    accept: str | None,
) -> Response:
    bundle = _get_or_create_bundle(bundle_id)
    query_id = str(uuid.uuid4())
    submitted_at = time.perf_counter()

    is_sse = accept is not None and "text/event-stream" in accept

    if is_sse:
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def sink(event: str, data: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        cancel_flag = threading.Event()
        _CANCEL_FLAGS[query_id] = cancel_flag

        def worker() -> QueryResult:
            try:
                res = run_query(
                    bundle=bundle,
                    question=question,
                    query_id=query_id,
                    allow_llm_router=allow_llm_router,
                    emit=sink,
                    cancel_check=cancel_flag,
                    submitted_at=submitted_at,
                )
                _QUERY_STORE[query_id] = res
                return res
            finally:
                _CANCEL_FLAGS.pop(query_id, None)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(asyncio.to_thread(worker))

        async def event_generator() -> AsyncGenerator[str, None]:
            event_idx = 0
            while True:
                item = await queue.get()
                if item is None:
                    break
                ev, dt = item
                event_idx += 1
                data_str = json.dumps(dt)
                yield f"id: {query_id}_{event_idx}\nevent: {ev}\ndata: {data_str}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming JSON response
    sync_flag = threading.Event()
    _CANCEL_FLAGS[query_id] = sync_flag
    try:
        result = run_query(
            bundle=bundle,
            question=question,
            query_id=query_id,
            allow_llm_router=allow_llm_router,
            cancel_check=sync_flag,
            submitted_at=submitted_at,
        )
    finally:
        _CANCEL_FLAGS.pop(query_id, None)
    _QUERY_STORE[query_id] = result

    result_dict = asdict(result)

    if result.state == "refused":
        refusal = result.refusal or {}
        if refusal.get("category") in ("validator", "parameter_gate"):
            code = (
                "QUERY_REFUSED"
                if refusal.get("category") == "validator"
                else "PARAM_REJECTED"
            )
            error_env = {
                "error": {
                    "code": code,
                    "message": refusal.get("reason", "Query refused"),
                    "details": refusal,
                },
                "result": result_dict,
            }
            return JSONResponse(status_code=422, content=error_env)
    elif result.state == "failed":
        error_env = {
            "error": {
                "code": "INTERNAL",
                "message": result.answer or "Query execution failed",
                "details": {"failures": result.failures},
            },
            "result": result_dict,
        }
        return JSONResponse(status_code=500, content=error_env)

    return JSONResponse(status_code=200, content=result_dict)


@app.get("/queries")
def list_queries_endpoint(bundle_id: str | None = None) -> list[dict[str, Any]]:
    results = list(_QUERY_STORE.values())
    if bundle_id:
        results = [r for r in results if r.bundle_id == bundle_id]
    return [asdict(r) for r in results]


@app.post("/queries")
async def create_query_endpoint(
    payload: CreateQueryRequest,
    accept: str | None = Header(default=None),
) -> Response:
    return await _process_query_submission(
        bundle_id=payload.bundle_id,
        question=payload.question,
        allow_llm_router=payload.allow_llm_router,
        accept=accept,
    )


@app.post("/bundles/{bundle_id}/queries")
async def submit_query_endpoint(
    bundle_id: str,
    payload: QuerySubmitRequest,
    accept: str | None = Header(default=None),
) -> Response:
    return await _process_query_submission(
        bundle_id=bundle_id,
        question=payload.question,
        allow_llm_router=payload.allow_llm_router,
        accept=accept,
    )
