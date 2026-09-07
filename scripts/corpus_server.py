"""A web browser over a whole corpus, served from the Modal Volume.

The sampled review page shows 51 rows. That is enough to prove the pipeline is
sound and not enough to judge a dataset. This serves **every** row, with
filtering, so a suspicion about one question type can be checked against
hundreds of examples instead of two.

Read-only by design: it opens the manifest and the images and serves them. It
cannot edit the corpus, so browsing it can never damage what trains.

Paged, not dumped. 74,000 rows with images is gigabytes; the page fetches 24 at
a time and the images stream individually from the Volume.
"""

import json
import os
from pathlib import Path

MANIFEST = os.environ.get("CORPUS_MANIFEST", "/data/manifests/rs_vqa_train.jsonl")
MANIFEST_DIR = Path(os.environ.get("CORPUS_MANIFEST_DIR", "/data/manifests"))
IMAGE_ROOT = Path(os.environ.get("CORPUS_IMAGE_ROOT", "/data"))

#: One cache entry per manifest, not one global list. Four adapters share this
#: server, and a single ``_ROWS`` would serve whichever corpus happened to be
#: requested first for every request after it.
_ROWS: dict[str, list[dict]] = {}


def manifests() -> list[str]:
    """Every corpus on the Volume, newest first."""
    if not MANIFEST_DIR.is_dir():
        return [MANIFEST] if Path(MANIFEST).exists() else []
    found = sorted(
        (p for p in MANIFEST_DIR.glob("*.jsonl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [str(p) for p in found]


def resolve(name: str) -> str:
    """Map a requested manifest to a real one, refusing anything undiscovered.

    The name arrives from a query string, so it is treated as untrusted: only a
    path this server already listed is openable. Without the membership check
    this endpoint would read any file on the Volume by absolute path.
    """
    available = manifests()
    if not name:
        return available[0] if available else MANIFEST
    candidates = [m for m in available if m == name or Path(m).name == name]
    if not candidates:
        raise KeyError(name)
    return candidates[0]


def rows(name: str = "") -> list[dict]:
    """One manifest, loaded once per container."""
    path = resolve(name)
    if path not in _ROWS:
        with open(path, encoding="utf-8") as handle:
            _ROWS[path] = [json.loads(line) for line in handle if line.strip()]
    return _ROWS[path]


def source_of(row: dict) -> str:
    """The dataset a row came from, under either field name.

    ``merge_corpus`` writes ``corpus_source`` while the canonical schema and the
    per-adapter generators write ``source``. Reading only the first made every
    change_vqa row show up as "?" and filter to nothing.
    """
    return row.get("corpus_source") or row.get("source") or "?"


def type_of(row: dict) -> str:
    """The question type, as ``question_type`` or as the canonical ``task``."""
    return row.get("question_type") or row.get("task") or "?"


def build_app():
    import io

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    api = FastAPI(title="Corpus Browser")

    @api.get("/api/meta")
    def meta(manifest: str = ""):
        try:
            path = resolve(manifest)
        except KeyError as err:
            raise HTTPException(404, f"no manifest named {manifest!r}") from err
        data = rows(manifest)
        answers = {}
        for row in data:
            answers.setdefault(type_of(row), set()).add(str(row.get("answer")))
        return {
            "total": len(data),
            "sources": sorted({source_of(r) for r in data}),
            "types": sorted({type_of(r) for r in data}),
            # How many distinct answers a question type admits is the single
            # most useful number when judging a new corpus: a type with one
            # answer is a constant the model can score on without looking.
            "distinct_answers": {k: len(v) for k, v in sorted(answers.items())},
            "manifest": path,
            "manifests": [Path(m).name for m in manifests()],
        }

    @api.get("/api/rows")
    def api_rows(
        offset: int = 0,
        limit: int = Query(24, le=100),
        source: str = "",
        qtype: str = "",
        contains: str = "",
        manifest: str = "",
    ):
        try:
            data = rows(manifest)
        except KeyError as err:
            raise HTTPException(404, f"no manifest named {manifest!r}") from err
        if source:
            data = [r for r in data if source_of(r) == source]
        if qtype:
            data = [r for r in data if type_of(r) == qtype]
        if contains:
            needle = contains.lower()
            data = [
                r
                for r in data
                if needle in r["question"].lower() or needle in str(r["answer"]).lower()
            ]
        page = data[offset : offset + limit]
        return JSONResponse(
            {
                "matched": len(data),
                "offset": offset,
                "rows": [
                    {
                        "sample_id": r["sample_id"],
                        "question": r["question"],
                        "answer": r["answer"],
                        "question_type": type_of(r),
                        "source": source_of(r),
                        "gsd": (r.get("effective_gsd_m") or [None])[0],
                        "roles": r.get("image_roles", []),
                        "images": r["images"],
                    }
                    for r in page
                ],
            }
        )

    @api.get("/img")
    def image(path: str):
        # Confined to the image root: a browser must not be able to read
        # arbitrary files off the Volume by asking for ../../something.
        target = (IMAGE_ROOT / path).resolve()
        if not str(target).startswith(str(IMAGE_ROOT.resolve())):
            raise HTTPException(status_code=400, detail="path outside the image root")
        if not target.exists():
            raise HTTPException(status_code=404, detail="no such image")

        from PIL import Image

        with Image.open(target) as source:
            picture = source.convert("RGB")
            if picture.width > 384:
                ratio = 384 / picture.width
                picture = picture.resize(
                    (384, max(1, int(picture.height * ratio))), Image.LANCZOS
                )
            buffer = io.BytesIO()
            picture.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    @api.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    return api


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Corpus Browser</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{--ground:#f6f7f5;--panel:#fff;--edge:#dfe2dc;--ink:#1b1f1c;--soft:#5c635c;
--faint:#8b918a;--accent:#1f6f5c;--accentbg:#e4efe9;--signal:#a8541c;--signalbg:#f6e9de}
@media(prefers-color-scheme:dark){:root{--ground:#12140f;--panel:#1a1d18;--edge:#2c3129;
--ink:#e8ebe4;--soft:#a3aa9e;--faint:#6f766c;--accent:#6fd0ae;--accentbg:#1c2f28;
--signal:#e39a63;--signalbg:#33261b}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.5}
.wrap{max-width:1250px;margin:0 auto;padding:26px 20px 70px}
h1{font-size:1.35rem;margin:0 0 4px;font-weight:600}
p.sub{color:var(--soft);margin:0 0 18px;font-size:.9rem}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px}
select,input,button{font:inherit;font-size:.85rem;padding:6px 10px;border-radius:7px;
border:1px solid var(--edge);background:var(--panel);color:var(--ink)}
button{cursor:pointer}
button.page{background:var(--accentbg);border-color:var(--accent);color:var(--accent);font-weight:500}
button:disabled{opacity:.4;cursor:default}
.count{font-family:"IBM Plex Mono",monospace;font-size:.8rem;color:var(--soft);
font-variant-numeric:tabular-nums;margin:10px 0 16px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
article{background:var(--panel);border:1px solid var(--edge);border-radius:10px;overflow:hidden;
display:flex;flex-direction:column}
.views{display:flex;gap:2px;background:var(--edge)}
.views figure{flex:1;min-width:0;margin:0;position:relative}
.views img{width:100%;aspect-ratio:1;object-fit:cover;display:block}
.views figcaption{position:absolute;left:0;bottom:0;padding:2px 6px;
  font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:.66rem;
  letter-spacing:.04em;color:#fff;background:rgba(12,18,14,.72)}
.body{padding:12px 13px 13px;display:flex;flex-direction:column;gap:9px;flex:1}
.q{font-size:.9rem;margin:0}
.a{font-family:"IBM Plex Mono",monospace;font-size:.83rem;background:var(--signalbg);
color:var(--signal);padding:6px 9px;border-radius:6px;margin:0;
white-space:pre-wrap;word-break:break-word}
.meta{display:flex;flex-wrap:wrap;gap:5px;margin-top:auto;font-family:"IBM Plex Mono",monospace;
font-size:.66rem;color:var(--faint)}
.chip{border:1px solid var(--edge);border-radius:5px;padding:2px 6px}
.chip.gsd{color:var(--accent);border-color:var(--accent)}
</style></head><body><div class="wrap">
<h1>Corpus Browser</h1>
<p class="sub">Every row in the training manifest, served from the Modal volume. Read-only.</p>
<div class="bar">
  <select id="manifest"></select>
  <select id="source"><option value="">all sources</option></select>
  <select id="qtype"><option value="">all question types</option></select>
  <input id="contains" placeholder="search question or answer" size="26">
  <button id="apply">Apply</button>
</div>
<div class="count" id="count">loading…</div>
<div class="grid" id="grid"></div>
<div class="bar" style="margin-top:20px">
  <button class="page" id="prev">&larr; Previous</button>
  <button class="page" id="next">Next &rarr;</button>
</div>
</div><script>
let offset=0, limit=24, matched=0;
const $=id=>document.getElementById(id);
let manifests=[];
async function meta(first){
  const q=first?'':'?manifest='+encodeURIComponent($('manifest').value);
  const m=await (await fetch('api/meta'+q)).json();
  if(first){
    manifests=m.manifests;
    manifests.forEach(n=>$('manifest').add(new Option(n,n)));
  }
  // Rebuild the dependent filters: a question type from rs_vqa does not exist
  // in change_vqa, and leaving the stale options would filter to zero rows.
  for(const id of ['source','qtype']){
    const s=$(id); s.length=0;
    s.add(new Option(id==='source'?'all sources':'all question types',''));
  }
  m.sources.forEach(s=>$('source').add(new Option(s,s)));
  m.types.forEach(t=>$('qtype').add(
    new Option(t+' ('+(m.distinct_answers[t]||0)+' answers)',t)));
}
async function load(){
  const p=new URLSearchParams({offset,limit,source:$('source').value,
    qtype:$('qtype').value,contains:$('contains').value,
    manifest:$('manifest').value});
  const d=await (await fetch('api/rows?'+p)).json();
  matched=d.matched;
  const hi=Math.min(offset+limit,matched);
  $('count').textContent=
    `${matched.toLocaleString()} matching rows · showing ${offset+1}–${hi}`;
  $('grid').innerHTML=d.rows.map(r=>`<article>
    <div class="views">${r.images.map((i,n)=>
      // The role caption is not decoration for a change corpus: "did buildings
      // appear between the first and second image" cannot be checked by eye
      // unless the reader knows which tile is which date.
      `<figure><img loading="lazy" src="img?path=${encodeURIComponent(i)}" alt="">
       ${r.roles&&r.roles[n]?`<figcaption>${esc(r.roles[n])}</figcaption>`:''}</figure>`
      ).join('')}</div>
    <div class="body"><p class="q">${esc(r.question)}</p><p class="a">${esc(String(r.answer))}</p>
    <div class="meta"><span class="chip">${esc(r.source||'')}</span>
    <span class="chip">${esc(r.question_type||'')}</span>
    ${r.gsd?`<span class="chip gsd">${r.gsd} m</span>`:''}</div></div></article>`).join('');
  $('prev').disabled=offset<=0; $('next').disabled=offset+limit>=matched;
  window.scrollTo({top:0});
}
function esc(s){return s.replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
$('apply').onclick=()=>{offset=0;load()};
$('next').onclick=()=>{offset+=limit;load()};
$('prev').onclick=()=>{offset=Math.max(0,offset-limit);load()};
$('manifest').onchange=()=>{offset=0;meta(false).then(load)};
meta(true).then(load);
</script></body></html>"""
