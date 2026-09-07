"""Ask the trained adapter a question about an image you upload.

The evaluation harness answers "how well does it score". This answers the
question a person actually has, which is "what does it say about *this*
picture" -- and it is the only way to notice failures a benchmark average
hides, because an aggregate cannot show you a wrong answer.

**It reproduces the training prompt exactly**, which is the whole difficulty.
The adapter was trained on three composite views per sample and a scale prefix
naming the ground sample distance, the scene extent and, for size words, the
threshold that applies. Ask it a bare question with one image and it answers
worse for reasons that have nothing to do with what it learned -- so this
rebuilds that prompt, and shows it, rather than letting a demo quietly measure
the wrong thing.

Uploads are held in memory for the duration of the request and never written
to the Volume: this is a window onto the model, not a way to add data to it.
"""

import io
import os

#: Where the LoRA lives on the Volume. The base model is pulled from the HF
#: cache that the training runs already populated.
ADAPTER = os.environ.get("ASK_ADAPTER", "/data/checkpoints/rs_vqa/adapter")
BASE_MODEL = os.environ.get("ASK_BASE", "Qwen/Qwen3-VL-4B-Instruct")

#: The scales the adapter was actually trained on, so the prefix it sees at
#: demo time is one it saw during training. "Other" exists because a real
#: Cartosat tile is neither of these, and pretending otherwise would hide the
#: domain gap rather than show it.
#: The third element is the corpus ``source`` string, because the size-word
#: thresholds are keyed by source and not by resolution -- BigEarthNet.txt is
#: 10 m like RSVQA-LR but defines no size categories at all. Passing the source
#: through means the demo resolves the rule with the same lookup training uses,
#: rather than guessing from the GSD.
PRESETS = {
    "10": ("Sentinel-2 / BigEarthNet, RSVQA-LR", 10.0, "RSVQA-LR"),
    "0.3048": ("Aerial 0.3 m / RSVQA-HR", 0.3048, "RSVQA-HR"),
    "2.5": ("Cartosat-2S (untrained scale)", 2.5, ""),
}


def _source_for(gsd: float) -> str:
    """The corpus source whose size rule applies at this resolution."""
    for _, value, source in PRESETS.values():
        if abs(value - gsd) < 1e-6:
            return source
    return ""


def build_app(answer_fn):
    """The ASGI app.

    ``answer_fn(items, use_adapter) -> list[str]`` is supplied by the caller so
    that the model is loaded once per container rather than per request, and so
    that turning the adapter off costs nothing: PEFT can disable it in place,
    which is a great deal cheaper than holding a second copy of the base model.
    """
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    api = FastAPI(title="Ask SatQuery")

    @api.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @api.post("/api/ask")
    async def ask(
        question: str = Form(...),
        gsd: float = Form(10.0),
        adapter: str = Form("on"),
        # Form/File in a default is FastAPI's own documented idiom.
        images: list[UploadFile] = File(...),  # noqa: B008
    ):
        from PIL import Image

        loaded = []
        for upload in images[:3]:
            raw = await upload.read()
            if raw:
                loaded.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        if not loaded:
            return JSONResponse({"error": "no readable image"}, status_code=400)

        # Three views per sample, as trained. One upload is repeated rather
        # than padded with blanks: repetition is what the corpus did for the
        # single-image sources, so it is the in-distribution choice.
        while len(loaded) < 3:
            loaded.append(loaded[len(loaded) % len(loaded)])

        # The *same* function training and evaluation use. Reimplementing it
        # here is how a demo ends up showing a prompt the model never saw.
        from satquery.training.dataset import scale_prefix

        prefix = scale_prefix([gsd], question, _source_for(gsd), loaded)
        prompt = f"{prefix}{question}"

        raw_answer = answer_fn(
            [{"images": loaded, "question": prompt}], adapter == "on"
        )[0]

        # No answer contract is applied here on purpose. The formatter maps onto
        # a *benchmark's* closed vocabulary, and a free-form question belongs to
        # no benchmark -- forcing "yes" or an option letter onto an arbitrary
        # question would hide what the model actually said, which is the one
        # thing this page exists to show.
        return JSONResponse(
            {
                "prompt": prompt,
                "answer": raw_answer,
                "views": len(loaded),
                "adapter": ADAPTER if adapter == "on" else None,
            }
        )

    return api


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ask SatQuery</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{--ground:#f6f7f5;--panel:#fff;--edge:#dfe2dc;--ink:#1b1f1c;--soft:#5c635c;
--faint:#8b918a;--accent:#1f6f5c;--accentbg:#e4efe9;--signal:#a8541c;--signalbg:#f6e9de}
@media(prefers-color-scheme:dark){:root{--ground:#12140f;--panel:#1a1d18;--edge:#2c3129;
--ink:#e8ebe4;--soft:#a3aa9e;--faint:#6f766c;--accent:#6fd0ae;--accentbg:#1c2f28;
--signal:#e39a63;--signalbg:#33261b}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.55}
.wrap{max-width:860px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:1.4rem;margin:0 0 4px;font-weight:600}
p.sub{color:var(--soft);margin:0 0 24px;font-size:.9rem;max-width:60ch}
label{display:block;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;
color:var(--faint);margin:0 0 6px;font-weight:600}
.row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.row>div{flex:1;min-width:190px}
input,select,textarea,button{font:inherit;width:100%;padding:9px 11px;border-radius:8px;
border:1px solid var(--edge);background:var(--panel);color:var(--ink)}
textarea{min-height:74px;resize:vertical}
button{cursor:pointer;background:var(--accent);color:#fff;border-color:var(--accent);
font-weight:600;margin-top:8px}
button:disabled{opacity:.5;cursor:default}
.thumbs{display:flex;gap:8px;margin:12px 0}
.thumbs img{width:104px;height:104px;object-fit:cover;border-radius:8px;
border:1px solid var(--edge)}
.out{margin-top:26px;background:var(--panel);border:1px solid var(--edge);
border-radius:10px;padding:16px;display:none}
.out.show{display:block}
.answer{font-family:"IBM Plex Mono",monospace;font-size:1.15rem;background:var(--signalbg);
color:var(--signal);padding:12px 14px;border-radius:8px;margin:0 0 14px;word-break:break-word}
.meta{font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--faint);
white-space:pre-wrap;word-break:break-word;margin:0}
.meta b{color:var(--soft);font-weight:500}
.note{font-size:.8rem;color:var(--soft);border-left:2px solid var(--edge);
padding-left:12px;margin-top:22px}
</style>
<div class="wrap">
<h1>Ask SatQuery</h1>
<p class="sub">Upload a satellite image and ask a question. The scale prefix the adapter
was trained with is rebuilt from your inputs and shown below the answer, so you can see
exactly what the model received.</p>

<label for="q">Question</label>
<textarea id="q">Is a residential area present?</textarea>

<div class="row">
  <div>
    <label for="gsd">Resolution</label>
    <select id="gsd">
      <option value="10">10 m — Sentinel-2 (BEN, RSVQA-LR)</option>
      <option value="0.3048">0.30 m — aerial (RSVQA-HR)</option>
      <option value="2.5">2.5 m — Cartosat-2S (untrained scale)</option>
    </select>
  </div>
  <div>
    <label for="ad">Model</label>
    <select id="ad">
      <option value="on">Fine-tuned adapter</option>
      <option value="off">Base model (no adapter)</option>
    </select>
  </div>
</div>

<label for="f">Image — up to 3 views; one is repeated to three, as in training</label>
<input id="f" type="file" accept="image/*" multiple>
<div class="thumbs" id="thumbs"></div>
<button id="go">Ask</button>

<div class="out" id="out">
  <p class="answer" id="ans">…</p>
  <p class="meta" id="meta"></p>
</div>

<p class="note">This adapter was trained only on one-word answers, so it will not
describe a scene — that capability belongs to a different adapter. Switch to the base
model to see the difference in both answer style and speed.</p>
</div>
<script>
const $=i=>document.getElementById(i);
$('f').onchange=()=>{
  $('thumbs').innerHTML='';
  [...$('f').files].slice(0,3).forEach(file=>{
    const img=document.createElement('img');
    img.src=URL.createObjectURL(file); $('thumbs').append(img);
  });
};
$('go').onclick=async()=>{
  const files=[...$('f').files].slice(0,3);
  if(!files.length){alert('Choose an image first.');return;}
  $('go').disabled=true; $('go').textContent='Thinking…';
  const body=new FormData();
  body.append('question',$('q').value);
  body.append('gsd',$('gsd').value);
  body.append('adapter',$('ad').value);
  files.forEach(f=>body.append('images',f));
  try{
    const r=await fetch('api/ask',{method:'POST',body});
    const d=await r.json();
    $('out').classList.add('show');
    if(d.error){$('ans').textContent='Error: '+d.error;$('meta').textContent='';return;}
    const t0=performance.now();
    $('ans').textContent=d.answer||'(empty)';
    $('meta').innerHTML='<b>prompt sent</b>\\n'+esc(d.prompt)+
      '\\n\\n<b>views</b> '+d.views+'   <b>adapter</b> '+(d.adapter||'none (base model)');
  }catch(e){
    $('out').classList.add('show');
    $('ans').textContent='Request failed: '+e;
  }finally{ $('go').disabled=false; $('go').textContent='Ask'; }
};
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
</script>
"""
