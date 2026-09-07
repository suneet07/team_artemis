"""Render a sampled corpus as a page a person can actually inspect.

    python scripts/build_corpus_page.py --sample logs/corpus_sample.json \
        --out logs/corpus_review.html

Automated validation answers "is this corpus internally consistent". It cannot
answer "does this question make sense for this picture", and that is the failure
that survives every check: a corpus can be structurally perfect and still ask
about pastures in a photograph of open sea.

So this puts the image, the question and the answer side by side, grouped by
source, with the resolution and question type visible -- because those are the
fields that decide whether an answer is plausible, and they are exactly the ones
a spreadsheet hides.
"""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

TEMPLATE_HEAD = """<title>Corpus Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --ground: #f6f7f5;
  --panel: #ffffff;
  --edge: #dfe2dc;
  --ink: #1b1f1c;
  --ink-soft: #5c635c;
  --ink-faint: #8b918a;
  --accent: #1f6f5c;
  --accent-soft: #e4efe9;
  --signal: #a8541c;
  --signal-soft: #f6e9de;
  --shadow: 0 1px 2px rgba(20, 28, 22, .06), 0 8px 24px rgba(20, 28, 22, .05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #12140f;
    --panel: #1a1d18;
    --edge: #2c3129;
    --ink: #e8ebe4;
    --ink-soft: #a3aa9e;
    --ink-faint: #6f766c;
    --accent: #6fd0ae;
    --accent-soft: #1c2f28;
    --signal: #e39a63;
    --signal-soft: #33261b;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
  }
}
:root[data-theme="dark"] {
  --ground: #12140f;
  --panel: #1a1d18;
  --edge: #2c3129;
  --ink: #e8ebe4;
  --ink-soft: #a3aa9e;
  --ink-faint: #6f766c;
  --accent: #6fd0ae;
  --accent-soft: #1c2f28;
  --signal: #e39a63;
  --signal-soft: #33261b;
  --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  line-height: 1.5;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 24px 80px; }
header { display: flex; flex-direction: column; gap: 10px; margin-bottom: 8px; }
h1 { font-size: 1.6rem; font-weight: 600; margin: 0; letter-spacing: -.01em; }
.lede { color: var(--ink-soft); max-width: 62ch; margin: 0; font-size: .95rem; }
.stats {
  display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 22px 0 8px;
  padding: 14px 0; border-top: 1px solid var(--edge); border-bottom: 1px solid var(--edge);
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .8rem;
  font-variant-numeric: tabular-nums; color: var(--ink-soft);
}
.stats b { color: var(--ink); font-weight: 500; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 22px 0 26px; }
button.filter {
  font: inherit; font-size: .82rem; cursor: pointer;
  padding: 6px 13px; border-radius: 999px;
  border: 1px solid var(--edge); background: var(--panel); color: var(--ink-soft);
}
button.filter[aria-pressed="true"] {
  background: var(--accent-soft); border-color: var(--accent); color: var(--accent);
  font-weight: 500;
}
button.filter:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
h2.group {
  font-size: .78rem; text-transform: uppercase; letter-spacing: .1em;
  color: var(--ink-faint); font-weight: 600; margin: 34px 0 14px;
}
.grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); }
article {
  background: var(--panel); border: 1px solid var(--edge); border-radius: 10px;
  box-shadow: var(--shadow); overflow: hidden; display: flex; flex-direction: column;
}
.views { display: flex; gap: 2px; background: var(--edge); }
.view { flex: 1; position: relative; min-width: 0; }
.view img { display: block; width: 100%; aspect-ratio: 1; object-fit: cover; }
.view span {
  position: absolute; left: 5px; bottom: 5px;
  font-family: "IBM Plex Mono", monospace; font-size: .62rem; letter-spacing: .04em;
  background: rgba(8, 12, 9, .72); color: #f2f4f0; padding: 2px 6px; border-radius: 4px;
}
.body { padding: 14px 15px 15px; display: flex; flex-direction: column; gap: 10px; flex: 1; }
.q { font-size: .92rem; margin: 0; }
.a {
  font-family: "IBM Plex Mono", monospace; font-size: .85rem;
  background: var(--signal-soft); color: var(--signal);
  padding: 7px 10px; border-radius: 6px; margin: 0;
  word-break: break-word; white-space: pre-wrap;
}
.meta {
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: auto; padding-top: 4px;
  font-family: "IBM Plex Mono", monospace; font-size: .68rem; color: var(--ink-faint);
}
.chip { border: 1px solid var(--edge); border-radius: 5px; padding: 2px 7px; }
.chip.ctx{background:var(--sunk,#f0efea);color:var(--ink-soft,#5c635c);
  border-color:var(--edge,#dfe2dc)}
.chip.gsd { color: var(--accent); border-color: var(--accent); }
footer { margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--edge);
         color: var(--ink-faint); font-size: .8rem; }
[hidden] { display: none !important; }
</style>
"""


def build(samples: list[dict], footer: str = "") -> str:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_source[s.get("source") or "unknown"].append(s)

    types = sorted({s.get("question_type", "?") for s in samples})
    gsds = sorted({s.get("gsd_m") for s in samples if s.get("gsd_m")})

    parts = [TEMPLATE_HEAD, '<div class="wrap"><header>']
    parts.append("<h1>Corpus Review</h1>")
    parts.append(
        '<p class="lede">A stratified sample of the <code>rs_vqa</code> training '
        "corpus — every source and question type represented. Automated checks "
        "confirm the corpus is internally consistent; this page is for the thing "
        "they cannot check, which is whether each question makes sense for the "
        "picture beside it.</p></header>"
    )

    parts.append('<div class="stats">')
    parts.append(f"<span><b>{len(samples)}</b> samples shown</span>")
    parts.append(f"<span><b>{len(by_source)}</b> sources</span>")
    parts.append(f"<span><b>{len(types)}</b> question types</span>")
    parts.append(
        "<span>resolutions <b>"
        + " · ".join(f"{g:g} m" for g in gsds)
        + "</b></span>"
    )
    parts.append("</div>")

    parts.append('<div class="controls">')
    parts.append('<button class="filter" aria-pressed="true" data-type="all">All types</button>')
    for t in types:
        parts.append(
            f'<button class="filter" aria-pressed="false" data-type="{html.escape(t)}">'
            f"{html.escape(t)}</button>"
        )
    parts.append("</div>")

    for source in sorted(by_source):
        rows = by_source[source]
        parts.append(f'<h2 class="group">{html.escape(source)} — {len(rows)} shown</h2>')
        parts.append('<div class="grid">')
        for s in rows:
            parts.append(
                f'<article data-type="{html.escape(s.get("question_type", "?"))}">'
            )
            if s["images"]:
                parts.append('<div class="views">')
                for image, role in zip(
                    s["images"], s.get("roles") or [i["role"] for i in s["images"]],
                    strict=False,
                ):
                    parts.append(
                        f'<div class="view"><img src="{image["uri"]}" alt="" loading="lazy">'
                        f"<span>{html.escape(str(role))}</span></div>"
                    )
                parts.append("</div>")
            parts.append('<div class="body">')
            parts.append(f'<p class="q">{html.escape(s["question"])}</p>')
            parts.append(f'<p class="a">{html.escape(str(s["answer"]))}</p>')
            parts.append('<div class="meta">')
            # `question_type` and `task` are the same string for a
            # generator-written corpus, so chip them once rather than twice.
            labels = [s.get("question_type"), s.get("task")]
            for label in dict.fromkeys(x for x in labels if x):
                parts.append(f'<span class="chip">{html.escape(str(label))}</span>')
            if s.get("gsd_m"):
                parts.append(f'<span class="chip gsd">{s["gsd_m"]:g} m</span>')
            for key, value in (s.get("context") or {}).items():
                parts.append(
                    f'<span class="chip ctx">{html.escape(f"{key} {value}")}</span>'
                )
            parts.append("</div></div></article>")
        parts.append("</div>")

    parts.append(
        f"<footer>{html.escape(footer)} Images are the real training inputs, "
        "downscaled for display only.</footer></div>"
    )
    parts.append("""
<script>
const buttons = document.querySelectorAll('.filter');
buttons.forEach(b => b.addEventListener('click', () => {
  const want = b.dataset.type;
  buttons.forEach(o => o.setAttribute('aria-pressed', String(o === b)));
  document.querySelectorAll('article').forEach(card => {
    card.hidden = want !== 'all' && card.dataset.type !== want;
  });
  document.querySelectorAll('h2.group').forEach(h => {
    const grid = h.nextElementSibling;
    const any = [...grid.querySelectorAll('article')].some(c => !c.hidden);
    h.hidden = !any; grid.hidden = !any;
  });
}));
</script>""")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", default="logs/corpus_sample.json")
    parser.add_argument("--out", default="logs/corpus_review.html")
    parser.add_argument(
        "--footer",
        default="Sampled from the corpus manifest.",
        help="provenance line; name the actual corpus, not a hardcoded one",
    )
    parser.add_argument(
        "--title",
        default="Corpus Review",
        help="page title; name the corpus, since this is what a gallery shows",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    page = build(data["samples"], args.footer).replace(
        "<title>Corpus Review</title>", f"<title>{html.escape(args.title)}</title>", 1
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{len(data['samples'])} sample(s) -> {out} ({out.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
