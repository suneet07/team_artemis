# Slide 6 — RESEARCH AND REFERENCES

**Required:** details / links of the reference and research work.

**The framing:** this slide quietly proves the rest of the deck. Every number on
slides 2–4 traces to something here, and the licence column shows we did the work
most teams skip.

---

## Block A — Benchmarks we are measured on

| benchmark | what it grades | our result | reference |
|---|---|---|---|
| **RSVQA-HR / LR** | single-image VQA | **85.06 / 83.08** AA | Lobry et al., *IEEE TGRS* 2020 — arXiv 2003.07333 |
| **CDVQA** | change VQA | **68.0** AA | Yuan et al., *IEEE TGRS* 2022 — arXiv 2112.06343 |
| **VRSBench** | referring grounding + captioning | **62.7%** acc@0.5 · ROUGE-L 25.2 | Li et al., 2024 — arXiv 2406.12384 |
| **BigEarthNet.txt** | multi-label VQA / MCQ | **76.78 / 73.62** | BEN.txt — CDLA-Permissive 1.0 |
| **reBEN** | SAR land cover | **74.95%** | Clasen et al., *IGARSS* 2025 — arXiv 2407.03653 |

---

## Block B — Models and weights

| component | source | licence |
|---|---|---|
| **Qwen3-VL-4B-Instruct** — backbone | Alibaba / HuggingFace | Apache 2.0 |
| **BIFOLD `resnet50-s1`** — radar land cover | BIFOLD-BigEarthNetv2-0 | MIT |
| **PEFT / LoRA** — Hu et al., arXiv 2106.09685 | HuggingFace | Apache 2.0 |
| BigEarthNet-pretrained ViT (S1+S2) | TU Berlin | CDLA-Permissive 1.0 |

---

## Block C — Training data (all licence-cleared)

| source | used for | licence |
|---|---|---|
| **BigEarthNet.txt** | `rs_vqa` — primary adaptation set | CDLA-Permissive 1.0 |
| **RSVQA-LR / HR** | `rs_vqa` | CC BY 4.0 |
| **CDVQA** | `change_vqa` | Apache-2.0 |
| **SpaceNet 7 / MUDS** | change corpus | CC BY-SA 4.0 |
| **RarePlanes** | grounding / captioning | CC BY 4.0 |
| **SpaceNet 2 / 6** | grounding / captioning | CC BY-SA 4.0 |
| **reBEN** | cross-modal | CDLA-Permissive 1.0 |

> **Full provenance:** `CREDITS.md` — every model, dataset, library and method
> with its citation, licence and status. **The barrier is enforced by
> `tests/test_license_blocklist.py`, which fails the build.**

---

## Block D — Methods we built on

- **LoRA** — Hu et al., arXiv 2106.09685
- **Otsu thresholding** — Otsu, *IEEE SMC* 1979 *(used behind a bimodality gate)*
- **AROSICS** co-registration — Scheffler et al., *Remote Sensing* 2017 (Apache 2.0)
- **Spectral indices** — NDVI (Rouse 1974) · NDWI (McFeeters 1996) · MNDWI (Xu 2006) · NDBI (Zha 2003)
- **RS-InternVL** — comparison baseline, BEN.txt paper §4.2
- **GeoChat** — grounding/captioning baseline, arXiv 2311.15826
- **EarthMind** — cross-modal fusion design + negative result, arXiv 2506.01667

---

## Block E — Our own documentation

> Written to be checked, not to be read once.

```
docs/paper/          15 files -- method, results, limitations, failure atlas
docs/segments/        9 files -- per-capability engineering record
CREDITS.md                     -- every source, licence and status
```

**Including a limitations chapter and a failure atlas** — nine defects found by
replaying held-out benchmark rows through the live system, and what each proved.

---

## Layout direction

- Four compact tables (benchmarks · models · data · methods). **Licence column
  visible on every one** — that column is the argument.
- Small footer strip: `docs/paper/` · `docs/segments/` · `CREDITS.md`.
- QR code to the repo or the live demo, if permitted.
- Dense is fine here. This slide is scanned, not read — but a judge who *does*
  read it should find every claim sourced.

---

## Speaker notes

- One line only: **"Every number on the previous slides traces to a benchmark on
  this page, and every source has a licence we can ship."**
- If asked about IP: *what is ours* is the two adapters, the corpus generators,
  the agent and trace layer, and the geospatial validation — all built on
  Apache/MIT/CC foundations with nothing encumbered.

## If a judge pushes

**"Did you use anything you cannot ship?"**
> No. `CREDITS.md` lists what was rejected and why — Google-Earth-derived sets
> with no shippable licence, non-commercial weights, and datasets whose
> permissive *code* wrapped encumbered *imagery*. A test fails the build if any
> of them appears in training code.
