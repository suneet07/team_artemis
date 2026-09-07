# Slide 3 — TECHNICAL APPROACH

**Required sub-headings:** technologies to be used · methodology and process
(flow charts / images / working prototype).

This is the slide that proves it exists. **Make the architecture diagram the hero
and put a real screenshot on it.**

---

## Block A — Technologies

| layer | what |
|---|---|
| **Backbone** | Qwen3-VL-4B-Instruct (Apache 2.0) — **one** model, loaded once |
| **Adapters** | 2 × LoRA, r=16 α=32, all-linear · **40,271,872 params (0.899%)** each |
| **Radar** | BIFOLD `resnet50-s1` — 19-class land cover, MIT |
| **Deterministic** | NDVI/NDWI/MNDWI/NDBI · SAR backscatter dB · texture segmentation · co-registration · set arithmetic · centroid prior |
| **Geospatial** | rasterio · GDAL · scikit-image · AROSICS |
| **Agent** | rules-first router · parameter gate · dependency-wave executor · trace builder |
| **Serving** | Modal (L4 GPU) · PEFT multi-adapter hot-swap · FastAPI |
| **Console** | React + TypeScript + Vite · deck.gl / MapLibre · Vercel |
| **Verification** | 440 unit tests · 50 route checks · 200-row held-out replay |

**The constraint that shaped everything:** one shared base with adapters swapped
over it. Adapters are tens of MB; the base is gigabytes. **Four capabilities fit
on a single L4.**

---

## Block B — The architecture (the hero diagram)

```
        question + imagery (GeoTIFF: optical / SAR / pair)
                          │
              ┌───────────▼───────────┐
              │   ROUTER  (rules)     │  100% on 285 cases
              │   8 task branches     │  15 of 16 need no LLM
              └───────────┬───────────┘
                          │  smallest sufficient plan
              ┌───────────▼───────────┐
              │   PARAMETER GATE      │  refuses before running
              │   manifest + bands    │  D3 band gating
              └───────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  DETERMINISTIC      LEARNED           RADAR
  spectral_index     rs_vqa (LoRA)     lulc_classifier
  texture_seg        change_vqa (LoRA) 74.95%
  sar_backscatter    rs_ground_caption
  coreg_check        (base + prompt)
  change_stats
        └─────────────────┼─────────────────┘
                          ▼
              ┌───────────────────────┐
              │  D1 DECISION FUSION   │  5 physical rules
              │  optical vs SAR       │  disagreement → intersection
              └───────────┬───────────┘
                          ▼
            ANSWER  +  EVIDENCE  +  TRACE
         (text)     (GeoTIFF masks)  (every step, threshold, why)
```

---

## Block C — Methodology, in five steps

```
1  INGEST      probe modality, band inventory, GSD, CRS   -> capability, not claim
2  ROUTE       words + input shape -> 1 of 8 tasks         -> deterministic
3  GATE        manifest + bands decide what MAY run        -> refuse, don't fail
4  EXECUTE     dependency waves, parallel within a wave    -> masks + answers
5  RECONCILE   D1 physical rules, then compose + trace     -> auditable output
```

**Worked example — put this on the slide:**

> *"Use the optical and SAR images together to identify built-up regions"*
> → `crossmodal_vqa` → `coreg_check` (1.58 px) → `lulc_classifier` →
> `texture_seg` + `sar_backscatter` → **D1: IoU 0.01, no physical rule applies →
> report the agreed extent, confidence × 0.6**

That last step is the pitch: **the system says when it does not know.**

---

## Block D — Two engineering claims worth a line each

**Train/serve parity is asserted, not assumed.** The views the model gets at
serving time are checked **pixel-for-pixel against the images it trained on**.

**The deploy verifies itself.** Tests → snapshot → deploy → wait for the exact
content-addressed build id → drive all 20 endpoints and all 8 tasks. **Refuses to
deploy on a red tree.**

---

## Layout direction

- **Top 55%:** the architecture diagram, redrawn properly. Colour-code the three
  execution lanes (deterministic / learned / radar) and make the ROUTER and GATE
  boxes visually dominant — they are the novelty.
- **Bottom left:** the 5-step methodology as a horizontal pipeline.
- **Bottom right:** a **real screenshot** — the workspace with a grounded box and
  the trace panel open. Label it *"live system"*.
- Technologies as a compact table or logo strip along the very bottom.

---

## Speaker notes

- The router and the gate are the intellectual contribution. Say: **"the model is
  one tool among several, and the router decides when it is the right one."**
- The `coreg_check 1.58 px` figure is worth speaking aloud — it shows the system
  measures its own inputs.
- If time is short, cut Block D and keep the worked example.

## If a judge pushes

**"Why 4B and not a larger model?"**
> The BEN.txt paper settles it: their 1B fine-tuned model beats every zero-shot
> entry including a 2-trillion-parameter GPT. This task needs vocabulary and
> in-domain supervision, not scale. We also measured an 8B base on captioning —
> **+0.016 ROUGE-L for double the weights.**

**"Is the router just if-statements?"**
> Deliberately. The problem statement grades orchestration and says internal
> reasoning is not evaluated. A rules router is **auditable in the trace and
> reproducible**; an LLM router is sampled. 100% on 285 unambiguous cases.
