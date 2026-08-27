# CREDITS.md

Every model, dataset, library and method used by SatQuery AI, with citation and licence. Maintained from day one (master plan Part 11). Status markers: `CLEARED` = verified shippable, `⚠ PENDING` = Phase 0 check outstanding, `REJECTED` = excluded with reason.

When an ISRO judge asks "what's yours and what isn't" — the answer is this file.

## Models

| Source | Artifact | Use | Licence | Status |
|---|---|---|---|---|
| Alibaba / HF | `Qwen3-VL-4B-Instruct` | Base VLM for all four LoRA adapters | Apache 2.0 | ✅ CLEARED |
| TU Berlin / BigEarthNet | BigEarthNet-pretrained ViT (S1 + S2) | `lulc_classifier`, stretch experiment only | CDLA-Permissive 1.0 (C57) | ✅ CLEARED |
| torchgeo | Pretrained S1/S2 ResNet/ViT weights | Named substitute if BEN ViT restricted | MIT (C57) | ✅ CLEARED |
| DOFA / SatMAE | Wavelength-aware / MAE RS encoders | Second-line substitutes | verify | ⚠ Phase 0 |
| Meta | SAM / SAM2 | Optional mask refinement | Apache 2.0 | ✅ CLEARED |

## Datasets

| Source | Content / use | Licence | Status |
|---|---|---|---|
| BigEarthNet.txt (arXiv 2603.29630) | Primary adaptation dataset + benchmark split | CDLA-Permissive 1.0 (verified annotation layer, C51) | ✅ CLEARED |
| BigEarthNet v2.0 / reBEN | Underlying imagery (manifest subset) | CDLA-Permissive 1.0 — confirmed; no restrictions on results of computational use | ✅ CLEARED |
| CDVQA (arXiv 2112.06343) | Change training + PS-nominated evaluation | Confirm annotation terms; imagery SECOND-derived | ⚠ PENDING |
| VRSBench (arXiv 2406.12384) | **EVALUATION ONLY** — grounding acc@0.5/0.7. Never in training, never in shipped weights, never in repo/demo images | Text CC-BY-4.0; images DOTA-derived academic-only | ✅ as eval-only |
| RSVQA-LR | VQA train minority + eval | CC BY 4.0 (C54) | ✅ CLEARED |
| RSVQA-HR | First-class train + eval (closest GSD to Cartosat-2S) | USGS imagery public domain; annotations CC BY 4.0 (C54) | ✅ CLEARED |
| RarePlanes | Aircraft object grounding @30 cm | CC BY-SA 4.0 (AWS Open Data) | ✅ CLEARED |
| OpenEarthMap-SAR | Sub-metre cross-modal pairs, single-pol validation | SAR: Umbra Lab CC BY 4.0; optical: NAIP PD, IGN France CC BY 2.0, GSI Japan | ✅ CLEARED |
| SpaceNet 6 MSAW | X-band quad-pol cross-modal @0.5 m, building footprints, D1 validation | CC BY-SA 4.0 | ✅ CLEARED |
| LS-SSDD-v1.0 | Ship boxes on Sentinel-1 | Apache-2.0 (C53) | ✅ CLEARED |
| SARLANG-1M | SAR-only QA, pol-dropout training | Per-subset: SpaceNet 6 portion clean; DFC2023 / OpenEarthMap-SAR / SARDet-100K each need own check | ⚠ PENDING (C35) |
| SpaceNet 7 / MUDS | 11M manually-annotated building footprints, expert change source (C59) | CC BY-SA 4.0 (AWS Open Data) | ✅ CLEARED |
| HRSCD | 0.5 m aerial over France, semantic change (C60) | IGN licence ouverte (Caveat: 2006 images non-redistributable) | ✅ CLEARED |
| Bhoonidhi (NRSC) | Cartosat-2S + RISAT samples → India holdout v1 | ISRO terms | apply Day 1 |
| Open India proxy | Sentinel over Indian AOIs → India holdout v0 | Open | ✅ CLEARED |
| SRTM / Copernicus DEM | Terrain correction | Open | ✅ CLEARED |
| DynamicEarthNet | LULC change labels on 75 AOIs (C61) | ⚠ Unverified (commercial Planet Fusion core) | ⚠ PENDING |

## Libraries

GDAL (MIT/X) · rasterio (BSD-3) · rioxarray/xarray (Apache 2.0) · pyproj (MIT) · torchgeo (MIT) · AROSICS (Apache-2.0 via pinned >= 1.0.0, C58) · scikit-image (BSD-3) · ESA SNAP + pyroSAR (**GPL-3.0**) · transformers (Apache 2.0) · peft (Apache 2.0) · bitsandbytes (MIT) · vLLM (Apache 2.0) · Outlines (Apache 2.0) · LangGraph (MIT) · FastAPI (MIT) · Redis (BSD) · Celery (BSD) · React (MIT) · MapLibre GL (BSD-3) · scikit-learn (BSD-3) · Weights & Biases (free tier).

**SNAP GPL note:** SNAP is invoked strictly as an external process (subprocess / Docker service). It is never imported or linked into our code, keeping our codebase outside the GPL boundary.

## Methods read and reimplemented (not cloned)

RS-InternVL hyperparameters (BEN.txt paper §4.2) · GeoPixel adaptive partitioning (arXiv 2501.13925) · EarthMind HCA fusion design + negative result (arXiv 2506.01667) · Earth-Agent dual-level evaluation (ICLR 2026) · CDVQA Qwen study targets (arXiv 2604.18429) · SMARTIES/DOFA band projection idea (cut-list item) · Change-Agent multi-task change interpretation (TGRS 2024).

Classical algorithms implemented ourselves: NDVI, NDWI, MNDWI, NDBI, Otsu with bimodality gate, Refined Lee speckle filter, phase cross-correlation, connected-component analysis, percentile stretch, temperature scaling, GLCM texture, local coefficient of variation, isotonic calibration fit.

## Explicit exclusions (with reasons — do not reintroduce)

| Source | Reason |
|---|---|
| xView3-SAR | T&C page 404s, registration wall, composites carry "© Cambrio LLC; rights reserved" despite open underlying Sentinel-1 (C38) |
| DIOR, FAIR1M, NWPU-Captions, RSICD | All Google Earth-derived imagery; no shippable licence (C32) |
| VRSBench (training use) | Eval-only per PS role assignment; DOTA-derived academic-only images must never enter shipped weights (C20) |
| TinyCD / ChangeFormer | Pretrained weights are RESTRICTED (non-commercial/academic only) (C55) |
| LEVIR-CD / LEVIR-MCI / SECOND / QAG-360K | Academic-only images. Artifact-type exception reversed (C56). Purged from ALL training manifests. |

## Standing licence rules

1. **Provenance Traceability Rule (C45):** Before any dataset enters a training manifest, trace it to the imagery programme it originates from, not merely to the paper that published it. A derived dataset inherits the most restrictive licence in its ancestry.
2. **SNAP rule:** external process only, never linked.
3. **No training on data whose licence is still ⚠.**
4. **Per-account training policy:** one adapter per person, on that person's own account; no cross-account relay of runs.

## Citations to add before submission

- BigEarthNet.txt paper: arXiv 2603.29630
- VRSBench: arXiv 2406.12384 · CDVQA: arXiv 2112.06343 · SARLANG-1M: arXiv 2504.03254
- GeoPixel: arXiv 2501.13925 · EarthMind: arXiv 2506.01667
- SpaceNet 6, SpaceNet 7, RarePlanes, OpenEarthMap-SAR, LS-SSDD-v1.0 dataset papers/DOIs (fill at staging time)
