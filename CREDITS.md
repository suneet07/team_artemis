# Credits and licences

Every model, dataset, library and paper SatQuery AI relies on, with its licence
and its status. When a reviewer asks *"what is yours and what isn't?"* — this is
the answer.

**Status markers**

| marker | meaning |
|---|---|
| ✅ CLEARED | licence verified against a primary source; shippable |
| 📊 EVAL ONLY | may be measured on, permanently barred from training |
| ⚠ PENDING | licence not yet verified at the primary source |
| ❌ REJECTED | excluded, with the reason recorded in §7 |

The barrier is **enforced mechanically**, not by convention:
`tests/test_license_blocklist.py` fails the build if an excluded source appears
outside this file, if an eval-only source reaches a training manifest, or if a
shipped weight has no cleared entry here.

---

## 1. The licence position

This deliverable has a **dual character**, decided 2026-08-29. It is an SIH
entry for ISRO/SAC — academic research work — *and* a candidate for operational
deployment. Both are true, so licence questions are judged **per use**, not per
project:

- **Anything that ships** — training corpora, adapter weights, deployed
  inference — is **commercial use**. NonCommercial and academic-only sources are
  excluded absolutely.
- **Benchmark evaluation and reported research numbers** are **academic use**,
  which is exactly what academic-only terms grant. Read-only evaluation does not
  enter the weights and is not redistributed.
- **ShareAlike is permitted** in both, with an obligation attached — see §6.

**The limit of this position.** It holds only while the evaluation stays
academic. If a benchmark number is later used as a commercial claim — a sales
deck, a procurement response, a product page — that use is commercial, and the
academic-only sources cannot back it.

---

## 2. Models

Everything the deployed system loads.

| Artifact | Source | Role | Licence | Status |
|---|---|---|---|---|
| `Qwen3-VL-4B-Instruct` | Alibaba / HuggingFace | Base VLM. One copy, shared by both LoRA adapters and the untrained grounding and captioning paths | Apache 2.0 | ✅ CLEARED |
| `rs_vqa` LoRA | **ours** | Single-image VQA. r=16 α=32, 40,271,872 params (0.899%) | — | ours |
| `change_vqa` LoRA | **ours** | Bi-temporal change VQA. Same shape | — | ours |
| `BIFOLD-BigEarthNetv2-0/resnet50-s1` | TU Berlin / BIFOLD | `lulc_classifier` — 19-class radar land cover, used as published, untrained by us | See the note below | ⚠ PENDING |
| SAM / SAM2 | Meta | Optional mask refinement | Apache 2.0 | ✅ CLEARED |

**The `resnet50-s1` licence needs one verbatim check.** Two records in this
project disagree: the deck and README state **MIT**, while the master plan's C57
records the BIFOLD/TU Berlin BigEarthNet weights as **CDLA-Permissive-1.0** — and
C57 was written about the *ViT*, a different artifact from the ResNet-50 we
actually serve. Both licences are permissive and either would clear, so nothing
shipped is at risk; what is missing is a reading of the model card for the exact
repository. Marked ⚠ until someone reads it and records the finding in §8.

Substitutes evaluated and available if the above were ever restricted:
**torchgeo** pretrained S1/S2 weights (MIT) · **DOFA** wavelength-aware encoder
(MIT for the code; HF weights carry no separate statement, treated as MIT).

---

## 3. Datasets we trained on

| Dataset | What it contributes | Licence | Status |
|---|---|---|---|
| **BigEarthNet.txt** (arXiv 2603.29630) | Primary adaptation corpus — 40,000 rows, 10 m | CDLA-Permissive 1.0 (annotation layer verified) | ✅ CLEARED |
| **BigEarthNet v2.0 / reBEN** | The underlying imagery for the manifest subset | CDLA-Permissive 1.0; no restriction on results of computational use | ✅ CLEARED |
| **RSVQA-HR** (train split) | 20,529 rows at 0.30 m — the only sub-metre optical in the whole inventory | USGS imagery public domain; annotations CC BY 4.0 | ✅ CLEARED |
| **RSVQA-LR** (train split) | 10,000 rows at 10 m | CC BY 4.0 | ✅ CLEARED |
| **CDVQA** (arXiv 2112.06343) | The whole `change_vqa` adapter | Apache 2.0 | ✅ CLEARED |
| RarePlanes | Aircraft grounding at 30 cm | CC BY-SA 4.0 | ✅ CLEARED · SA |
| SpaceNet 6 MSAW | X-band quad-pol cross-modal pairs at 0.5 m | CC BY-SA 4.0 | ✅ CLEARED · SA |
| SpaceNet 7 / MUDS | 11M building footprints; change source | CC BY-SA 4.0 | ✅ CLEARED · SA |
| OpenEarthMap-SAR | Sub-metre cross-modal, single-pol validation | SAR: Umbra Lab CC BY 4.0 · optical: NAIP public domain, IGN France CC BY 2.0, GSI Japan | ✅ CLEARED |
| OSCD — **imagery only** | AOI and date seeds, re-fetched from Copernicus | Modified Copernicus data, open. *Its change labels are rejected — see §7* | ✅ CLEARED |
| HRSCD — **imagery only** | 0.5 m aerial over France | IGN *licence ouverte*. 2006 images are non-redistributable and must come from IGN directly. *Its annotations are rejected — see §7* | ✅ CLEARED |
| SRTM / Copernicus DEM | Terrain correction | Open | ✅ CLEARED |
| Open India proxy | Sentinel over Indian AOIs — India holdout v0 | Open | ✅ CLEARED |
| Bhoonidhi (NRSC) | Cartosat-2S + RISAT samples — India holdout v1 | ISRO terms | ⚠ PENDING |

### The RSVQA train splits, declared plainly

The problem statement assigns roles: *"BigEarthNet.txt will serve as the primary
dataset for adapting… VRSBench and RSVQA will be used to evaluate…"*. We do use
RSVQA's **official train splits**, and this is a disclosure, not something a
reader should have to discover.

**It is not benchmark leakage.** RSVQA ships author-defined splits as separate
files. Train and test share **no questions and no images** — verified, not
assumed: LR train 57,223 against test 10,004, overlap zero on both counts.
Training on train and reporting on test is how every published RSVQA number was
produced, including the baselines we compare against. RSVQA-HR's fourth split,
`test_phili` (Philadelphia held out as a different city), is never touched.

**Why RSVQA and not VRSBench.** The eval role alone is not what bars a dataset.
VRSBench is barred because its eval role comes *with* a licence problem —
DOTA-derived academic-only imagery in a deliverable that ships weights. RSVQA
carries no such problem.

**One question type is excluded, on evidence.** RSVQA-HR's `area` answers are
computed from OpenStreetMap polygons and are unusable: 62% are `0m2` where OSM
has nothing mapped; 275 rows claim more building area than the 6,088 m² tile
physically contains, the largest at 21,264 m²; 39 more are implausibly small,
including a "school" of 6 m². Roughly 29% are believable. 3,471 rows dropped.
BigEarthNet.txt asks the same concept as a bounded choice over rasterised CORINE
maps and does not share the fault, so the capability is kept.

**Size categories are kept with their thresholds stated in the prompt.** The two
scales define the same words 30× apart — "small" is under 3,000 m² at 10 m and
under 100 m² sub-metre (RSVQA paper, Table I). Rather than drop 3,067 rows, each
row carries its own rule in the prompt prefix, keyed on source.

---

## 4. Datasets we evaluated on only

Measured against, never trained on. The barrier is enforced by
`scripts/stage_benchmarks.py` (`eval_only`, `train_forbidden`) and by
`tests/test_license_blocklist.py`.

| Dataset | Used for | Licence | Status |
|---|---|---|---|
| **VRSBench** (arXiv 2406.12384) | Referring grounding, acc@0.5 | Text CC BY 4.0. Images are **DOTA-derived: academic purposes permitted, commercial use prohibited**. Academic evaluation is the use DOTA grants — licensed, not merely tolerated | 📊 EVAL ONLY · ❌ barred from training |

---

## 5. Libraries

**Geospatial** — GDAL (MIT/X) · rasterio (BSD-3) · rioxarray / xarray (Apache 2.0)
· pyproj (MIT) · torchgeo (MIT) · AROSICS (Apache 2.0, pinned `>=1.0.0`) ·
scikit-image (BSD-3)

**Model stack** — transformers (Apache 2.0) · peft (Apache 2.0) · bitsandbytes
(MIT) · vLLM (Apache 2.0) · Outlines (Apache 2.0) · scikit-learn (BSD-3)

**Serving and console** — FastAPI (MIT) · Redis (BSD) · Celery (BSD) ·
LangGraph (MIT) · React (MIT) · MapLibre GL (BSD-3)

**External processes** — ESA SNAP + pyroSAR (**GPL-3.0**) · Weights & Biases
(free tier)

**SNAP GPL note.** SNAP is invoked strictly as an external process (subprocess or
Docker service). It is never imported or linked into our code, which keeps this
codebase outside the GPL boundary. Enforced by `test_snap_is_never_imported`.

---

## 6. Papers and methods

### Read and reimplemented, not cloned

| Method | Source | What we took |
|---|---|---|
| RS-InternVL hyperparameters | BigEarthNet.txt paper §4.2 | Training configuration |
| GeoPixel adaptive partitioning | arXiv 2501.13925 | Tiling strategy |
| EarthMind HCA fusion | arXiv 2506.01667 | Design **and its negative result** |
| Earth-Agent dual-level evaluation | ICLR 2026 | Evaluation framing |
| CDVQA Qwen study targets | arXiv 2604.18429 | Comparison targets |
| Change-Agent multi-task interpretation | TGRS 2024 | Task decomposition |
| SMARTIES / DOFA band projection | — | Cut-list item, not built |

### Classical algorithms implemented from the literature

NDVI · NDWI · MNDWI · NDBI · Otsu with a bimodality gate · Refined Lee speckle
filter · phase cross-correlation · connected-component analysis · percentile
stretch · temperature scaling · GLCM texture · local coefficient of variation ·
isotonic calibration.

### Citations

- BigEarthNet.txt — arXiv 2603.29630
- CDVQA — arXiv 2112.06343
- VRSBench — arXiv 2406.12384
- GeoPixel — arXiv 2501.13925 · EarthMind — arXiv 2506.01667
- DOTA — arXiv 1711.10398
- SpaceNet 6, SpaceNet 7, RarePlanes, OpenEarthMap-SAR — dataset papers and DOIs

### ShareAlike: permitted, but not obligation-free

Three train-eligible corpora carry CC BY-SA 4.0 — **RarePlanes**,
**SpaceNet 6 MSAW**, **SpaceNet 7 / MUDS**.

SA permits commercial use. Its obligation is that **publicly shared adaptations
carry the same licence**, and it triggers on public sharing rather than on
training. Whether LoRA weights trained on SA imagery are an "adaptation" is
unsettled; Creative Commons' own guidance notes that cases where model weights
are held to be derivative works are considered quite limited, while allowing that
"adapted material" could be read to reach trained models.

Three consequences, and the second needs a decision rather than a lookup:

1. **Attribution is not optional and is not contested.** It must appear in the
   deliverable itself, not only in this file.
2. **If the adapters are released publicly, decide in advance whether they ship
   under CC BY-SA.** Deciding after release is not a position that can be
   recovered.
3. If the adapters stay private, SA is not triggered at all.

---

## 7. What we rejected, and why

Grouped by the reason. **Do not reintroduce** any of these without re-reading the
primary source and recording the finding in §8.

### NonCommercial — cannot enter a shipped weight

| Source | Finding |
|---|---|
| SARDet-100K | `LICENSE` in `zcablii/SARDet_100K` is CC BY-NC 4.0 — "for NonCommercial purposes only" |
| OSCD — change labels | CC BY-NC-SA. The imagery half is open and retained; only the groundtruth masks are excluded |
| HRSCD — change annotations | CC BY-NC-SA, same author and same split as OSCD. Unlike OSCD the imagery cannot be re-fetched, and the value here *was* the annotations |
| SatMAE weights | Pretrained on fMoW under the Functional Map of the World Challenge Public License — NonCommercial only. DOFA is the substitute |
| TinyCD / ChangeFormer | Pretrained weights are non-commercial / academic only |

### Academic-only or Google Earth-derived imagery

| Source | Finding |
|---|---|
| LEVIR-CD · LEVIR-MCI · QAG-360K | Academic-only imagery. Purged from **all** training manifests |
| DIOR · FAIR1M · NWPU-Captions · RSICD | Google Earth-derived; no shippable licence |
| VRSBench (*training* use) | DOTA-derived academic-only images must never enter shipped weights. Evaluation use is permitted — see §4 |

### No licence grant located at the source

| Source | Finding |
|---|---|
| SARLANG-1M | **No licence anywhere.** The paper (arXiv 2504.03254, 21 pp) contains zero occurrences of "licen", "copyright" or "terms of use"; the official repo has no LICENSE file; the HF dataset declares none. Its own contribution is the QA *text*, which carries no grant. The clean subset is 8,124 of 118,331 images (6.9%) and is SpaceNet 6 + OpenEarthMap-SAR, both held directly — so the text was regenerated in-house instead |
| DFC2023 | IEEE GRSS contest terms state an acknowledgement requirement and **no grant**. Imagery is SuperView-1 / Gaofen-2 / Gaofen-3 — commercial and state operators |
| DynamicEarthNet | Core imagery is **Planet Fusion**, a commercial product the paper itself calls "typically not freely available". No open grant at the source release. A third-party HF re-upload tags `cc-by-4.0`, but a re-uploader cannot grant rights Planet did not |
| xView3-SAR | Terms page 404s behind a registration wall; composites carry "© Cambrio LLC; rights reserved" despite open underlying Sentinel-1 |

### Cleared annotations over unverified imagery

| Source | Finding |
|---|---|
| LS-SSDD-v1.0 | The **annotations are Apache-2.0 and that is verified** — the repo carries a LICENSE file. **But that repository holds no imagery.** The 15 Sentinel-1 scenes come from a CAS portal whose terms were never read, and one reading recorded during the Phase 0 sweep was *"research and teaching only"*. An Apache grant over annotations conveys nothing over pixels it does not contain, and Sentinel-1 being Copernicus-open is not the same as a portal's redistribution terms being open. This entry previously read ✅ CLEARED, which overstated a clearance covering only half the dataset |

---

## 8. Standing rules

1. **Provenance Traceability Rule.** Before any dataset enters a training
   manifest, trace it to the **imagery programme** it originates from, not merely
   to the paper that published it. A derived dataset inherits the most
   restrictive licence in its ancestry. *A permissive badge on a repository says
   nothing about the pixels underneath it.*
2. **Dual-character rule.** Judge each licence question by the *use*, not by the
   project. Anything that ships is commercial use. Benchmark evaluation and
   research reporting are academic use. A benchmark number must not be
   repurposed as a commercial claim.
3. **No training on data whose licence is still ⚠.**
4. **SNAP is an external process only**, never linked.
5. **Attribution for ShareAlike sources appears in the deliverable**, not only
   in this file.

## 9. Verification log

Checks performed against primary sources. Every row is re-checkable.

| Item | Source consulted | Finding | Date |
|---|---|---|---|
| CDVQA | arXiv 2112.06343; `github.com/YZHJessica/CDVQA` LICENSE | Apache 2.0 | 2026-08-29 |
| SARDet-100K | `github.com/zcablii/SARDet_100K` LICENSE | CC BY-NC 4.0 | 2026-08-29 |
| DFC2023 | grss-ieee.org 2023 DFC page | Acknowledgement only; no grant | 2026-08-29 |
| DynamicEarthNet | arXiv 2203.12560; TUM mediatum 1650201 | Planet Fusion commercial core; no open grant | 2026-08-29 |
| SatMAE / fMoW | `github.com/fMoW/dataset` LICENSE | fMoW Challenge Public License — NonCommercial | 2026-08-29 |
| DOFA | `github.com/zhu-xlab/DOFA` | MIT code; weights carry no separate statement | 2026-08-29 |
| AROSICS | pypi.org/pypi/arosics/json | Apache 2.0 at 1.13.2; pin present in `pyproject.toml` | 2026-08-29 |
| OSCD | ieee-dataport.org (DOI 10.21227/asqe-7s69) | Split licence: imagery open, change labels CC BY-NC-SA | 2026-08-29 |
| DOTA (via VRSBench) | DOTA terms; arXiv 1711.10398 | Academic purposes only; commercial use prohibited | 2026-08-29 |
| CC BY-SA and trained weights | creativecommons.org guidance on CC-licensed works for AI training | SA triggers on public sharing of adaptations; whether weights are adaptations is unsettled | 2026-08-29 |
| Dual-character determination | Project decision | Shipping = commercial use; evaluation and research reporting = academic use | 2026-08-29 |
| `resnet50-s1` model card | **not yet read** | Open: MIT per README/deck vs CDLA-Permissive-1.0 per plan C57 (written about the ViT). Either clears | ⚠ outstanding |
