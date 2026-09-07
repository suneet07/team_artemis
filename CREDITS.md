# CREDITS.md

Every model, dataset, library and method used by SatQuery AI, with citation and licence. Maintained from day one (master plan Part 11). Status markers: `CLEARED` = verified shippable, `⚠ PENDING` = Phase 0 check outstanding, `REJECTED` = excluded with reason.

When an ISRO judge asks "what's yours and what isn't" — the answer is this file.

## Determination — this deliverable has a DUAL CHARACTER

Recorded 2026-08-29 by project decision, revised the same day. The deliverable is an SIH
entry for ISRO/SAC: it is **academic and research work**, and it is **also a candidate for
commercial deployment**. Both are true, and licence questions are judged per *use*, not
per project:

* **Anything that ships** — training corpora, adapter weights, deployed inference — is
  judged as **commercial use**. NC and academic-only sources are excluded from all of it,
  with no exception.
* **Benchmark evaluation and the numbers reported in the paper/deck** are judged as
  **academic research use**, which is what academic-only terms grant. Read-only
  evaluation on a public benchmark does not enter the weights and is not redistributed.
* **ShareAlike data is permitted** in both, with an obligation attached — see the SA note.

Consequence: the **eval-only carve-outs stand**. A benchmark may be evaluated on while
being permanently barred from training. That barrier is enforced mechanically, not by
convention — see `scripts/stage_benchmarks.py` (`eval_only`, `train_forbidden`) and
`tests/test_license_blocklist.py`.

**The limit of this position.** It rests on the evaluation genuinely staying academic. If
a benchmark number is later used as a commercial claim — in a sales deck, a procurement
response, a product page — that use is commercial and the academic-only sources cannot
back it. Report VRSBench numbers as research results, in research contexts.

## Models

| Source | Artifact | Use | Licence | Status |
|---|---|---|---|---|
| Alibaba / HF | `Qwen3-VL-4B-Instruct` | Base VLM for all four LoRA adapters | Apache 2.0 | ✅ CLEARED |
| TU Berlin / BigEarthNet | BigEarthNet-pretrained ViT (S1 + S2) | `lulc_classifier`, stretch experiment only | CDLA-Permissive 1.0 (C57) | ✅ CLEARED |
| torchgeo | Pretrained S1/S2 ResNet/ViT weights | Named substitute if BEN ViT restricted | MIT (C57) | ✅ CLEARED |
| zhu-xlab | DOFA (wavelength-aware encoder) | Second-line substitute encoder | MIT (repo). HF weights carry no separate statement; treated as MIT (C50) | ✅ CLEARED |
| sustainlab-group | SatMAE | Was a second-line substitute | Pretrained on fMoW, which ships the **Functional Map of the World Challenge Public License** — "reproduce and Share the Licensed Material… for NonCommercial purposes only". NC ancestry (C45) | ❌ REJECTED |
| Meta | SAM / SAM2 | Optional mask refinement | Apache 2.0 | ✅ CLEARED |

## Datasets

| Source | Content / use | Licence | Status |
|---|---|---|---|
| BigEarthNet.txt (arXiv 2603.29630) | Primary adaptation dataset + benchmark split | CDLA-Permissive 1.0 (verified annotation layer, C51) | ✅ CLEARED |
| BigEarthNet v2.0 / reBEN | Underlying imagery (manifest subset) | CDLA-Permissive 1.0 — confirmed; no restrictions on results of computational use | ✅ CLEARED |
| CDVQA (arXiv 2112.06343) | Change-VQA benchmark — PS-nominated | Apache-2.0, `github.com/YZHJessica/CDVQA` | ✅ CLEARED |
| VRSBench (arXiv 2406.12384) | **EVALUATION ONLY** — the grounding eval for gate G3 (acc@0.5/0.7) | Text CC-BY-4.0. Images are **DOTA-derived; DOTA permits academic purposes and prohibits commercial use**. Evaluation is academic use and therefore within the grant; training and shipped weights are not, and remain barred (C20) | ✅ CLEARED for evaluation · ❌ barred from training |
| RSVQA-LR | VQA train minority + eval | CC BY 4.0 (C54) | ✅ CLEARED |
| RSVQA-HR | First-class train + eval (closest GSD to Cartosat-2S) | USGS imagery public domain; annotations CC BY 4.0 (C54) | ✅ CLEARED |
| RarePlanes | Aircraft object grounding @30 cm | CC BY-SA 4.0 (AWS Open Data) | ✅ CLEARED |
| OpenEarthMap-SAR | Sub-metre cross-modal pairs, single-pol validation | SAR: Umbra Lab CC BY 4.0; optical: NAIP PD, IGN France CC BY 2.0, GSI Japan | ✅ CLEARED |
| SpaceNet 6 MSAW | X-band quad-pol cross-modal @0.5 m, building footprints, D1 validation | CC BY-SA 4.0 | ✅ CLEARED |
| LS-SSDD-v1.0 | ~~Ship boxes on Sentinel-1~~ — was the only clean route to SAR *object* grounding (C40) | **The annotations are Apache-2.0 and that is verified**: `TianwenZhang0825/LS-SSDD-v1.0-OPEN` carries a LICENSE file and GitHub renders the badge, which is what C53 recorded. **But that repository holds no imagery** — it is documentation plus the 6,015 ship boxes. The 15 Sentinel-1 scenes come from a CAS portal whose terms have never been read, and one reading of them recorded during the Phase 0 sweep was *"research and teaching only"*. An Apache grant over annotations conveys nothing over pixels it does not contain, and the underlying Sentinel-1 being Copernicus-open is not the same as the portal's redistribution terms being open. Unresolved, so treated as unresolved. | ⚠ **NOT STAGED** — annotations cleared, imagery unverified (C40). Previously marked ✅ CLEARED, which overstated a clearance that covered only half the dataset |
| SARLANG-1M | ~~SAR-only QA, pol-dropout training~~ | **No licence anywhere.** The paper (arXiv 2504.03254, 21 pp) contains zero occurrences of "licen", "copyright" or "terms of use"; the official repo `Jimmyxichen/SARLANG-1M` has no LICENSE file; the HuggingFace dataset declares none. C35's per-subset filter was sound for the *imagery* but SARLANG's own contribution is the QA **text**, which carries no grant — the same split that barred OSCD and HRSCD. The clean subset is only 8,124 of 118,331 images (6.9%); SARDet-100k, at 88.7%, is CC BY-NC. Those 8,124 are SpaceNet 6 + OpenEarthMap-SAR, both of which we hold directly under verified licences — so the text is regenerated in-house instead. | ❌ DROPPED |
| OSCD — imagery half (C49) | 24 Sentinel-2 AOI/date pairs. Usable as **AOI+date seeds re-fetched from Copernicus**, feeding the C46 self-generated pair pipeline | IEEE DataPort: "modified Copernicus data from 2015-2018. Original Copernicus Sentinel Data available from the European Space Agency" — open | ✅ CLEARED |
| OSCD — change labels (C49) | Pixel-level urban change groundtruth | IEEE DataPort: change labels "released under Creative-Commons BY-NC-SA. For commercial purposes, please contact the authors." **NonCommercial + ShareAlike** | ❌ REJECTED for training — see the NC consistency note |
| SpaceNet 7 / MUDS | 11M manually-annotated building footprints, expert change source (C59) | CC BY-SA 4.0 (AWS Open Data) | ✅ CLEARED |
| HRSCD — imagery half (C60) | 0.5 m aerial over France | IGN *licence ouverte* (Caveat: the 2006 images are non-redistributable and must be downloaded from IGN directly) | ✅ CLEARED |
| HRSCD — change annotations (C60) | Multi-class semantic change from Urban Atlas 2006/2012 | Author's dataset page: "This dataset is released under Creative-Commons **BY-NC-SA** licence. For commercial purposes, please contact the authors." **NonCommercial + ShareAlike** | ❌ REJECTED for training — see the NC consistency note |
| Bhoonidhi (NRSC) | Cartosat-2S + RISAT samples → India holdout v1 | ISRO terms | apply Day 1 |
| Open India proxy | Sentinel over Indian AOIs → India holdout v0 | Open | ✅ CLEARED |
| SRTM / Copernicus DEM | Terrain correction | Open | ✅ CLEARED |
| DynamicEarthNet | LULC change labels on 75 AOIs (C61) | Core imagery is **Planet Fusion**, a commercial Planet Labs product the paper itself calls "typically not freely available", released so the academic community can explore it. No open grant located at the source release. A third-party HuggingFace re-upload tags `cc-by-4.0`; a re-uploader cannot grant rights Planet did not (C45) | ❌ NOT STAGED |

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
| LEVIR-CD / LEVIR-MCI / QAG-360K | Academic-only images. Artifact-type exception reversed (C56). Purged from ALL training manifests. |
| SARDet-100K (SARLANG-1M subset) | `LICENSE` in zcablii/SARDet_100K is **CC BY-NC 4.0** — "for NonCommercial purposes only". NC cannot enter a shipped adapter (C35/C45). |
| DFC2023 (SARLANG-1M subset) | IEEE GRSS contest terms state an acknowledgement requirement and **no licence grant** — no redistribution, commercial-use or model-shipping permission. Imagery is SuperView-1 / Gaofen-2 / Gaofen-3, commercial and state operators (C35/C45). |
| SatMAE weights | Pretrained on fMoW under the Functional Map of the World Challenge Public License: NonCommercial only (C45). DOFA is the substitute. |
| DynamicEarthNet | Planet Fusion commercial core, no open grant at the source release (C61). |
| OSCD change labels | CC BY-NC-SA — NonCommercial and ShareAlike (C49). Imagery half is open and retained; only the groundtruth masks are excluded. |
| HRSCD change annotations | CC BY-NC-SA — NonCommercial and ShareAlike (C60). The same split as OSCD and by the same author: open imagery, non-commercial annotations. C60 recorded only the IGN image licence and missed the annotation licence, so the entry read CLEARED until the author's page was read verbatim. Unlike OSCD, the imagery half cannot be re-fetched — the value here *was* the semantic annotations, and the 2006 imagery is non-redistributable. |

## Note — ShareAlike is now the live risk, not NonCommercial

The NonCommercial question is settled for everything that ships: no NC or academic-only
source enters a corpus, a weight, or a deployment. What remains open is **ShareAlike**,
which is permitted but not obligation-free, and three train-eligible corpora carry it:

| Source | Licence |
|---|---|
| RarePlanes | CC BY-SA 4.0 |
| SpaceNet 6 MSAW | CC BY-SA 4.0 |
| SpaceNet 7 / MUDS | CC BY-SA 4.0 |

SARLANG-1M inherits SA through its SpaceNet 6 portion.

SA permits commercial use. Its obligation is that **publicly shared adaptations carry the
same licence**, and it triggers on public sharing rather than on training. Whether LoRA
weights trained on SA imagery are an "adaptation" is an unsettled question; Creative
Commons' own guidance notes that cases where model weights are held to be derivative
works are considered quite limited, while acknowledging the concepts of "adapted
material" and "technical modification" could be read to reach trained models.

Practical position, and it needs a decision rather than a lookup:

* Attribution for all three is **not** optional and is not contested. It must appear in
  the deliverable, not only in this file.
* If the adapters are released publicly, decide in advance whether they ship under
  CC BY-SA. Deciding after release is not a position that can be recovered.
* If the adapters stay private, SA is not triggered at all.

## Note — VRSBench is evaluation-only

**VRSBench** (gate G3 grounding) rests on an actual grant: DOTA's terms permit academic
purposes explicitly, so academic evaluation is the use DOTA allows — licensed, not
merely tolerated. It is barred from **training** because that half of the deliverable is
commercial, and that barrier is enforced by `tests/test_license_blocklist.py` rather
than by convention.

## Note — RSVQA train splits are used, and disclosed

The problem statement assigns roles: *"BigEarthNet.txt will serve as the primary
dataset for adapting… VRSBench and RSVQA will be used to evaluate… CDVQA will be
used to evaluate."* This section states plainly what we do with RSVQA, so it is
something we declared rather than something a reader discovers.

**What we do.** The `rs_vqa` corpus draws on three sources. BigEarthNet.txt is
the largest single training source; both RSVQA contributions use their
**official train splits**, and evaluation uses their **official test splits**,
untouched.

| Source | Rows | Native GSD |
|---|---|---|
| **BigEarthNet.txt** | **40,000** | 10 m |
| RSVQA-HR (train split) | 20,529 | 0.30 m |
| RSVQA-LR (train split) | 10,000 | 10 m |

**One question type is excluded, on evidence.** RSVQA-HR's `area` answers are
computed from OpenStreetMap polygons and are not usable: 62% are `0m2` where OSM
has nothing mapped, 8% (275 rows) claim more building area than the 6,088 m²
tile physically contains -- the largest is 21,264 m² -- and 39 more are
implausibly small, including a "school" of 6 m². Roughly 29% carry a believable
value. 3,471 rows dropped. BEN.txt asks the same concept as a bounded choice
over rasterised CORINE maps and does not share the fault, so the capability is
retained.

**RSVQA's size categories are kept, with their thresholds stated in the prompt.**
The two scales define the same words 30x apart (paper Table I: "small" is under
3,000 m² at 10 m and under 100 m² sub-metre). Rather than drop 3,067 rows, each
row now carries its own rule in the prompt prefix, keyed on source. BEN.txt gets
none -- it states its ranges inline in the question, so asserting RSVQA's
convention over it would be false.

BigEarthNet.txt is primary in the ordinary sense of the word: it is the chief
source, ahead of the next-largest by two thirds. *Primary* presupposes secondary
sources; had the PS meant BEN-only it would have said "the dataset for
adapting".

**Why this is not benchmark leakage.** RSVQA ships author-defined splits as
separate files. Train and test share **no questions and no images** -- verified,
not assumed: LR train 57,223 against test 10,004, overlap zero on both counts.
Training on the train split and reporting on the test split is how every
published RSVQA number was produced, including the baselines we are compared
against. RSVQA-HR's fourth split, `test_phili` (Philadelphia held out as a
different city), is a geographic generalisation test and is never touched.

**Why RSVQA and not VRSBench.** The eval role alone is not what excludes a
dataset here. VRSBench is excluded because the eval role comes *with* a licence
problem -- DOTA-derived academic-only imagery, inside a deliverable that ships
weights. RSVQA has no such problem: CC-BY-4.0 annotations over USGS
public-domain and Sentinel-2 imagery.

**Why the RSVQA-HR share is what it is.** RSVQA-HR is USGS aerial at 0.1524 m,
downsampled 2x to ~0.30 m. It is the **only sub-metre optical imagery in the
entire inventory**; every other optical source is 10 m Sentinel-2, and the hidden
set is Cartosat-2S at metre class. A smaller share would be easier to defend on
paper and worse on the thing that is actually scored.

## Standing licence rules

1. **Provenance Traceability Rule (C45):** Before any dataset enters a training manifest, trace it to the imagery programme it originates from, not merely to the paper that published it. A derived dataset inherits the most restrictive licence in its ancestry.
2. **SNAP rule:** external process only, never linked.
3. **No training on data whose licence is still ⚠.**
4. **Per-account training policy:** one adapter per person, on that person's own account; no cross-account relay of runs.
5. **Dual-character rule (2026-08-29):** judge each licence question by the *use*, not by the project. Anything that ships — corpora, weights, deployed inference — is commercial use and excludes NC and academic-only sources absolutely. Benchmark evaluation and research reporting are academic use and may rely on academic-only grants. A benchmark number must not be repurposed as a commercial claim. ShareAlike sources are permitted in both and carry an attribution obligation in the deliverable itself.

## Licence verification log

Checks performed 2026-08-29 against primary sources. Each row is re-checkable.

| Item | Source consulted | Finding |
|---|---|---|
| CDVQA | arXiv 2112.06343; github.com/YZHJessica/CDVQA `LICENSE` | Apache-2.0 |
| SARDet-100K | github.com/zcablii/SARDet_100K `LICENSE` | CC BY-NC 4.0 |
| DFC2023 | grss-ieee.org 2023 DFC page | Acknowledgement requirement only; no grant. SuperView-1 / Gaofen-2 / Gaofen-3 |
| DynamicEarthNet | arXiv 2203.12560; TUM mediatum 1650201 | Planet Fusion commercial core; no open grant at source |
| SatMAE / fMoW | github.com/fMoW/dataset `LICENSE` | Functional Map of the World Challenge Public License — NonCommercial |
| DOFA | github.com/zhu-xlab/DOFA | MIT code; weights carry no separate statement |
| AROSICS | pypi.org/pypi/arosics/json | Apache-2.0 at 1.13.2; pin `arosics>=1.0.0` present in `pyproject.toml:38` |
| OSCD | ieee-dataport.org/open-access/oscd-onera-satellite-change-detection (DOI 10.21227/asqe-7s69) | Split licence: imagery modified Copernicus (open); change labels CC BY-NC-SA |
| DOTA (via VRSBench) | DOTA dataset terms; CVPR 2018 arXiv 1711.10398 | Academic purposes only; commercial use prohibited |
| CC BY-SA and trained weights | creativecommons.org/using-cc-licensed-works-for-ai-training-2 | SA triggers on public sharing of adaptations; whether weights are adaptations is unsettled |
| Dual-character determination | Project decision, 2026-08-29 | Shipping = commercial use; benchmark evaluation and research reporting = academic use. Eval-only carve-outs stand on that basis |

## Citations to add before submission

- BigEarthNet.txt paper: arXiv 2603.29630
- VRSBench: arXiv 2406.12384 · CDVQA: arXiv 2112.06343 · SARLANG-1M: arXiv 2504.03254
- GeoPixel: arXiv 2501.13925 · EarthMind: arXiv 2506.01667
- SpaceNet 6, SpaceNet 7, RarePlanes, OpenEarthMap-SAR dataset papers/DOIs (fill at staging time)
- LS-SSDD-v1.0 is **not** staged and needs no citation unless its imagery terms clear first
