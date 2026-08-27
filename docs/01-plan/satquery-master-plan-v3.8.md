# SatQuery AI — Master Plan v3.8

**SIH26167 · ISRO / SAC · Space Technology**
*Single source of truth for architecture, pipelines, sources, and implementation sequence.*
*v3.8 — licence verification COMPLETE. All items resolved. Supersedes all earlier versions.*

---

## v3.8 change log — verification complete, and C33 reversed

External licence verification returned on 2026-08-26. **Every outstanding item is now resolved.** Three good outcomes, one bad one, and one flaw in our own earlier reasoning that the bad one exposed.

| # | Change | Why |
|---|---|---|
| **C53** | **LS-SSDD-v1.0 CLEARED — Apache-2.0.** Ships added to the G3 trained vocabulary. | Object grounding now covers **regions, buildings, aircraft AND ships**. Ships are SAR objects on Sentinel-1 imagery — directly relevant to the RISAT half of the hidden set, and the closest analogue to a real ISRO maritime use case. |
| **C54** | **RSVQA CLEARED — CC BY 4.0 (both splits). The HR dispute is resolved in our favour: RSVQA-HR is USGS High-Resolution Orthoimagery at 15 cm over the US northeast, public domain.** C27 stands unchanged. | The plan was right and the secondary source was wrong — blogs conflate the two splits because they share a paper. RSVQA-HR remains our only sub-metre optical VQA supervision and our closest public proxy to Cartosat-2S. |
| **C55** | **TinyCD and ChangeFormer pretrained weights are RESTRICTED — non-commercial/academic only. Both excluded.** `change_map` rebuilds on a **TorchGeo MIT-licensed backbone** (SSL4EO-S12 / SeCo). | Author repositories restrict use to academic and non-commercial purposes. These checkpoints cannot ship inside a government deliverable — and "codes and models" means they would. |
| **C56** | **C33 REVERSED. LEVIR-CD, LEVIR-MCI, SECOND and QAG-360K are dropped from `change_map` as well as `change_vqa` — i.e. removed entirely.** | **C33's artifact-type distinction does not survive contact with C55.** It permitted restricted data for `change_map` on the reasoning that its output is a *mask*, not weights. But `change_map` is a fine-tuned Siamese CD model, and that fine-tuned model **is** a shipped weight under "codes and models". The distinction only held if we shipped outputs and not the model — which we don't. The C55 finding made this visible by killing the "fall back to the pretrained backbone" escape hatch that v3.3 relied on. |
| **C57** | **BigEarthNet ViT weights CLEARED — CDLA-Permissive-1.0, hosted by BIFOLD/TU Berlin.** `lulc_classifier` is unblocked and the §5.6 stretch experiment's licence precondition is satisfied. **TorchGeo foundation weights CLEARED — MIT**, confirmed as a substitute. | Removes the last ⚠ on the pretrained-weight path. |
| **C58** | **AROSICS pinned to ≥ 1.0.0.** | Apache-2.0 on modern releases, but **pre-1.0 releases were GPL-3.0**. Version pin is the control; without it a resolver could pull a GPL build into the deliverable. |

| **C59** | **SpaceNet 7 / MUDS added — CC BY-SA 4.0, AWS Open Data.** 101 AOIs across 6 continents, 24 monthly mosaics each, ~40,000 km², **11M manually-annotated building footprints with tracking IDs**, 4 m Planet. Becomes the **primary expert-labelled change source**. | Largely repairs the C56 quality loss. Change labels come from **differencing manually-annotated footprints**, not index thresholding. 24 timestamps → 276 possible pairs per site across 60+ labelled sites = tens of thousands of expert-labelled bi-temporal pairs, versus SECOND's 2,968. Same licence family and AWS channel as SpaceNet 6, already staged. |
| **C60** | **HRSCD added — IGN *licence ouverte*, with one caveat.** 0.5 m aerial over France, multi-class semantic change (Urban Atlas classes). **The 2006 images are non-redistributable and must be downloaded directly from IGN**; the 2012 images are open. | Our only sub-metre *semantic* change source. We train on it but never redistribute the imagery, which the caveat permits. Record the split-licence status in `../../CREDITS.md`. |
| **C61** | **DynamicEarthNet flagged ⚠, NOT added.** 75 AOIs, daily Planet 3 m multispectral, monthly 7-class LULC labels. | Attractive — but **commercial Planet Fusion data constitutes its core**, and the licence is unverified. Goes to the verification job (§2.2b) before any use. Do not stage it. |

> **[v3.8] Licence confidence on C59–C61, stated honestly.** SpaceNet 7 is verified from the MUDS and challenge papers directly. HRSCD is verified from the author's own dataset page, including the 2006 caveat. **DynamicEarthNet is unverified.** All three should still pass through the same external verification job that produced the C53–C58 results — seven rounds of review each found something the previous round missed, and these have had one.

> **[v3.8] C56 makes the story simpler, not weaker.** The entire change pipeline — `change_vqa` and `change_map` — now trains end-to-end on data we fully control: **self-generated Sentinel bi-temporal pairs over Indian AOIs (C46) plus OSCD (C49), initialised from MIT-licensed TorchGeo weights.** No academic-only data anywhere in the change path, no artifact-type caveat to explain, and the training imagery is Indian rather than aerial photography of Hangzhou. One sentence to a judge instead of three.

> **[v3.8] Cost of C56 — largely recovered by C59/C60.** The original concern was that index-differenced labels are weaker supervision than SECOND's expert pixel annotations. **SpaceNet 7 restores expert labels on a clean licence** (manually-annotated footprints, differenced), and **HRSCD restores sub-metre semantic change**. Residual gaps: SpaceNet 7 is buildings-only at 4 m; HRSCD is France-only. Self-generated Sentinel data remains in the mix for Indian domain match and unlimited volume. `change_stats` (D4) is unaffected either way, since it is arithmetic on whatever mask exists.

---

## v3.7 change log — the foundation is verified

| # | Change | Why |
|---|---|---|
| **C51** | **BEN.txt annotation layer CLEARED — CDLA-Permissive 1.0, confirmed on the official site.** AROSICS CLEARED — **Apache-2.0, not GPL**, so it can be imported normally rather than isolated like SNAP. Three items remain open (§2.2b) and are handed to named owners. | BEN.txt is the primary training source for every adapter; its licence was **assumed** through six revisions on the reasoning that it inherits BigEarthNet's. That is the exact inference pattern that failed for CDVQA-from-SECOND and VRSBench-from-DOTA. It is now verified. The remaining three require opening download pages and reading terms — not resolvable from public search. |
| **C52** | **RSVQA-HR imagery source flagged as DISPUTED.** Verify before relying on it. | We recorded it as USGS 15 cm, which is the whole basis of C27's "closest GSD match to Cartosat-2S". At least one secondary source describes it as Sentinel-2 over the Netherlands — the description of RSVQA-**LR**. If that is right, C27 collapses and we lose our only sub-metre optical VQA source. |

> **[v3.7] Standing instruction to the integration lead.** Seven revisions of licence review each found something the previous one missed, and two confident conclusions in this plan were later reversed. **Treat every ✅ in this document as a claim to re-verify, not as authority.** §2.2a gives the procedure; apply it to entries marked clean as well as to new datasets.

---

## v3.6 change log — provenance traced to the root

### The rule that should have been here from v1

**Every problem found across four licence sweeps came from the same place: a dataset built from another dataset.** CDVQA from SECOND. SECOND-CC from SECOND. QAG-360K from SECOND + LEVIR + Hi-UCD. SARLANG-1M from four sources. SARDet-100K from *ten*. VRSBench from DOTA. NWPU-Captions from RESISC45.

Every clean dataset shares the opposite property: it is built **directly on an open imagery programme** — Copernicus, NAIP, USGS — or on commercial imagery its owner deliberately released under CC.

| # | Change | Why |
|---|---|---|
| **C45** | **PROVENANCE TRACEABILITY RULE — standing Phase 0 procedure.** Before any dataset enters a training manifest, trace it to the **imagery programme** it originates from, not merely to the paper that published it. A derived dataset **inherits the most restrictive licence in its ancestry**, and papers almost never state this. Record the full chain in `../../CREDITS.md`. | Four sweeps, four cascades. This rule catches all of them prospectively and lets the team check future datasets without another round of research. |
| **C46** | **The SECOND cascade documented. `change_vqa` training moves to self-generated Sentinel bi-temporal data + OSCD.** CDVQA stays for **evaluation** (PS-nominated). SECOND-derived data is confined to `change_map` under the C33 artifact rule. | SECOND has **no licence statement of any kind** — not academic-only, *nothing*. Just Google Drive links and a contact email. **Absence of a licence grants fewer rights than a restrictive one.** And CDVQA, SECOND-CC and QAG-360K all descend from it, so what looked like three sources was one source three times. G4 had no clean training data at all. |
| **C47** | **SARLANG-1M filtered to its clean subset.** Use **only** the SpaceNet 6- and OpenEarthMap-SAR-derived portions. **DFC2023 and SARDet-100K portions excluded.** | Its four sources split two clean, two not. DFC2023 is IEEE GRSS contest data — registration-gated, contest terms, SuperView-1/Gaofen-2/Gaofen-3 commercial Chinese imagery. SARDet-100K is a standardisation of **ten** existing SAR datasets released "for research purposes". Note we already hold SpaceNet 6 and OpenEarthMap-SAR independently, so the clean subset costs us nothing new. |
| **C48** | **QAG-360K moved under the C33 rule** — `change_map` only, never `change_vqa`. | It draws imagery from Hi-UCD, SECOND **and LEVIR-CD**. After C33 removed LEVIR-CC from `change_vqa`, QAG-360K was quietly reintroducing the same LEVIR imagery through the back door. The line we drew in v3.3 leaked. |
| **C49** | **OSCD added, with its Sentinel-1 extension.** ONERA, 24 Sentinel-2 pairs, openly available as a benchmark, on IEEE DataPort open-access. The Ebel et al. S1 extension adds SAR for **optical–SAR multimodal change detection**. | Small, but clean, and the only cross-modal *change* data in the whole inventory. Sentinel-2 also matches BEN.txt's domain exactly. |
| **C50** | **Five residual checks enumerated with owners and deadlines** (§2.7). Nothing trains on them until cleared. | Named so they cannot be forgotten: LS-SSDD annotations, RSVQA-LR annotation layer, BEN.txt annotation layer, pretrained weights (BEN ViT / TinyCD / ChangeFormer / torchgeo), AROSICS licence family. |

> **[v3.6] Self-generated change data is an upgrade, not a workaround.** Sentinel-1 and Sentinel-2 are fully open under Copernicus and cover India. We already built the acquisition machinery for India holdout v0. Two dates over the same Indian AOI, change labels from established index differencing, captions via the BEN.txt template-plus-augmentation recipe. Result: **unlimited volume, perfect licence, and Indian imagery** — versus 0.5 m aerial photographs of Hangzhou. It fixes a licence problem and a domain-gap problem in one move.

> **[v3.6] Where the project actually stands on data.** Five of six gates have clean data with room to spare. G4 needed a workaround that takes roughly one week of the geospatial engineer's time and improves domain match as a side effect. **Data volume has never been the constraint** — BEN.txt alone is 464k pairs and we use ~2%.

---

## v3.5 change log — compute is no longer the binding constraint

| # | Change | Why |
|---|---|---|
| C41 | **§5.5 rebuilt around ~50 hours on a single A100-80GB.** 31 h scheduled, **19 h reserve**. bf16 and FlashAttention-2 on, QLoRA dropped, batch size raised. T4 tables and triage ladder retained as the access-falls-through fallback. | Confirmed access. The A100 is ~6–8× a T4 for this workload — 5× raw throughput plus bf16 and FA2, both blocked on Turing. The four-adapter budget drops from 200–320 T4-h to ~31 A100-h. |
| C42 | **Reserve allocation rule:** if ≥15 h remain at end of Phase 2, spend them raising `rs_ground_caption` sample count — **not** on more reruns. | Referring detection is our weakest §6.1 target, and it is weak for exactly one reason: RS-InternVL trained per-task on ~100% of a ~347k-pair corpus, we train on ~2%. More data is the only lever that moves it. |
| C43 | **Three-composite optical input (C22) must be ablated before commitment.** Estimated 5–8 h of the 50. | It raises image tokens 2–3× on multispectral samples. Now that hours are a visible finite budget, this decision has a price tag and should be measured rather than assumed. |
| C44 | **Five non-renewable-budget guard rails** added to §5.5, plus a weekly burn-down owned by the integration lead. | 50 hours is enough for one clean pass *plus* reruns. It stops being enough the moment a run is started on an unsmoked config. |

> **[v3.5] The 60/40 rule is the point.** 31 hours scheduled against 19 in reserve. Nobody's first training run is their last, and a compute plan that consumes 45 of 50 hours on the happy path is not a plan — it is a hope. If the estimates hold after the Phase 0 timing test, this budget is comfortable. If they don't, the reserve absorbs it.

> **[v3.5] What is now the binding constraint.** Not compute, and not data. **Execution time and integration risk.** The two unrun Phase 0 items — the vLLM serving smoke test (item 7) and the 200-step timing measurement (item 12) — can still invalidate assumptions the whole schedule rests on. Everything else is downstream of those two numbers being real.

---

## v3.4 change log — object grounding partially recovered

A second sweep, working backwards from **licence-clean imagery bases** rather than searching for "object datasets" as a category, found sources the first sweep missed. **C32 is partially reversed.**

### The unlock

Two imagery bases are fully permissive, and anything built on them inherits that:

- **Copernicus Sentinel** — EU law grants free access for reproduction, distribution, communication to the public, adaptation, modification and combination with other data. Attribution notice only (`Copernicus Sentinel data [Year]`). **No non-commercial clause.**
- **NAIP** — public domain (CC0), usable for any purpose without attribution.

| # | Change | Why |
|---|---|---|
| C36 | **RarePlanes added to `rs_ground_caption`. CLEARED — CC BY-SA 4.0**, AWS Open Data Program. 253 Maxar WorldView-3 scenes, 112 locations, 2,142 km², **14,700 hand-annotated real aircraft at 30 cm**, plus ~630,000 synthetic annotations across 50,000 synthetic images. | Genuine object-level grounding supervision on a shippable licence. Aircraft at 30 cm is directly Cartosat-scale. The synthetic half is a free data-volume multiplier for a capability we had written off. |
| C37 | **OpenEarthMap-SAR added. CLEARED — SAR is Umbra Lab under CC BY 4.0**; optical is NAIP (public domain), IGN France (CC BY 2.0) and GSI Japan. 5,033 images, 35 regions across Japan/France/USA, **0.15–0.5 m**, VV **or HH single-pol**, 8 classes incl. building/water/road, expert-aligned optical–SAR pairs. Public on Zenodo. | **A second sub-metre co-registered cross-modal source**, independent of SpaceNet 6 — and single-pol, which is exactly the C2/C26 case. Geographically far more diverse than SpaceNet 6's single city. |
| C38 | **xView3-SAR REJECTED.** Not added. | Its terms-and-conditions page 404s, download is behind a registration wall, and the delivered composites carry "© Cambrio LLC; rights reserved" even though the underlying Sentinel-1 is open. Its sibling xView1 is CC BY-NC-SA. Unverifiable terms plus a rights-reserved processing layer is exactly what must not enter a government deliverable. |
| C39 | **G3 rescoped again — from "region only" to "regions, buildings and aircraft".** The honest claim is now narrower than "all objects" but wider than v3.3, and it is a promise we can keep. | See §5.3a. Ships remain uncovered pending the LS-SSDD check (C40). |
| C40 | **LS-SSDD-v1.0 flagged for Phase 0 check** — 15 Sentinel-1 scenes cut into 9,000 sub-images with 6,015 expert ship boxes, VV+VH. Imagery is Copernicus-clean; the annotation repo's terms need reading. | The only remaining clean route to ship objects. Low effort to verify, meaningful if it clears. |

> **[v3.4] What still is NOT covered, stated plainly.** Vehicles, storage tanks, bridges, harbours and the rest of the DOTA/DIOR class vocabulary. Those exist only in Google Earth-derived datasets. `object_box_fallback` (§4.6.9) remains the answer for them, and it stays in the build.

---

## v3.3 change log — licence review completed, and one correction

The Phase 0 licence review was executed early. **One prior conclusion (C28) was wrong and is reversed.**

| # | Change | Why |
|---|---|---|
| C31 | **C28 REVERSED. SpaceNet 6 MSAW added as a first-class cross-modal source.** ~0.5 m Capella **X-band quad-pol** SAR co-registered with ~0.5 m Maxar optical, ~48,000 building footprints, 120 km², **CC BY-SA 4.0**. | v3.2 claimed the high-resolution optical–SAR gap "is not fixable by finding another dataset." **That was overconfident and wrong.** SpaceNet 6 is close to purpose-built for our case — and critically it is **X-band**, the exact RISAT-2B/2BR1 case C2 was written to defend against and which we otherwise could not test at all. |
| C32 | **G3 formally scoped to referring *region* grounding.** Object-level grounding is out of scope, stated in plan, validator and deck. Plus a **deterministic object-box fallback** (§4.6.9) so an object query returns partial credit rather than zero. | All four C25 candidates failed. DIOR, FAIR1M, RSICD and NWPU-Captions all trace to Google Earth imagery with no shippable licence. Not a search failure — the structure of the high-resolution optical RS field. The PS's own grounding example is region-level, so we are narrow, not blind. |
| C33 | **LEVIR handled by artifact type — Option C.** LEVIR stays for `change_map`, whose output is a **mask**; LEVIR leaves `change_vqa`, whose output is **shipped weights**. Disclosed in CREDITS either way. | LEVIR-CD carries the *identical* academic-only, no-commercial restriction that made us remove VRSBench — and v3.2 left it in the training mix. One standard for VRSBench and another for LEVIR would not survive a judge reading CREDITS. Option C draws the line at the real exposure: distributed weights. |
| C34 | **Licence status resolved for every major source** (§2.2). BEN is **CDLA-Permissive 1.0**, which explicitly imposes no restrictions on results of computational use — models trained on it are unencumbered. | Removes the ⚠ blocker on our primary dataset and confirms the plan rests on a shippable foundation. |
| C35 | **SARLANG-1M licence is per-subset, not single.** Its SAR is drawn from SpaceNet 6, DFC2023, OpenEarthMap-SAR and SARDet-100K. | Must be resolved subset by subset. The SpaceNet 6 portion is CC BY-SA and clean; the others need individual checks before entering a training manifest. |

> **[v3.3] The framing that resolves most of the licence anxiety.** The PS *nominates* VRSBench, RSVQA and CDVQA for evaluation by name. **Evaluating on ISRO's own nominated datasets is compliance, not exposure.** The exposure was only ever about *shipping weights trained on* restricted imagery — a narrower question, now closed by C20 and C33.

> **[v3.3] We are not short on data.** BEN.txt alone is 464k pairs and 9.6M annotations on a permissive licence, and every triage rung in §5.5 *cuts* samples. The constraint has always been T4 hours, never data volume. The two genuinely thin slots were object grounding (closed by scoping, C32) and clean-licence bi-temporal change (closed by C33).

---

## v3.2 change log — data coverage gaps

A line-by-line audit of the training mix against the PS's **Defined Input Scope** and **Representative Queries** found four gaps. One was self-inflicted in v3.1.

| # | Change | Why |
|---|---|---|
| C25 | **Object-level grounding data restored via a permissive source** (§5.3a). Phase 0 licence review must clear one of DIOR / FAIR1M / NWPU-Captions / RSICD. If none clears, G3 is scoped to **region** grounding and we say so openly. | **Self-inflicted in v3.1.** C20 removed VRSBench — correct for the deliverable — but VRSBench was the *only* object-level source. BEN.txt's referring annotations target CORINE **land-cover regions**, not objects. The PS asks for "land-cover **and major objects**". |
| C26 | **SARLANG-1M added to the `rs_vqa` mix, plus modality dropout** — some BEN samples trained with S2 withheld, so the adapter has seen SAR alone. | The PS input scope permits "one optical/multispectral **or SAR** image" for VQA, captioning and grounding. BEN.txt is *always* co-registered S1+S2, so a SAR-only single image was out of distribution for `rs_vqa`. |
| C27 | **RSVQA-HR promoted to a first-class training and evaluation set.** | USGS 15 cm, public domain, and by far the closest GSD match to Cartosat-2S in the entire inventory — every other optical source is 10 m Sentinel. We were using only RSVQA-LR. Free domain-gap mitigation we were leaving on the table. |
| C28 | ~~**Superseded by C31 — see v3.3 log**~~ **G5's data honesty stated plainly** (§5.3, §1.3): the deterministic path *carries* G5; `optsar_fusion` is a bonus we cannot guarantee. | The only co-registered optical–SAR supervision anywhere in the plan is BEN.txt at 10 m, against a metre-class hidden set. SARLANG-1M is SAR image–text, not cross-modal pairs. *(The struck claim: "no high-resolution optical–SAR paired training data is available at any price." **This was wrong — see C31.** The G5 conclusion it supported still stands.)* |
| C29 | **Storage estimate corrected** from ~40 GB to **~100–140 GB**, with a staged download order. | "60–80k samples" is *per adapter*. Four adapters means 240–320k samples total. C13's estimate was light by 2–3×. |
| C30 | **Synthetic GeoTIFF fixtures in CI.** | GeoTIFF/TIFF is a mandatory input format, but only reBEN and Bhoonidhi ship GeoTIFF — CDVQA and the LEVIR family are PNG. Without fixtures the ingest path is only ever exercised on two sources. |

> **[v3.2] Open question for the integration lead.** The PS describes the ISRO set as "Cartosat-2S optical **and** RISAT SAR image **pairs**" — which reads as cross-modal only, with no bi-temporal component. If confirmed, change understanding is graded solely on CDVQA, and our LEVIR-family data (~0.5 m Google Earth) is a *better* GSD match than anything in the cross-modal mix. Ask, because it changes where effort is worth spending.

---

## v3.1 change log — problem-statement and benchmark corrections

Eight further decisions, from a close reading of the official PS wording and the actual numbers in the BEN.txt paper (arXiv 2603.29630).

| # | Change | Why |
|---|---|---|
| C17 | **Grounding is now mandatory in practice. Removed from every fallback path.** Triage rung 4 no longer degrades `rs_ground_caption` to caption-only. | The PS states the ISRO set carries "reference answers, labels, bounding boxes, or masks, as applicable". Bounding boxes on the hidden set mean grounding is scored there regardless of which G3 option we nominally elect. Electing captioning would score zero on that portion. |
| C18 | **`change_map` comes off the cut list; every mask-producing tool must write full-scene-resolution GeoTIFFs in the source CRS.** | Masks are a graded reference type on the hidden set. Our deterministic masks are therefore scoreable outputs, not just evidence — which raises the value of the whole non-learned path. |
| C19 | **Parameter enforcement gate (new §4.5.4)** between planner and executor. Manifests declare permitted names and ranges; violations are rejected, never clamped; the check is recorded in the trace. | The PS names "permitted parameters" as one of only four graded trace fields, and requires the controller "configure only permitted task parameters". We recorded parameters but never validated them — we would have faithfully documented our own bad values. |
| C20 | **VRSBench dropped from training entirely; eval-only.** BEN.txt becomes the clear majority of every adapter's mix; other benchmark train splits used in the minority and disclosed. | The PS assigns BEN.txt to *adaptation* and VRSBench/RSVQA/CDVQA to *evaluation*. Separately, "codes and models" is a listed deliverable — shipping weights trained on DOTA-derived academic-only imagery to a government agency is a compliance exposure we can simply avoid. |
| C21 | **Graded-field summary block promoted to the top of the trace schema.** | Exactly four fields are evaluated: selected task, models/tools, permitted parameters, outputs. They must be unambiguous and machine-readable, not nested three levels deep in a superset. |
| C22 | **Multi-composite optical input** — true-colour + false-colour NIR + SWIR composite passed as multiple images. | Recovers most of the 10 m/20 m spectral range that RS-InternVL got from dedicated encoders, without breaking the C1 uniform input contract or vLLM serving. The BEN.txt benchmark specifically tests hard CLC classes that need bands beyond RGB. |
| C23 | **§6.1 targets rewritten against the paper's real numbers**, including dual MCQ reporting and a realistic referring-detection band. | The paper's "Qwen" row is 8B **zero-shot**, not a fine-tuned bar. The real bar is RS-InternVL, trained per-task on ~100% of a 347k-pair corpus. We train on ~2%. Also: the MCQ tasks we correctly drop are Qwen's *strongest*, so our restricted-set number is not comparable to the 51.49 headline. |
| C24 | **D2 downgraded further; expect the reframe branch.** | The paper's two referring tables use different sample sets, and the point prior *hurts* LLaVa (23.04→17.27) and EarthMind-rgb (21.90→9.35). On Qwen specifically the gain is the smallest of any model that gained: 15.60→20.42. |

> **Two items still outstanding with the integration lead.** (a) The PS contains a literal placeholder — *"Add 'Evaluation/Judging Criteria' table here"* — so the weighting between public benchmarks and the ISRO set is unpublished. Chase it; it decides whether we optimise for Sentinel benchmarks or the Cartosat/RISAT holdout. (b) Scores are "normalised before combining different metrics", which rewards breadth over a single peak — so every one of the six gates must produce a scoreable output before any one of them gets polished.

---

## v3 change log — what changed and why

Sixteen decisions were made in this revision. Each traces to a specific vulnerability found in review.

| # | Change | Why |
|---|---|---|
| C1 | **All four adapters are now plain LoRA on the unmodified base architecture.** The RS-InternVL dual-ViT multi-encoder recipe is demoted to an optional Phase 3 stretch experiment (§5.6) served outside vLLM. | The old §5.2 architecture (extra frozen ViTs + projections) is not a LoRA and cannot be served by vLLM multi-LoRA. Training and serving plans were incompatible and would have collided at integration in Phase 3–4. |
| C2 | **Single-pol / X-band SAR fallback is now a first-class input path**, with polarisation-dropout training and band-parameterised, warning-only sanity ranges. | The old `(VV,VH,VV/VH)` stack is undefined for single-pol data. RISAT-1 FRS modes include single-pol HH; RISAT-2B/2BR1 are X-band. Hard C-band asserts could have refused the graded hidden-set input. |
| C3 | **Compute triage ladder with numeric thresholds**, decided within 48 h of the Phase 0 timing test. Plus a pre-authorised ~$60–100 paid-GPU insurance budget. | The 200–320 T4-h estimate had a 60 % spread and no failure branch. One serial adapter (`rs_ground_caption`, 60–120 h) could eat a phase. |
| C4 | **India holdout v0 is a proxy set needing no approvals** (open data over Indian AOIs, resampled). Real Bhoonidhi data swaps in if/when granted. | Bhoonidhi access was an unowned external dependency gating the plan's single Critical risk. |
| C5 | **D2 is reclassified as a hypothesis and gets a free zero-shot ablation in Week 1** (~200 samples, prior vs no prior, no training). | The cited 25.16 → 38.95 mIoU numbers compare two *different tasks*, not prior-vs-no-prior on one task. The causal claim was unsupported. |
| C6 | **D1 and D3 move from Phase 3 to Phases 1–2.** The `optsar_fusion` adapter becomes a Phase 3 *upgrade* over the deterministic D1 floor. | The differentiators are the cheapest items in the plan (rule table + band lookup, no training) yet were scheduled last, behind everything that can slip. |
| C7 | **Headless eval mode from Week 2**: one command, `(images, question) → (answer, trace.json)`, no frontend/queue/network, CPU fallback for deterministic tools, green in CI. Submission spec is a Phase 0 action item. | No packaging spec existed for how the hidden set is actually run. A stack that doesn't execute in the eval environment scores zero, indistinguishably from a broken model. |
| C8 | **Per-benchmark answer formatter + official scorer scripts integrated in Phase 1.** | A correct answer in the wrong shape scores zero. Reimplemented metrics routinely differ from official ones by large margins. |
| C9 | **Otsu now sits behind a bimodality gate with fixed physical fallback thresholds**; the chosen path is recorded in the trace. | Otsu fails silently on unimodal tiles and its output feeds D1, D2, and D4 — a bad threshold corrupted all three differentiators with no signal. |
| C10 | **Calibration target changed to P(answer correct)** via isotonic/logistic regression over component signals, fit on ~500 labelled end-to-end outputs. `lulc_classifier` demoted from "primary confidence source". | min/product over heterogeneous signals is not calibrated even if each part is; the BEN-trained classifier's temperature fit does not transfer to 2 m Cartosat tiles. |
| C11 | **Rules-first router**, LLM only as tie-breaker; routing eval set built in Phase 0 and zero-shot routing measured in Week 1. | The router was a zero-shot LLM single point of failure, unvalidated until Phase 2, and G6 is graded on its output. |
| C12 | **Latency SLA split**: preprocessing is an async job (≤ 5 min/scene) with progress UI; the < 20 s target applies to query time on a prepared `ImageBundle`. | The old single 20 s target was impossible on the cold path (full SNAP chain alone is minutes). |
| C13 | **Manifest-first data staging**: pick the 60–80k training patch IDs from metadata *first*, download only those + benchmark splits, stage as one private Kaggle Dataset (~40 GB). Full-corpus downloads banned. | The full corpora are several hundred GB and do not fit free-tier disk. |
| C14 | **All ⚠ licences resolved in Phase 0** (assigned, one day) with named substitute encoders. | Unverified licences included the primary training dataset and — under the old plan — the backbone the whole recipe depended on. |
| C15 | **Degenerate-input routing branch** for pan-only / RGB-only optical, plus texture/morphology tools and an explicit explained-refusal path. | Cartosat-2S is most commonly distributed as panchromatic; the old plan had no defined behaviour when both NDVI and NDWI are impossible. |
| C16 | **Process fixes**: risk table now has trigger / threshold / owner / deadline; a pre-agreed Week-6 cut list exists; team load rebalanced; capacity stated in person-hours; per-person-per-account training (no cross-account relay of a single run). | Mitigations were techniques, not decisions; one seat carried double load; cross-account relay of one run violates free-tier ToS and risks a mid-project ban. |

---

## How to use this document

- **Parts 1–2** — why the system is shaped this way, and everything we take from elsewhere. Read once, fully.
- **Parts 3–6** — the build spec. Each pipeline has inputs, outputs, implementation notes, and its source attribution. This is the reference you return to.
- **Parts 7–11** — repo layout, phase-by-phase sequence, team, risks, credits.

Companion documents: `satquery-ai-study-primer.md` (background knowledge), `satquery-landscape-survey.md` (competitive analysis), `satquery-plan-v2.md` (superseded — kept for history).

---

# PART 1 — Thesis

## 1.1 What the problem statement asks for

A web application where a non-expert uploads satellite imagery, asks a question in plain English, and receives a trustworthy answer with visual proof — with the system itself deciding which specialist models and tools the question requires.

Six mandatory gates:

| Gate | Requirement |
|---|---|
| **G1** | ≥1 visual/VL component fine-tuned on BigEarthNet.txt or other open data |
| **G2** | Single-image VQA (mandatory) |
| **G3** | Captioning **or** text-guided grounding (pick one) |
| **G4** | Change description **or** change VQA from a bi-temporal pair; change map optional |
| **G5** | Complementary information extraction from a co-registered optical–SAR pair |
| **G6** | Agentic orchestration with an auditable execution trace |

Plus: GeoTIFF/TIFF support, compatibility checking, visual evidence, confidence information, execution summaries, downloadable reports, interactive GUI.

Scoring: prescribed public benchmark splits **plus** a hidden ISRO/SAC set of pre-georeferenced, co-registered Cartosat-2S optical + RISAT SAR pairs. Scores normalised before combining.

### [v3.1] Four things the PS wording pins down that we had left loose

| PS wording | What it actually constrains |
|---|---|
| The ISRO set carries "reference answers, labels, **bounding boxes, or masks**, as applicable" | Grounding **and** masks are graded on the hidden set. G3 nominally lets us elect captioning instead of grounding — but electing it would forfeit the bbox portion. Grounding is mandatory in practice (C17); masks are scoreable outputs, not decoration (C18) |
| The controller "must configure **only permitted task parameters**"; the graded trace contains "the selected task, model/tool names, and key parameters" | Exactly four graded fields. Parameters must be *enforced*, not merely logged (C19, C21) |
| "BigEarthNet.txt will serve as the **primary dataset for adapting**… VRSBench and RSVQA will be used to **evaluate**… CDVQA will be used to **evaluate**" | The datasets have assigned roles. BEN.txt dominates training; the others are eval-first (C20) |
| Deliverables: "Codes **and models** including test and demonstration" | We ship weights. Anything those weights trained on becomes part of a government deliverable — which is why VRSBench's DOTA-derived academic-only imagery leaves the training mix entirely (C20) |
| "Internal reasoning text is neither required nor evaluated" | Confirms Instruct over Thinking (§5.1). Spend zero tokens on chain-of-thought in the trace |

### [v3.2] Input-scope coverage matrix — what actually trains each capability

The PS's Defined Input Scope crossed against our training data. This table exists so nobody discovers a hole in Week 8.

| PS-required capability | Training data | Status |
|---|---|---|
| Single **optical** image · VQA | BEN.txt VQA (primary), RSVQA-LR + **HR** | ✅ Covered |
| Single **optical** image · captioning | BEN.txt captions | ✅ Covered |
| Single optical image · **region** grounding | BEN.txt referring LULC + point detection | ✅ Covered — matches the PS query *"Highlight the water body referred to in the query"* |
| Single optical image · **object** grounding — *buildings & aircraft* | **[v3.4] COVERED (C36, C37)** — RarePlanes aircraft @30 cm (CC BY-SA 4.0); SpaceNet 6 + OpenEarthMap-SAR buildings | ✅ On shippable licences |
| Single **SAR** image · **object** grounding — *ships* | **[v3.8] COVERED (C53)** — LS-SSDD-v1.0, Apache-2.0, 6,015 expert ship boxes on Sentinel-1 | ✅ SAR objects — directly relevant to RISAT |
| Single optical image · **object** grounding — *vehicles, tanks, bridges, harbours* | **OUT OF SCOPE.** `object_box_fallback` (§4.6.9) supplies deterministic partial credit | ⚖️ Only exist in Google Earth-derived sets |
| Single **SAR** image · VQA / captioning / grounding | **SARLANG-1M + BEN modality dropout (C26)** | ✅ Covered as of v3.2 — was a gap |
| **Bi-temporal** pair · change VQA / description | **[v3.3] CDVQA, SECOND-CC, QAG-360K** — LEVIR removed from this adapter (C33), since it trains shipped weights | ✅ Covered. CDVQA is the PS-nominated change benchmark |
| Bi-temporal pair · change **map** | **LEVIR-CD / MCI + SECOND** — LEVIR retained here (C33); output is a mask, not weights | ✅ Covered, with disclosure |
| **Cross-modal** optical–SAR pair · joint extraction | BEN.txt S1+S2 at 10 m **+ SpaceNet 6 MSAW at 0.5 m quad-pol (C31) + [v3.4] OpenEarthMap-SAR at 0.15–0.5 m single-pol across 35 regions (C37)** | ✅ **Two independent sub-metre sources** — see below |
| GeoTIFF / TIFF ingest | reBEN, Bhoonidhi, **+ synthetic fixtures (C30)** | ✅ Covered as of v3.2 |

> **[v3.3] C28 reversed — the gap is narrower than we claimed (C31).** v3.2 stated that no high-resolution co-registered optical–SAR dataset existed and that the gap "is not fixable by finding another dataset." **That was wrong.** SpaceNet 6 MSAW provides ~0.5 m Capella SAR co-registered with ~0.5 m Maxar optical over Rotterdam, ~48,000 building footprint labels across 120 km², under CC BY-SA 4.0.
>
> Three things it gives us that nothing else in the inventory can:
> 1. **X-band.** RISAT-2B/2BR1 are X-band; every other SAR source we hold is C-band Sentinel-1. C2 built an entire single-pol/X-band defence that we previously had **no way to test**. Now we can.
> 2. **Quad-pol at sub-metre.** We can synthesise genuine single-pol from real quad-pol data at metre scale, validating the C26 dropout strategy instead of assuming it.
> 3. **Building footprints as ground truth.** Exactly the target of D1's built-up branch — so cross-modal agreement IoU can be *scored*, not just reported.
>
> **What it is not:** it carries no image–text annotations, so it is not drop-in VLM training data, and it covers one city. The BEN.txt pipeline shows the route — template captions generated from reference maps, then LLM linguistic augmentation — and we apply the same approach to the footprints (§5.3a).
>
> **The G5 conclusion is unchanged, and still correct.** D1 carries G5 because it is deterministic and resolution-agnostic; `optsar_fusion` remains the upside. SpaceNet 6 makes the upside materially more likely and gives D1 a real validation set. It does not change which one we bet the mandatory gate on.

> **[v3] Open question with an owner:** we do not yet know *how* the hidden set is executed — container we submit, API we host, or organisers running our code on their hardware. Integration lead obtains the submission/packaging spec in Phase 0 (item 3). Until answered, the headless eval mode (§4.11) is built to satisfy the most restrictive plausible answer: single container, single command, no network, CPU-capable deterministic path.

## 1.2 The shortcomings we are answering

Every one of these is documented, not asserted. This table is both our design rationale and our pitch spine.

| # | Shortcoming | Evidence | Our answer |
|---|---|---|---|
| S1 | Generic VLMs fail on multi-sensor RS | GPT-5.2 (2T params): 60.39% binary VQA, 34.93% MCQ on the BEN.txt benchmark | LoRA adapters (§5) |
| S2 | RS-specific VLMs are narrow, lose to general models | GeoChat 50.82%, LHRS-Bot 48.23%, SkyEyeGPT 48.87% vs Qwen3-VL 61.96% | Modern base + task adapters, **not** GeoChat |
| S3 | Multi-sensor models don't exploit multi-sensor input | EarthDial-S2 collapsed to 8.43% MCQ / 0.49 mIoU with 12 bands vs 32.94 / 7.13 on RGB | Never widen inputs without training for them (§5.2); ablate vs RGB-only every run |
| S4 | Monolithic models cap out on hard sub-tasks | "Smallest change" 32–37% across every Qwen size; grounding without priors 25.16 mIoU | Route to deterministic tools (§4.6.2, §4.6.4) |
| S5 | Cross-sensor fusion has never been made agentic | EarthMind / Earth-OneVision = one model with a fusion block; Earth-Agent = RGB+spectral only | Band-aware routing + decision-level fusion (§4.5, §4.7) |
| S6 | No real geospatial handling, no confidence anywhere | Entire RS-VLM literature runs on PNG/JPEG chips; zero calibration reported | Full GeoTIFF pipeline (§4.1–4.4), calibration (§4.7.3) |

## 1.3 The four differentiators

**D1 — Decision-level cross-modal fusion with physically-explained disagreement.**
Independent optical and SAR decisions, reconciled with a named physical cause when they conflict. **[v3] Fully deterministic, requires no training, and is now built in Phase 1–2 as the guaranteed floor.** The learned `optsar_fusion` adapter is a Phase 3 upgrade on top of it, never a dependency.

**D2 — Deterministic priors feeding learned models.** *(status: hypothesis until Week 1 ablation)*
Manufacture a centroid from a spectral mask, hand it to the grounding model.
**[v3] Honest evidence status:** the BEN.txt numbers we previously cited (25.16 vs 38.95 mIoU) compare *referring LULC detection* against *referring point detection* — two different tasks with different outputs. They suggest the mechanism but do **not** prove that injecting a point prior improves grounding. We therefore validate D2 ourselves, for free, in Week 1: prompt the **base** model on ~200 grounding samples with and without an injected centroid. No training required.
- Lift ≥ 5 mIoU → D2 is confirmed; build `centroid_prior` into the grounding path and quote *our own* ablation number.
- Lift < 5 mIoU → D2 is reframed (it still stands): the deterministic mask gives a **geo-referenced anchor for evidence export and search-space narrowing**, which is defensible regardless of mIoU lift, and we stop claiming a grounding-accuracy improvement we can't show.

**D3 — Band-availability-driven routing.**
Cartosat-2S MX has no SWIR, so NDBI is impossible and built-up detection must fall to SAR. The agent inspects bands and swaps toolchains. **[v3] Extended to degenerate inputs:** pan-only or RGB-only optical (no NIR) kills NDVI *and* NDWI — the router now has a defined texture/morphology branch and an explicit explained-refusal path for this case (§4.1.4, §4.5.3). D3 is a lookup table on `BandInventory`; it requires no training and is built in Phase 1–2.

**D4 — Quantitative sub-tasks bypass the VLM.**
"Smallest change", "change ratio", "largest change" are arithmetic on a change map. Compute them exactly.

---

# PART 2 — Complete source inventory

Everything we take from elsewhere, with licence status.
**[v3] Licence rule: every ⚠ below is resolved in Phase 0 — item 2 of the checklist, owned by the integration lead, budgeted one day. Nothing trains on data whose licence is still ⚠.**

## 2.1 Model weights

| Source | Artifact | Use | Licence |
|---|---|---|---|
| Alibaba / HF | `Qwen3-VL-4B-Instruct` (or `Qwen3.5-2B`) | Base VLM | Apache 2.0 |
| TU Berlin / BigEarthNet | BigEarthNet-pretrained ViT (S1 + S2) | `lulc_classifier` + §5.6 stretch only *(no longer load-bearing — see C1)* | ⚠ resolve Phase 0 |
| **[v3] torchgeo** | Pretrained S1/S2 ResNet/ViT weights | **Named substitute** if the BEN ViT is unavailable/restricted | MIT (weights per-model — verify) |
| **[v3] DOFA / SatMAE** | Wavelength-aware / MAE RS encoders | Second-line substitutes | verify in Phase 0 |
| Meta | SAM / SAM2 | Optional mask refinement | Apache 2.0 |
| TinyCD / ChangeFormer authors | Pretrained CD backbone | `change_map` starting point | ⚠ resolve Phase 0 |

## 2.2 Datasets

**[v3] Staging rule — manifest first.** We subsample per adapter, so we never download full corpora. Sequence: (1) pull metadata/annotation indices only; (2) select the training patch-ID manifest (balanced per §5.4); (3) download **only** manifest patches + official benchmark splits; (4) stage as private Kaggle Datasets every teammate attaches. Full-corpus downloads (~600 GB+) are banned — they don't fit free-tier disk and we don't need them.

**[v3.2] Storage estimate corrected (C29).** The "60–80k samples" figure is **per adapter**. Four adapters is 240–320k samples, so realistic staged size is **~140–190 GB**, not the ~40 GB stated in C13. Kaggle's per-dataset limit means this must be **split into three or four datasets**, staged in this order so training can start before staging finishes:

| Order | Contents | Approx. | Unblocks |
|---|---|---|---|
| 1 | BEN.txt manifest subset (VQA + captions + referring) + all benchmark splits | ~50–65 GB | `rs_vqa`, `rs_ground_caption` — the Phase 1 critical path |
| 2 | **SpaceNet 7 / MUDS (C59, primary) + HRSCD (C60) + self-generated Sentinel pairs (C46) + OSCD (C49)** for `change_vqa`/`change_map`; SECOND / LEVIR-CD / LEVIR-MCI / QAG-360K for `change_map` only (C33/C48); CDVQA eval split | ~25–35 GB | `change_vqa`, `change_map` |
| 3 | SARLANG-1M subset + BEN S1+S2 pair subset | ~25–35 GB | `optsar_fusion`, plus C26 SAR-only samples |
| 4 | RSVQA-HR + **SpaceNet 6 MSAW (C31)** + **OpenEarthMap-SAR (C37)** + **RarePlanes real split (C36)** + holdout v0 | ~35–50 GB | C27 domain-gap work, D1 validation, single/quad-pol tuning, object grounding |

| Source | Content | Use | Licence |
|---|---|---|---|
| **BigEarthNet.txt** (arXiv 2603.29630) | 464,044 co-registered S1+S2, 9.6M annotations, 15 tasks | Primary training + benchmark split (manifest subset) | **✅ CDLA-Permissive 1.0** (inherits BEN) — verify the .txt annotation layer carries the same |
| **BigEarthNet v2.0 / reBEN / BEN-MM** | 549,488 S1/S2 pairs, CLC 2018 pixel maps | Underlying imagery (manifest subset) | **✅ CDLA-Permissive 1.0 — confirmed.** Explicitly imposes **no restrictions on results of computational use**, i.e. our weights are unencumbered |
| **CDVQA** (arXiv 2112.06343) | 2,968 pairs @512px, 122k QA, test1/test2 | **[v3.6] EVALUATION ONLY (C46)** — PS-nominated, so evaluating is compliance | ❌ **Derived from SECOND, which has no licence.** Not used for training |
| **VRSBench** (arXiv 2406.12384) | 29,614 images, captions, refers, 123k QA | **[v3.1] EVALUATION ONLY** — grounding acc@0.5/0.7 reporting. Removed from all training mixes (C20) | Text CC-BY-4.0; **images DOTA-derived, academic only — never in demo/repo, and now never in shipped weights** |
| **RSVQA-LR** | Sentinel-2, 10 m, Netherlands | VQA train (minority) + eval | **✅ [v3.8] CC BY 4.0 (Zenodo 6344333)** |
| **RSVQA-HR** | **USGS HRO 15 cm aerial, US northeast — VERIFIED (C54)** | **[v3.2] PROMOTED to first-class train + eval (C27)** — the closest GSD match to Cartosat-2S in the whole inventory; every other optical source is 10 m | **✅ [v3.8] USGS imagery public domain; annotations CC BY 4.0 (Zenodo 6344366)** |
| ~~Object-grounding candidates~~ **DIOR · FAIR1M · NWPU-Captions · RSICD** | Object-level detection / captioning | **[v3.3] ALL REJECTED (C32)** — G3 scoped to region grounding | ❌ **All Google Earth-derived**, no shippable licence. DIOR and RSICD explicitly Google Earth; FAIR1M is Gaofen + Google Earth; NWPU-Captions derives from NWPU-RESISC45 |
| **SpaceNet 7 / MUDS (C59)** | 101 AOIs, 6 continents, 24 monthly mosaics each, ~40,000 km², **11M manually-annotated building footprints with tracking IDs**, 4 m Planet | **PRIMARY expert-labelled change source** for `change_vqa` + `change_map`; tracking IDs give real validation data for `change_stats` (D4) | **✅ CC BY-SA 4.0**, AWS Open Data Program. *Verified from the MUDS and challenge papers* |
| **HRSCD (C60)** | 0.5 m aerial, France, multi-class semantic change (Urban Atlas classes), 2006 + 2012 | **Only sub-metre SEMANTIC change source.** Train only — never redistribute imagery | **⚠✅ IGN *licence ouverte* for images, BUT the 2006 images are non-redistributable** and must come from IGN directly. Record the split status in CREDITS |
| ~~DynamicEarthNet~~ | 75 AOIs, daily Planet 3 m multispectral, monthly 7-class LULC labels | **[C61] NOT STAGED** — pending licence check | ⚠ **Commercial Planet Fusion data at its core; licence unverified.** Send to the §2.2b verification job before any use |
| **[v3.6] OSCD + S1 extension (C49)** | 24 Sentinel-2 bi-temporal pairs, 13 bands, worldwide; Ebel et al. Sentinel-1 extension adds SAR | **`change_vqa` training** (clean); **the only optical–SAR *change* data we have** | **✅ ONERA, openly available as a benchmark**; IEEE DataPort open-access; Copernicus imagery |
| **[v3.6] Self-generated Sentinel change pairs (C46)** | Bi-temporal S1/S2 over Indian AOIs, unlimited volume, index-differenced change labels, generated captions | **Supplementary `change_vqa` source (C59 supersedes as primary)** — retained for Indian domain match and unlimited volume | **✅ Copernicus — fully open, commercial use permitted.** Attribution notice only |
| ~~DFC2023~~ | SuperView-1 / Gaofen-2 / Gaofen-3, 0.5–1 m optical+SAR | **[v3.6] EXCLUDED (C47)** — filtered out of SARLANG-1M | ❌ IEEE GRSS contest data: approved-participant access, contest T&Cs, commercial Chinese imagery |
| ~~SARDet-100K~~ | ~117k images, 246k instances, 6 classes | **[v3.6] EXCLUDED (C47)** — filtered out of SARLANG-1M | ❌ Standardisation of **ten** existing SAR datasets, released "for research purposes". Cascade risk to the tenth degree |
| **[v3.4] RarePlanes (C36)** | 253 Maxar WV-3 scenes, 112 locations, 2,142 km², **14,700 hand-annotated real aircraft @30 cm** + ~630k synthetic annotations across 50k synthetic images | **Object-level grounding supervision** for `rs_ground_caption` — aircraft | **✅ CC BY-SA 4.0**, AWS Open Data Program |
| **[v3.4] OpenEarthMap-SAR (C37)** | 5,033 images, 35 regions (JP/FR/US), **0.15–0.5 m**, VV **or HH single-pol**, expert-aligned optical–SAR pairs, 8 classes incl. building (178.5k segments), water, road | **Second sub-metre cross-modal source**; single-pol validation; sub-metre region grounding | **✅ SAR: Umbra Lab CC BY 4.0. Optical: NAIP public domain, IGN France CC BY 2.0, GSI Japan.** On Zenodo |
| ~~xView3-SAR~~ | 991 Sentinel-1 images, 243,018 maritime objects | **[v3.4] REJECTED (C38)** | ❌ T&C page 404s; registration wall; composites "© Cambrio LLC; rights reserved". Sibling xView1 is CC BY-NC-SA |
| **LS-SSDD-v1.0 (C53)** | 15 Sentinel-1 scenes → 9,000 sub-images, **6,015 expert ship boxes**, VV+VH | **Ship object grounding for `rs_ground_caption`** — SAR objects, directly relevant to the RISAT half of the hidden set | **✅ [v3.8] Apache-2.0 — CLEARED.** Sentinel-1 imagery Copernicus-open |
| **[v3.3] SpaceNet 6 MSAW (C31)** | ~0.5 m Capella **X-band quad-pol** SAR + ~0.5 m Maxar optical, co-registered; ~48,000 building footprints, 120 km² Rotterdam | **Cross-modal `optsar_fusion` training** (via generated captions), **D1 validation with ground-truth masks**, **X-band and single-pol stress testing** | **✅ CC BY-SA 4.0** — free on AWS S3 |
| **SARLANG-1M** (arXiv 2504.03254) | 118,331 SAR images, 1.08M pairs, 0.1–25m, 59 cities | Sub-metre SAR training; single-channel pol-dropout; SAR-only supervision for `rs_vqa` (C26). SAR image–text, **not** cross-modal pairs | **✅ [v3.6] RESOLVED BY FILTERING (C47).** Use **only** the SpaceNet 6 and OpenEarthMap-SAR portions (both clean, both already staged independently). **Exclude DFC2023 and SARDet-100K portions.** Note SARDet-100K feeds only SARLANG-1M-VQA, so the Cap benchmark needs only the DFC2023 filter |
| ~~LEVIR-MCI~~ | Bi-temporal + masks + captions | **[v3.8] DROPPED ENTIRELY (C56)** | ❌ Academic-only. C33's mask-only exemption failed: `change_map` ships a fine-tuned weight |
| ~~LEVIR-CD~~ | Binary change detection, 0.5 m, 637 pairs | **[v3.8] DROPPED ENTIRELY (C56)** | ❌ Academic-only, Google Earth ToS |
| ~~SECOND~~ | 4,662 aerial pairs, 0.5–3 m, Hangzhou/Chengdu/Shanghai, 6 classes | **[v3.8] DROPPED ENTIRELY (C56)** | ❌ **NO LICENCE STATEMENT OF ANY KIND.** Google Drive links + a contact email. Absence of a licence grants fewer rights than a restrictive one. **Root of the CDVQA / SECOND-CC / QAG-360K cascade** |
| ~~QAG-360K~~ | 6,810 pairs, 360k Q/A/mask triplets, 24 regions | **[v3.8] DROPPED ENTIRELY (C56)** | ❌ **Sourced from Hi-UCD + SECOND + LEVIR-CD.** Was reintroducing LEVIR imagery into `change_vqa` through the back door |
| **Bhoonidhi (NRSC)** | Cartosat-2S + RISAT samples | India holdout **v1** (see below) | ISRO terms — apply Day 1 |
| **[v3] Open India proxy** | Sentinel-1/2 over Indian AOIs resampled to Cartosat/RISAT GSD + open NRSC/Bhuvan products | **India holdout v0** — exists Week 2 with zero approvals | Open |
| **SRTM / Copernicus DEM** | Elevation | Terrain correction | Open |

> **[v3] Bhoonidhi is no longer on the critical path.** Apply Day 1 (geospatial engineer owns the application; integration lead owns the chase). But the domain-gap mitigation — "India holdout from Week 2" — now runs on **holdout v0**, the proxy set above, which requires nobody's permission. When Bhoonidhi access lands, real Cartosat/RISAT scenes become **holdout v1** and v0 is retired to a secondary robustness set. If access never lands, v0 is what we ship, and it is still more India-testing than any competing team will have.

### 2.2a Provenance traceability rule **[v3.6, C45]**

**Before any dataset enters a training manifest, trace it to the imagery programme it originates from — not merely to the paper that published it.**

A derived dataset inherits the **most restrictive licence in its ancestry**, and papers almost never state this. Four separate licence sweeps on this project each found the same failure mode:

| Dataset | Traces to | Depth |
|---|---|---|
| CDVQA | SECOND *(no licence)* | 2 |
| SECOND-CC | SECOND *(no licence)* | 2 |
| QAG-360K | Hi-UCD + SECOND + LEVIR-CD | 2 |
| VRSBench | DOTA → Google Earth | 3 |
| NWPU-Captions | NWPU-RESISC45 → Google Earth | 3 |
| SARLANG-1M | SpaceNet6 + OEM-SAR + DFC2023 + SARDet-100K | 2 |
| SARDet-100K | **ten** SAR datasets | 3+ |

**The heuristic that predicts the answer:** clean datasets are built *directly* on an open imagery programme — **Copernicus, NAIP, USGS** — or on commercial imagery deliberately released under CC (SpaceNet/Maxar, Umbra, RarePlanes). Anything built on *another dataset* needs the chain walked.

**Procedure:**
1. Find the imagery source in the dataset paper's data section — not the abstract, not the repo README.
2. If the source is another dataset, repeat until you reach a satellite/aerial programme.
3. Record the full chain in `../../CREDITS.md`.
4. **No licence statement = no permission.** Treat silence as more restrictive than an explicit academic-only clause, not less.

### 2.2b Residual checks — **ALL RESOLVED [v3.8]**

External verification completed 2026-08-26. Full records in `../../CREDITS.md`.

| Item | Result | Verdict |
|---|---|---|
| **BigEarthNet.txt** | CDLA-Permissive-1.0, confirmed on `txt.bigearth.net` | ✅ CLEAR — no share-alike, no restriction on downstream weights |
| **LS-SSDD-v1.0** | **Apache-2.0** (root LICENSE, `TianwenZhang0825/LS-SSDD-v1.0-OPEN`); imagery Sentinel-1 | ✅ CLEAR — **ships enter the G3 vocabulary (C53)** |
| **RSVQA-LR** | CC BY 4.0 (Zenodo 6344333); imagery Sentinel-2 | ✅ CLEAR with attribution |
| **RSVQA-HR** | CC BY 4.0 (Zenodo 6344366); **imagery USGS HRO 15 cm, public domain** | ✅ CLEAR with attribution — **C27 validated (C54)** |
| **BigEarthNet ViT weights** | CDLA-Permissive-1.0, BIFOLD/TU Berlin on Hugging Face | ✅ CLEAR for inclusion |
| **TorchGeo foundation weights** | MIT (SSL4EO-S12, SeCo) | ✅ CLEAR — **now our CD backbone (C55)** |
| **TinyCD weights** | Non-commercial / research only | ❌ **RESTRICTED — excluded (C55)** |
| **ChangeFormer weights** | Non-commercial / research only | ❌ **RESTRICTED — excluded (C55)** |
| **AROSICS** | Apache-2.0 on ≥1.0.0; **pre-1.0 was GPL-3.0** | ✅ CLEAR — **pin ≥1.0.0 (C58)** |
| **SpaceNet 7 / MUDS** | CC BY-SA 4.0, AWS Open Data *(verified from MUDS + challenge papers; not yet externally re-verified)* | ✅ CLEAR — attribution + share-alike on redistributed derivatives |
| **HRSCD** | IGN *licence ouverte*; **2006 images non-redistributable** | ⚠✅ CLEAR to train on; **never redistribute the imagery** |
| **DynamicEarthNet** | **UNVERIFIED** — commercial Planet Fusion at core | ⚠ **OPEN — do not stage until checked** |

> **Note on RSVQA:** the dataset is CC BY 4.0 but the authors' *evaluation-script repository* (`syvlo/RSVQA`) is **GPL-3.0**. We use the data, not their codebase. Do not vendor those scripts.

## 2.3 Libraries## 2.3 Libraries## 2.3 Libraries

| Library | Purpose | Licence |
|---|---|---|
| GDAL | Raster I/O core | MIT/X |
| rasterio | Pythonic raster interface | BSD-3 |
| rioxarray / xarray | Labelled N-D geo arrays | Apache 2.0 |
| pyproj | CRS transforms | MIT |
| torchgeo | Geo datasets, transforms, pretrained models | MIT |
| scikit-image | Otsu, phase correlation, connected components, **[v3] GLCM texture** | BSD-3 |
| **ESA SNAP** + pyroSAR | SAR calibration, speckle, terrain correction | **GPL-3.0 — call as external process, do not link** |
| AROSICS **(pin >= 1.0.0)** | Sub-pixel co-registration | **✅ [v3.8] Apache-2.0 on ≥1.0.0 — verified.** Import normally, no GPL isolation. **Pre-1.0 releases were GPL-3.0 — the version pin is the control (C58)** |
| transformers | Model loading | Apache 2.0 |
| peft | LoRA / QLoRA | Apache 2.0 |
| bitsandbytes | 4-bit quantisation | MIT |
| vLLM | Multi-LoRA serving | Apache 2.0 |
| Outlines | Constrained JSON decoding | Apache 2.0 |
| LangGraph | Agent state graph | MIT |
| FastAPI / Redis / Celery | Backend (interactive app only — **not** in headless eval mode) | MIT / BSD |
| React / MapLibre GL | Frontend | MIT / BSD-3 |
| scikit-learn | **[v3] Isotonic/logistic calibration** | BSD-3 |
| Weights & Biases | Experiment tracking | Free tier |

> **GPL note:** SNAP is GPL-3.0. Invoking it as a separate process (subprocess / Docker service) keeps our code outside the GPL boundary. Do not import it as a library into our codebase.

## 2.4 Methods we read and reimplement

We do **not** clone these. We read the method and write our own against our schema.

| Source | What we take | Where used |
|---|---|---|
| **RS-InternVL** (BEN.txt paper §4.2) | LoRA hyperparameters, LR schedule, 60m-band drop | §5.2. **[v3] The dual-encoder architecture itself is stretch-only (§5.6)** — it cannot be served by vLLM multi-LoRA and no longer sits on the main path |
| **GeoPixel** (arXiv 2501.13925) | Adaptive local/global partitioning, 4K any aspect ratio | §4.4 tiling |
| **EarthMind** (arXiv 2506.01667) | HCA fusion design + the negative result that token concatenation fails | §4.7 fusion |
| **Earth-Agent** (ICLR 2026) | Dual-level evaluation: trajectory + outcome | §4.5.5 trace schema |
| **BEN.txt grounding tables** | Point-detection vs LULC-detection numbers — **[v3] treated as motivation, not proof; we run our own prior-vs-no-prior ablation Week 1** | §4.6.2 D2 |
| **CDVQA Qwen study** (arXiv 2604.18429) | Target numbers + "smallest change" 32–37% ceiling | §4.6.4 D4, §6 targets |
| **SMARTIES / DOFA** | Spectrum-aware band projection for unseen sensors | §4.1.5 fallback (cut-list item) |
| **Change-Agent** (TGRS 2024) | Multi-task change interpretation structure | §4.6.3 |

## 2.5 Classical algorithms we implement ourselves

NDVI · NDWI · MNDWI · NDBI · Otsu thresholding **[v3] + bimodality gate** · Refined Lee speckle filter · phase cross-correlation · connected-component analysis · percentile stretch · temperature scaling · **[v3] GLCM texture features · local coefficient of variation · isotonic calibration fit**.

## 2.6 Honest accounting

**Taken:** 1 base model, pretrained checkpoints, 11 datasets, ~20 libraries, 8 published methods.
**Built:** geospatial ingest layer, agentic controller, validator, trace system, all four differentiators, four LoRA adapters, frontend, backend, headless eval mode, eval harness, calibration layer.

**Ratio: ~70% original code, ~20% reimplemented methods, ~10% cloned and fine-tuned.**

---

# PART 3 — System architecture

```
┌──────────────────── FRONTEND (React) ─────────────────────┐
│  upload · map viewer (MapLibre) · geo overlays · chat     │
│  evidence panel · trace viewer · PDF export               │
└──────────────────────────┬────────────────────────────────┘
                           │ REST + SSE
┌──────────────────────────▼────────────────────────────────┐
│  API GATEWAY (FastAPI) · Redis queue · async workers      │
│  [v3] + HEADLESS EVAL CLI — same core, no web stack       │
└──────────────────────────┬────────────────────────────────┘
                           │
   ┌───────────────────────▼─────────────────────────────┐
   │  P1  INGEST & VALIDATION      (async job, cached)   │
   │  P2  SAR NORMALISATION        (async job, cached)   │
   │  P3  CO-REGISTRATION          (async job, cached)   │
   │  P4  TILING                   (async job, cached)   │
   │      → ImageBundle + BandInventory                  │
   └───────────────────────┬─────────────────────────────┘
                           │
   ┌───────────────────────▼─────────────────────────────┐
   │  P5  AGENTIC CONTROLLER (LangGraph)                 │
   │   rules-first route → validate → plan DAG →         │
   │   execute → fuse → confidence → emit trace          │
   └───────────────────────┬─────────────────────────────┘
                           │
   ┌───────────────────────▼─────────────────────────────┐
   │  P6  TOOL REGISTRY                                  │
   │   LEARNED: rs_vqa · rs_ground_caption ·             │
   │            change_vqa · optsar_fusion               │
   │   DETERMINISTIC: spectral_index · sar_backscatter · │
   │            change_map · change_stats ·              │
   │            lulc_classifier · centroid_prior ·       │
   │            coreg_check · tile_scorer ·              │
   │            texture_seg · object_box_fallback [v3.3]  │
   └───────────────────────┬─────────────────────────────┘
                           │
   ┌───────────────────────▼─────────────────────────────┐
   │  P7  FUSION & CONFIDENCE                            │
   │  P8  EVIDENCE & REPORTING                           │
   │  P9  MODEL SERVING (vLLM multi-LoRA)                │
   └─────────────────────────────────────────────────────┘
```

**[v3] Two SLAs, not one.** P1–P4 run as an asynchronous *preparation job* per uploaded scene (target ≤ 5 min/scene, progress bar in UI, result cached as an `ImageBundle`). The < 20 s latency target applies **only** to query time against a prepared bundle. Demo scenes are pre-warmed. We state both numbers; a measured "4 min prep + 12 s query" beats an unmet 20 s claim.

**Data contract:** every pipeline stage consumes and emits a versioned `ImageBundle`. Nothing is passed as loose arrays.

```python
@dataclass
class ImageBundle:
    images: list[ImageRef]          # normalised rasters on a common grid
    band_inventory: BandInventory   # drives routing — see §4.1.4
    pair_type: Literal["single","crossmodal","bitemporal"]
    coreg: CoregReport | None
    tiles: TileIndex | None
    provenance: list[ProvenanceStep] # every transform, for the trace
```

---

# PART 4 — Pipeline specifications

## 4.1 — P1: Ingest & Validation

**Owner:** Geospatial engineer · **Effort:** ~1.5 weeks · **Gate:** deliverable "input upload and compatibility checking"

### 4.1.1 Read
`rasterio.open()`. Handle GeoTIFF, TIFF, COG, and (benchmark-only) PNG/JPEG. Windowed reads via `rasterio.windows.Window` — **never `src.read()` a full Cartosat scene.**

Extract: `crs`, `transform`, `bounds`, `res`, `dtype`, `nodata`, `count`, `descriptions`, `tags()`.

### 4.1.2 Modality detection
Heuristic cascade:
1. Metadata tags naming the sensor → authoritative
2. Band count: 1–2 → likely SAR **or pan-only optical** [v3 — disambiguate via dynamic-range signature, GSD, and metadata before assuming SAR]; 3–4 → optical RGB/MX; 10–13 → multispectral
3. Dynamic range signature: SAR has heavy-tailed distribution with high local variance (speckle); optical does not
4. If ambiguous → **ask the user**, record the answer in the trace

### 4.1.3 Radiometry
Detect bit depth (8/12/16). Apply robust percentile stretch (2nd–98th) for model input. **Record the stretch bounds in provenance** — reproducibility depends on it.

Distinguish and record separately:
- `native_gsd_m` — what the sensor actually resolved
- `pixel_size_m` — the storage grid spacing

These differ after resampling and judges notice teams that conflate them.

### 4.1.4 BandInventory — the routing driver

```python
@dataclass
class BandInventory:
    bands: dict[str, int]        # {"blue":1,"green":2,"red":3,"nir":4}
    has_swir: bool
    has_nir: bool
    is_pan_only: bool            # [v3] single broadband optical channel
    polarisations: list[str]     # ["VV","VH"] | ["HH"] | ...
    sar_band: str | None         # [v3] "C" | "X" | "L" | None/unknown
    sensor_hint: str | None
    computable_indices: list[str]  # derived
```

`computable_indices` drives P5's toolchain selection:
- Cartosat-2S MX (B,G,R,NIR) → `["NDVI","NDWI"]` — **not** `["MNDWI","NDBI"]`. This drives D3.
- **[v3] RGB-only (no NIR)** → `[]` for water/vegetation indices; optical falls back to `texture_seg` (§4.6.8) at reduced confidence, or SAR takes primary.
- **[v3] Pan-only** → `[]`; optical contributes texture/morphology evidence only; if the question is inherently spectral ("is the vegetation healthy"), the validator issues an explained refusal (§4.5.3).

### 4.1.5 Band harmonisation
Project any input to a canonical band set via lookup table (build this first — 3 days). The SMARTIES/DOFA-style wavelength-conditioned projection is an explicit **cut-list item** — only if Phase 3 is green.

### 4.1.6 Validation output
Emit `CompatibilityReport`: format ok, bands present, CRS valid, nodata fraction, bit depth, GSD, and a list of `warnings[]`. Surface warnings in the UI, not just the log.

---

## 4.2 — P2: SAR Normalisation

**Owner:** Geospatial engineer · **Effort:** ~1 week · **Source:** ESA SNAP (external process)

> **Critical:** this chain must be byte-identical at training and inference. If BigEarthNet's S1 was processed one way and RISAT another, you have manufactured a domain gap by hand. Freeze it in Phase 0 and version the config.

### Chain (in order)
1. **Apply orbit file** — SNAP (skip gracefully if orbit metadata absent — record in provenance)
2. **Radiometric calibration → σ⁰** — SNAP
3. **Multilooking** — SNAP. Speckle σ drops ~√N with N looks; costs resolution
4. **Speckle filter — Refined Lee** — SNAP. Smooths uniform areas, preserves edges
5. **Terrain correction (Range-Doppler)** — SNAP + SRTM/Copernicus DEM. **Prerequisite for co-registration with optical.** [v3] If inputs are already terrain-corrected/pre-georeferenced (the ISRO eval pairs are), detect this from metadata and skip — re-correcting corrected data degrades it.
6. **dB conversion** — ours: `10 * log10(sigma0)`
7. **Robust percentile stretch** — ours
8. **Model-input stack** — ours, **[v3] now polarisation-aware:**

```
dual-pol (VV,VH or HH,HV):   [pol1_dB, pol2_dB, pol1/pol2 ratio_dB]
single-pol (HH or VV only):  [σ⁰_dB, refined-Lee-filtered σ⁰_dB, GLCM-entropy texture]
```

**[v3] The single-pol stack is a first-class citizen, not an error path.** RISAT-1/EOS-04 FRS modes ship single-pol; RISAT-2B/2BR1 are X-band spotlight. During `optsar_fusion` and SAR-facing training, apply **polarisation dropout**: convert ~25% of dual-pol training samples to the single-pol stack so the model has genuinely seen 1-channel SAR before the hidden set shows it one. SARLANG-1M's single-channel imagery feeds the same path.

**Never tile a single SAR band three times to fake RGB** — the single-pol stack above carries filtered + texture channels instead, which preserves information the raw repeat does not.

### Sanity ranges — **[v3] parameterised and warning-only**

Maintained as a table keyed by `(sar_band, polarisation)` in `preprocessing.yaml`. C-band VV values below are the calibrated reference; X-band columns start as provisional copies shifted per literature and get tuned on holdout v0/v1.

| Surface | C-band VV σ⁰ dB |
|---|---|
| Calm water | −25 to −20 |
| Smooth bare soil | −15 to −12 |
| Rough soil / crops | −12 to −8 |
| Forest | −8 to −6 |
| Urban | −5 to +5 |

**[v3] These asserts WARN, they never block.** A hard assert tuned on C-band would refuse a legitimate X-band scene from the graded hidden set — a self-inflicted zero. Out-of-range scenes get a trace warning + lowered SAR-side confidence, and processing continues.

**Implementation:** SNAP runs as a separate Docker service invoked via `pyroSAR` or subprocess with a graph XML. Keeps GPL isolated.

---

## 4.3 — P3: Co-registration

**Owner:** Geospatial engineer · **Effort:** ~4 days

Misregistration is the #1 cause of bogus change detection — every building outline becomes a false "change".

### Steps
1. Compare CRS, bounds, GSD across the pair
2. Reproject to a common metric grid (UTM). **Nearest-neighbour for masks, bilinear/cubic for continuous data** — never interpolate class IDs
3. Estimate residual shift: `skimage.registration.phase_cross_correlation`. For **optical↔SAR** use **mutual information** instead — the modalities share no visual features, so intensity correlation fails
4. Report `rmse_px`. Sub-pixel to ~1 px is good
5. If shift exceeds threshold → AROSICS correction, or refuse with explanation

### Output
```python
@dataclass
class CoregReport:
    coregistered: bool
    rmse_px: float
    method: str
    correction_applied: bool
    common_crs: str
    checks_passed: list[str]
```

**Note:** the ISRO eval pairs will arrive pre-co-registered. We still implement the check because (a) it's a named deliverable and (b) live-demo uploads won't be. [v3] On pre-co-registered pairs the check runs in verify-only mode and must not modify the data.

---

## 4.4 — P4: Tiling

**Owner:** Geospatial engineer · **Effort:** ~4 days · **Source:** GeoPixel partitioning method

A Cartosat scene can be 10,000×10,000+. Downsampling it to 512px destroys everything.

1. **Adaptive partition** into local tiles + one global overview (GeoPixel approach)
2. Overlap tiles by ~10% to avoid boundary artifacts
3. **Geo-index every tile** — store its transform so evidence maps back to scene coordinates
4. **Relevance scoring** (`tile_scorer`): CLIP-style similarity between the query text and each tile
5. Process top-k tiles only; hard cap k to bound latency
6. Mosaic outputs back into scene CRS

**[v3] Cut-list fallback:** if Phase 3 slips, tiling degrades to "global downsample + one query-relevant detail crop" (see Part 8 cut list). Benchmark chips (512px) never need tiling, so scores are unaffected — only full-scene demo polish is.

---

## 4.5 — P5: Agentic Controller

**Owner:** Agent/backend engineer · **Effort:** ~3 weeks · **Gate:** G6 (graded on the trace)

### 4.5.1 Task enum — freeze in Phase 0
```python
class Task(str, Enum):
    SINGLE_VQA = "single_vqa"
    SINGLE_CAPTION = "single_caption"
    SINGLE_GROUNDING = "single_grounding"
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_VQA = "change_vqa"
    CHANGE_MAP = "change_map"
    CROSSMODAL_EXTRACTION = "crossmodal_extraction"
    CROSSMODAL_VQA = "crossmodal_vqa"
```

### 4.5.2 Routing — **[v3] rules first, LLM as tie-breaker**

The old design put a zero-shot LLM at the single point that G6 grades, unvalidated until Phase 2. Reversed:

**Stage 1 — deterministic rules** (resolve the large majority of queries):
- 2 same-modality images (+ dates/temporal keywords) → `change_*`; sub-type by question form: ratio/area/count → `change_map` + `change_stats`; "describe the changes" → `change_description`; otherwise → `change_vqa`
- optical + SAR pair → `crossmodal_*`
- 1 image + locate/where/find + phrase → `single_grounding`; "describe/caption" → `single_caption`; else → `single_vqa`
- `BandInventory` gates which tools enter the plan (D3)

**Stage 2 — LLM tie-breaker** for the ambiguous residue only. **Constrained JSON decoding** via Outlines or vLLM guided decoding against a fixed schema. Never parse free-form LLM output for control flow.

Every trace records `router_path: "rules" | "llm"`. A mostly-rules trace is *more* defensible to a judge, not less — deterministic, auditable, reproducible.

**Validation timing:** the 300-query routing eval set is built in **Phase 0** alongside the enum freeze (same artifact, same sitting), and zero-shot routing accuracy — rules alone, LLM alone, hybrid — is measured in **Week 1**, not Week 5.

**[v3] Plan minimality:** prefer the smallest sufficient plan. If trajectory-style grading compares our plan to a reference, a 3-tool DAG where 1 tool suffices reads as inefficiency. The planner carries an explicit step-count penalty.

```json
{
  "task": "crossmodal_extraction",
  "required_inputs": ["optical", "sar"],
  "targets": ["water", "builtup"],
  "router_path": "rules",
  "plan": [
    {"tool": "spectral_index", "params": {"index": "NDWI", "threshold": "auto"}},
    {"tool": "sar_backscatter", "params": {"pol": "VV", "threshold": "auto"}},
    {"tool": "optsar_fusion", "params": {"adapter": "optsar_fusion@v3"}}
  ]
}
```

### 4.5.3 Validator gate — non-LLM, hard rules

Runs **before** any tool executes. Rules:

| Condition | Action |
|---|---|
| Change query + 1 image | Refuse, explain, request second image |
| Mismatched CRS/extent | Attempt registration, or refuse with RMSE |
| SAR-only + colour/spectral question | Explain modality limitation |
| Requested index needs absent band | **Reroute (D3)**, log substitution in `routing_notes` |
| **[v3] Pan-only/RGB-only optical + spectral question** | Reroute to SAR/texture if the question permits; otherwise refuse with explanation ("this sensor cannot answer spectral questions — here is what it *can* answer") |
| **[v3] Single-pol SAR + polarimetric question** | Explain limitation, answer what single-pol supports |
| nodata fraction > threshold | Warn, proceed with masked stats |

> A graceful, explained refusal scores better than a confident hallucination — and "compatibility checking" is a named deliverable.

### 4.5.4 Parameter enforcement gate **[v3.1 — new, C19]**

Sits between "the planner decided what to do" and "the tool actually runs." Roughly two days of work.

**The problem it solves.** Every tool has settings: `spectral_index` takes an index name and a threshold; `sar_backscatter` takes a polarisation and a dB cutoff. Under v3 we *recorded* those settings in the trace but never *checked* them. If the router handed `spectral_index` a threshold of 47, the system would run it, produce a meaningless mask, and faithfully write `"threshold_value": 47` into the graded artifact — documenting our own error rather than preventing it.

**How it works.** Every tool manifest already declares its permitted parameter names and their allowed ranges or enumerations. The gate reads the manifest and, for each planned step, checks:

1. Is every supplied parameter name **on the tool's permitted list**? Unknown names are rejected — not ignored.
2. Is every value **inside its declared range or enum**? Out-of-range values are **rejected, never silently clamped**. Clamping hides the planner's mistake; rejection surfaces it.
3. Are the parameters **coherent with `BandInventory`**? Asking for NDBI on a source with no SWIR fails here as well as in the validator — belt and braces, since this is a graded field.

On rejection the step does not execute. The controller either replans that step once or emits an explained refusal. Either way the trace records `parameter_check: {"passed": true|false, "rejected": [...]}`.

**Why it earns its two days.** The PS names permitted parameters as one of only four graded trace fields, and requires the controller to "configure only permitted task parameters." Without this gate that is a claim we assert. With it, it is a claim a judge can verify by reading the file — which is the entire point of the trace being the graded artifact.

### 4.5.5 Trace schema — freeze in Phase 0

**This is a graded artifact, not a log file.** The PS states only the observable execution trace is evaluated.

**[v3.1] Graded fields first (C21).** The PS evaluates exactly four things: selected task, model/tool names, permitted parameters, outputs. Those are hoisted into a top-level `graded` block so they are unambiguous and machine-readable, with the richer diagnostic content kept below it as a superset.

```json
{
  "query_id": "uuid", "timestamp": "ISO8601",
  "query_text": "...",

  "graded": {
    "task_selected": "crossmodal_extraction",
    "tools_invoked": ["spectral_index","sar_backscatter","optsar_fusion@qwen3vl-4b-v3"],
    "permitted_parameters": [
      {"tool":"spectral_index","params":{"index":"NDWI","threshold_method":"otsu",
       "threshold_value":0.14},"within_manifest":true},
      {"tool":"sar_backscatter","params":{"pol":"VV","threshold_db":-18.0},
       "within_manifest":true}
    ],
    "parameter_check": {"passed": true, "rejected": []},
    "outputs": {"answer":"...","masks":["water_mask.tif","builtup_mask.tif"],
                "area_km2":3.4,"confidence":0.81}
  },

  "router_path": "rules",
  "inputs": [{
    "file": "cartosat_mx.tif", "modality": "optical",
    "native_gsd_m": 2.0, "pixel_size_m": 2.0,
    "crs": "EPSG:32644", "bands": ["B","G","R","NIR"],
    "swir_available": false, "bit_depth": 12, "nodata_frac": 0.02
  }],
  "compatibility": {
    "coregistered": true, "rmse_px": 0.8,
    "checks_passed": ["crs_match","extent_overlap","gsd_ratio_ok"]
  },
  "routing_notes": ["SWIR unavailable; NDBI skipped; SAR primary for built-up"],
  "steps": [{
    "tool": "spectral_index",
    "params": {"index":"NDWI","threshold_method":"otsu","threshold_value":0.14,
               "bimodality_passed": true},
    "param_source": "manifest_validated",
    "outputs": {"mask_uri":"water_mask.tif","mask_crs":"EPSG:32644",
                "mask_res_m":2.0,"area_km2":3.4},
    "confidence": 0.79, "latency_ms": 180
  }],
  "agreement": {"iou": 0.86, "verdict": "consistent", "disagreement_cause": null},
  "fusion": {
    "model": "qwen3vl-4b-instruct+optsar_fusion@v3",
    "answer": "...", "confidence": 0.81
  },
  "evidence": ["overlay.png","water_mask.tif","builtup_mask.tif"],
  "warnings": ["NDBI unavailable: source lacks SWIR band"]
}
```

Rendered as a step panel in the UI and embedded in the PDF report.

### 4.5.6 Execution
LangGraph state machine. Independent tools run in parallel; dependent ones sequence. Every node appends to `steps[]`. **[v3.1] No node executes until the §4.5.4 parameter gate has passed for that step.**

---

## 4.6 — P6: Tool Registry

Every tool declares a manifest: name, description, required modalities, **permitted parameters with explicit names plus ranges or enumerations**, output schema, expected latency, confidence source.

**[v3.1] The manifest is a contract, not documentation.** It is the data the §4.5.4 gate enforces against, so it must be machine-readable and complete. A parameter that isn't in the manifest cannot be set. Example:

```yaml
name: spectral_index
required_modalities: [optical]
permitted_parameters:
  index:            {type: enum, values: [NDVI, NDWI, MNDWI, NDBI],
                     requires_bands: {NDVI: [nir,red], NDWI: [green,nir],
                                      MNDWI: [green,swir], NDBI: [swir,nir]}}
  threshold_method: {type: enum, values: [otsu, fixed]}
  threshold_value:  {type: float, range: [-1.0, 1.0], optional: true}
outputs:
  mask_uri:  {type: geotiff, crs: source, resolution: full_scene}
  area_km2:  {type: float}
```

**[v3.1] Mask output rule (C18).** The PS lists masks among the hidden set's reference annotation types, so every mask a tool produces is a potentially scoreable artifact. All masks are written as geo-referenced GeoTIFFs **in the source CRS at full scene resolution** — mosaicked back from tiles where tiling was used, never left at tile resolution and never only as a web overlay.

### 4.6.1 Learned tools (Qwen + hot-swap LoRA)

**[v3] Uniform contract:** all four adapters are plain LoRA on the *unmodified* base architecture, all consuming standard 3-channel `pixel_values` (multi-image where needed). This is what makes vLLM multi-LoRA serving actually work — see §5.2 and C1.

| Tool | Adapter | Inputs | Output |
|---|---|---|---|
| `rs_vqa` | `rs_vqa@vN` | 1 image + question | text answer + logprob |
| `rs_ground_caption` | `rs_ground_caption@vN` | 1 image + phrase (+optional point prior in prompt) | bbox / caption |
| `change_vqa` | `change_vqa@vN` | 2 images + question | text answer + logprob |
| `optsar_fusion` | `optsar_fusion@vN` | optical img + SAR stack img + question | text answer + logprob |

**Adapter naming rule:** always include the base — `change_vqa@qwen3vl-4b-v2`. Log it in the trace. Loading a 2B adapter onto a 4B base at 2am is a debugging session nobody needs.

### 4.6.2 `spectral_index` + `centroid_prior` — implements D2

```
NDVI  = (NIR − Red)   / (NIR + Red)
NDWI  = (Green − NIR) / (Green + NIR)     ← works on Cartosat MX
MNDWI = (Green − SWIR)/ (Green + SWIR)    ← needs SWIR
NDBI  = (SWIR − NIR)  / (SWIR + NIR)      ← needs SWIR
```

**[v3] Thresholding — Otsu behind a bimodality gate.** Otsu assumes a bimodal histogram; on a tile that is 95% land with one pond it returns a confident, meaningless threshold, and that error would flow silently into D1, D2, and D4. Gate:

1. Compute Otsu threshold and its between-class variance ratio (Otsu criterion / total variance).
2. Accept iff ratio ≥ 0.5 **and** both resulting classes ≥ 5% of valid pixels.
3. Else fall back to **fixed physical thresholds** (per-index table in `preprocessing.yaml`; initial values: NDWI > 0.2 water, NDVI > 0.3 vegetation; C-band VV σ⁰ < −18 dB water, > −3 dB dense built-up; provisional, tuned on holdout).
4. Record `threshold_method: "otsu" | "fixed_fallback"` and the value in the trace, and lower confidence on the fallback path.

"We detect when our own tools are out of their depth" is itself a judging story.

**`centroid_prior`:** connected-component analysis on the mask → largest component → centroid → normalised `(x, y)` point. Passed to `rs_ground_caption` inside the prompt as a spatial prior.

**[v3] Evidence status:** motivated by, not proven by, the BEN.txt tables (which compare two different tasks — see D2 in §1.3). Our own Week-1 zero-shot ablation is the number we build on and the number we quote.

### 4.6.3 `change_map`
**[v3.8] Rebuilt after C55/C56.** Siamese CD architecture (Open-CD or TorchGeo implementation — the *code*, not the authors' checkpoints), initialised from **TorchGeo MIT-licensed weights** (SSL4EO-S12 / SeCo), fine-tuned on **SpaceNet 7 / MUDS (C59, primary) + HRSCD (C60) + self-generated Sentinel pairs over Indian AOIs (C46) + OSCD (C49)**. Outputs a binary or semantic change mask as a full-scene-resolution geo-referenced GeoTIFF in the source CRS. Metrics: F1, IoU.

> **Why this changed.** TinyCD and ChangeFormer publish their pretrained weights under non-commercial/academic terms, so those checkpoints cannot ship (C55). That killed the escape hatch v3.3 relied on — and in doing so exposed a flaw in C33: it permitted LEVIR/SECOND for `change_map` because "the output is a mask, not weights." But `change_map` **is** a fine-tuned model, and "codes and models" means we ship it. The artifact-type distinction only held for a service that emits outputs without shipping the model. It doesn't apply here, so LEVIR, SECOND, LEVIR-MCI and QAG-360K are dropped entirely (C56).

> **What we gain.** The whole change pipeline is now clean end-to-end — clean architecture, MIT weights, self-generated Copernicus data over India — and the story to a judge is one sentence instead of three caveats.

> **What C59/C60 restore.** SpaceNet 7 brings back **expert labels** — change derived by differencing manually-annotated building footprints, the published recipe for that dataset — at 4 m across 6 continents, with tens of thousands of derivable bi-temporal pairs. HRSCD brings back **sub-metre semantic** change. Together they recover most of what C56 cost.
>
> **What still doesn't match SECOND/LEVIR, stated.** SpaceNet 7 is buildings-only at 4 m rather than multi-class at 0.5 m; HRSCD is France-only. Expect `change_map` F1/IoU somewhat below published benchmarks, though much closer than the self-generated-only position. Offsetting: CDVQA stays as the *evaluation* set (PS-nominated, so evaluating is compliance); `change_stats` (D4) is arithmetic and unaffected by mask model quality; and both SpaceNet 7 pair sampling and Copernicus volume are levers for spare A100 reserve (C42).
>
> **Bonus from C59:** SpaceNet 7's per-building tracking IDs give **counts** across timestamps — real ground truth for validating `change_stats` on the counting and ratio question types where every published model sits at 32–37%.

**[v3.8] Off the cut list and now fully clean (C18/C56).** The PS lists masks among the hidden set's reference types, so this tool produces a directly scoreable output.

### 4.6.4 `change_stats` — implements D4

Consumes the semantic change map, computes exactly:
- per-class area before / after / delta (m²)
- change ratio, class change ratio
- largest / smallest change by class

**Directly answers the CDVQA question types where every model from 2B to 9B is stuck at 32–37%.** [v3] Output passes through the answer formatter (§6.4) so the exact string matches what the official scorer expects.

### 4.6.5 `sar_backscatter`
Segmentation on σ⁰ dB, using the same Otsu-with-bimodality-gate + fixed-fallback machinery as §4.6.2, with thresholds keyed by `(sar_band, polarisation)`.
- Water: low backscatter (specular reflection away from sensor)
- Built-up: high backscatter (wall–ground double bounce)

Record the threshold, its method, and the band/pol assumption in the trace.

### 4.6.6 `lulc_classifier`
BigEarthNet-pretrained ViT, multi-label (sigmoid, not softmax — one 1.2 km patch can be forest *and* water *and* pasture). Temperature-scaled on validation.

**[v3] Demoted from "primary calibrated confidence source" to "one input signal among several."** It is trained on 1.2 km Sentinel patches; on a 2 m Cartosat tile its outputs are out-of-distribution and its temperature fit does not transfer. It contributes a feature to the end-to-end calibrator (§4.7.3); it no longer anchors system confidence. Also a cut-list item.

### 4.6.7 `coreg_check`, `tile_scorer`
Wrappers over P3 and P4, exposed as callable tools so their execution appears in the trace.

### 4.6.8 `texture_seg` **[v3 — new]**
### 4.6.9 `object_box_fallback` **[v3.3 — new, C32]**

**Why it exists.** As of v3.8 the learned adapter covers land-cover regions, **buildings, aircraft and ships** (C36/C37/C53). It does **not** cover vehicles, storage tanks, bridges or harbours — those classes exist only in Google Earth-derived datasets we won't ship. Without this tool, a query for one of them scores **zero**. With it, it scores **partial credit** — no restricted data, no training.

**[v3.4] Narrowed remit.** The router sends a grounding query here only when the requested class falls outside the trained vocabulary. Covered classes go to `rs_ground_caption` as normal.

**Method — entirely deterministic:**
1. Percentile-stretch and, on SAR, apply the σ⁰ threshold table; on optical, use `texture_seg` edge-density and local-variance response.
2. Connected-component analysis over the response map, with morphological opening to suppress speckle.
3. Filter components by area, aspect ratio and solidity against a small prior table keyed to the query noun — compact bright high-backscatter blobs for *ship / tank / aircraft*, elongated linear structures for *bridge / runway*, dense high-texture clusters for *built-up*.
4. Emit axis-aligned boxes, ranked by response strength, **capped at the top few**.

**Confidence contract.** The manifest declares this tool as a low-confidence proposer. Every box it emits is flagged `"method": "deterministic_fallback"` in the trace, and system confidence is floored accordingly. It never masquerades as learned grounding.

**Effort:** ~2 days. It reuses `texture_seg`, the connected-component code already written for `centroid_prior`, and the σ⁰ tables from `sar_backscatter` — so it is assembly, not new machinery. **[v3.4] Still worth building even with C36/C37 in hand** — it is the only thing standing between an uncovered-class query and a zero.

> **[v3.4] The honest framing for a judge:** *we train object grounding for the classes where permissively-licensed data exists — buildings and aircraft — and land-cover regions from BigEarthNet. For everything else, every available dataset is Google Earth-derived and academic-only, and we weren't willing to train your deliverable on imagery we can't license. So the system localises what it can with classical vision, labels those boxes as deterministic proposals, and says so in the trace.* That is a better answer than a confident wrong box from data we should not have used.

---

## 4.7 — P7: Fusion & Confidence

**Owner:** ML — temporal & cross-modal · **Effort:** ~2 weeks · **Gate:** G5

### 4.7.1 Decision-level fusion — implements D1

Two independent decisions, then reconcile. *(Not feature-level: EarthMind's own ablation showed naive token concatenation fails on optical–SAR heterogeneity, and decision-level is more robust to residual misregistration and gives interpretable confidence.)*

| Target | Optical evidence | SAR evidence |
|---|---|---|
| Water | MNDWI (or NDWI, or texture_seg) > gate | σ⁰ < threshold |
| Built-up | NDBI (or texture_seg) > gate | σ⁰ high (double bounce) |

**[v3] Schedule change (C6):** D1 is deterministic — a rule table over two masks — and is built in **Phases 1–2**, demoable by Week 4. The `optsar_fusion` adapter (Phase 3) is an upgrade layered on top. If the adapter underdelivers or training slips, the demo centrepiece already exists.

### 4.7.2 Disagreement classification

```python
DISAGREEMENT_RULES = [
  ("sar_water_optical_not", "cloud_over_water",     "sar"),
  ("sar_dark_optical_soil", "wet_smooth_soil",      "optical"),
  ("sar_dark_terrain_slope","radar_shadow",         "optical"),
  ("sar_bright_over_water", "wind_roughened_surface","optical"),
  ("sar_dark_arid_region",  "dry_smooth_sand",      "optical"),
]
```

Output the cause, the winning modality, and lowered confidence. **Never silently pick one.**

This is unique in the literature and it is the pitch's centrepiece — which is exactly why it now ships in Phase 2, not Phase 3.

### 4.7.3 Confidence & calibration — **[v3] rewritten**

**What we actually claim:** a calibrated **P(answer correct)** on the final system answer. Not per-component calibration stapled together.

- **Component signals (features, not the claim):** thresholded-mask statistics, cross-modal agreement IoU, `threshold_method` flag, classifier scores, VLM answer log-probability (a weak signal for free-form text — used as a feature, never alone), router path, warning count.
- **Calibrator:** isotonic regression (or logistic if data is thin) mapping the feature vector → P(correct), implemented with scikit-learn.
- **Fit data:** ~500 labelled end-to-end system outputs. These come from running the system over the 300-query routing set plus benchmark validation queries and labelling correctness — the same labelling effort the routing eval already requires. One labelled set, used twice. Collected Phase 2, calibrator fit early Phase 3, ECE reported on a held-out ~150.
- **Aggregation before the calibrator exists** (Phases 1–2): conservative min/product as an interim heuristic, clearly labelled `"confidence_basis": "heuristic"` in the trace until the fitted calibrator replaces it (`"calibrated"`).

**Produce a reliability diagram and report ECE in the final deck.** No system in the landscape survey reports calibration; this is close to free differentiation — but only if the number is honest, which the above makes it.

---

## 4.8 — P8: Evidence & Reporting

**Owner:** Frontend + backend · **Effort:** ~1.5 weeks

- **Masks** written as geo-referenced GeoTIFFs in the source CRS — downloadable, loadable in QGIS
- **Overlays** as PNG for web display, aligned to the map viewer
- **PDF report:** query, answer, confidence, evidence images, full trace table, tool parameters, warnings, timestamp
- **Trace panel:** step-by-step in the UI, expandable per tool

---

## 4.9 — P9: Model Serving — **[v3] rewritten**

**Owner:** Agent/backend · **Effort:** ~1 week

**vLLM with multi-LoRA.** One base model in VRAM, adapters swapped per request. This works **because** of decision C1: all four adapters are plain LoRA on the unmodified base with a uniform 3-channel input contract. (The old plan's extra vision encoders would have made this impossible — vLLM serves registered architectures, not custom towers.)

Cap `max_pixels` in the processor — the single biggest lever on both training and inference cost.

**[v3] Phase 0 exit test (non-negotiable):** base model + two dummy-trained LoRA adapters hot-swapping under vLLM **on the named demo machine**, before any real training data is formatted. If serving constraints force a change, they must force it before training starts, not after.

**[v3] Stretch experiment exception:** if the §5.6 multi-encoder experiment runs, it is served as a *separate* HF-transformers process behind the same tool manifest — never inside the vLLM instance, never on the demo critical path.

Fallback path: quantised local serving for an offline venue demo. Keep it working throughout, not as an afterthought.

## 4.10 — P10: Frontend

**Owner:** Frontend engineer · **Effort:** ~3 weeks

| Component | Detail |
|---|---|
| Upload | Drag-drop, multi-file, pair designation (optical/SAR, date1/date2) |
| **[v3] Preparation progress** | Async P1–P4 job status per scene, so the two-SLA split is visible, not hidden |
| Compatibility panel | Shows the `CompatibilityReport`, warnings visible not buried |
| Map viewer | MapLibre GL, geo-referenced overlays, opacity slider, layer toggle |
| Chat | Query input, streamed response via SSE |
| Evidence panel | Masks, overlays, area statistics, agreement verdict |
| **Trace viewer** | Step-by-step, per-tool params and confidence — *the graded artifact* |
| Export | PDF report + GeoTIFF mask download |

## 4.11 — Headless evaluation mode **[v3 — new]**

**Owner:** Agent/backend · built Week 2, green in CI from Week 2 onward.**

```
python -m satquery.evalcli \
    --images scene_opt.tif scene_sar.tif \
    --question "..." \
    --out answer.json trace.json
```

- Single container, single command. No frontend, no Redis, no Celery, no network.
- Deterministic tools run CPU-only if no GPU is present; learned tools use the quantised local path.
- Batch mode: consumes a JSONL of (images, question) rows — the shape any benchmark harness or hidden-set evaluation will take.
- This is simultaneously: (a) the artifact we submit if organisers run our code, (b) the venue-demo parachute, (c) the CI smoke test.

Whatever the submission spec turns out to say, this mode satisfies the most restrictive plausible version of it.

---

# PART 5 — Model & training pipeline

## 5.1 Base model

**`Qwen3-VL-4B-Instruct`**, standard checkpoint (not FP8 — FP8 needs Ada/Hopper and breaks LoRA tooling).

| Reason | Evidence |
|---|---|
| 4B beats 8B on our key task | CDVQA test1 AA 67.86 (4B) vs 66.94 (8B) |
| Native dynamic resolution | No forced 448px squish — critical for RS |
| Instruct > Thinking for our tasks | 94.4 vs 93.6 ScreenSpot grounding; PS doesn't evaluate reasoning text; our data has no reasoning traces |
| Apache 2.0 | Clean for deliverables |

**Open bake-off (Phase 0):** `Qwen3.5-2B` scored highest on CDVQA test2 (AA 69.56) and Alibaba claims native-multimodal beats Qwen3-VL at similar scales. But grounding scales with size (RefCOCO val@50: 85.7 at 4B → 86.9 at 8B), and grounding is G3. **Test both zero-shot on CDVQA *and* a grounding set, then commit to one.** Don't split bases across adapters — that doubles VRAM and kills the multi-LoRA argument. [v3] The bake-off also feeds the compute triage ladder (§5.5): if the ladder forces 2B, the bake-off numbers tell us exactly what we're giving up.

## 5.2 Adaptation architecture — **[v3] rewritten (decision C1)**

**All four adapters: plain LoRA on the unmodified Qwen3-VL architecture. No added encoders, no custom projections, uniform input contract.**

Why the change: the previously planned RS-InternVL dual-ViT recipe is an architectural modification, not a LoRA. vLLM multi-LoRA cannot serve it; the four adapters would not have shared an input pipeline (12-band tensors vs `pixel_values`); and the incompatibility would have surfaced at integration in Phase 3–4. The recipe survives as an explicitly optional stretch experiment (§5.6) with its own serving path.

**Input representations (frozen in `preprocessing.yaml`, identical at train and inference):**

| Modality | Model input |
|---|---|
| Optical, RGB only | True-colour 3-channel composite |
| Optical, NIR present *(e.g. Cartosat-2S MX)* | True-colour **+** false-colour (NIR, R, G) composite, as two images |
| **[v3.1] Optical, full multispectral *(e.g. Sentinel-2 / BEN.txt)*** | **Three composites: true-colour (B,G,R) + false-colour (NIR,R,G) + short-wave (SWIR2,SWIR1,NIR)** — passed as three images. Qwen3-VL is natively multi-image, so this reaches most of the 10 m/20 m band set through the standard vision tower (C22) |
| Optical, pan-only | Single-channel replicated is *not* used for learned tools; pan inputs route to deterministic `texture_seg`/SAR paths (§4.1.4) |
| SAR, dual-pol | `[pol1_dB, pol2_dB, ratio_dB]` stack (§4.2) |
| SAR, single-pol | `[σ⁰_dB, filtered_dB, GLCM texture]` stack (§4.2), with 25% pol-dropout during training |

> **[v3.1] Why three composites (C22).** RS-InternVL's headline gains come partly from genuine 12-band input via dedicated frozen S1/S2 encoders — and the BEN.txt benchmark deliberately tests CLC classes that RGB cannot separate (the paper's example: *Mixed forest* vs *Coniferous forest*). Decision C1 gives up those encoders to keep vLLM serving working. The three-composite trick recovers most of the spectral range at zero serving cost, since it is still standard `pixel_values` through the standard tower. **Cost:** ~2–3× the image tokens per multispectral sample, which lands directly on the `max_pixels` budget — so cap resolution per composite accordingly and ablate two-composite vs three-composite in Phase 3. On Cartosat inputs the question is moot: there is no SWIR, so it is always two.

**What we keep from RS-InternVL (hyperparameters, not architecture):**
- LoRA r=16, α=32, dropout 0.1 (paper used r=8; scaled slightly for the larger base)
- LR: warmup 1e-6 → 1e-4 over first 1% of steps, then cosine decay
- 1 epoch on train+val
- Process 10 m and 20 m S2 bands only when composing inputs; drop 60 m bands
- QLoRA 4-bit base if VRAM-constrained

> **S3 warning still applies, transposed:** never feed the model an input distribution it wasn't trained on. That is exactly why pol-dropout, GSD conditioning, and the frozen input-representation table exist. Ablate against RGB-only on every run to prove the extra views help.

## 5.3 Adapters

**[v3.1] Dataset roles follow the PS (C20).** The PS assigns BEN.txt to adaptation and VRSBench/RSVQA/CDVQA to evaluation. So BEN.txt is the clear majority of every mix; the others contribute train splits only in the minority, and that use is disclosed in `../../CREDITS.md`. **VRSBench leaves training entirely** — it is designated evaluation *and* it is the DOTA-derived academic-only source, and "codes and models" is a listed deliverable, so weights trained on it would become part of a government deliverable. BEN.txt supplies referring-expression annotations anyway, so the cost is near zero.

| Adapter | Data (BEN.txt majority) | Gate |
|---|---|---|
| `rs_vqa` | **BEN.txt binary + MCQ VQA (primary)**; RSVQA-LR train split; **[v3.2] RSVQA-HR train split (C27)**; **[v3.2] SARLANG-1M QA + BEN modality dropout (C26)** | G2 |
| `rs_ground_caption` | **BEN.txt referring LULC + point detection + captions (primary)**; **RarePlanes aircraft (C36)**, **SpaceNet 6 + OpenEarthMap-SAR buildings (C31/C37)**, **[v3.8] LS-SSDD ships (C53 — CLEARED)** (+ centroid-prior prompts if D2 confirmed) | G3 *(regions + buildings + aircraft + ships)* |
| `change_vqa` | **[C59] SpaceNet 7 / MUDS (PRIMARY — expert footprint-differenced labels)** + **[C60] HRSCD sub-metre semantic change** + self-generated Sentinel pairs over Indian AOIs (C46, domain match) + OSCD incl. S1 extension (C49). CDVQA / SECOND-CC / QAG-360K removed — all descend from unlicensed SECOND | G4 |
| `optsar_fusion` | **BEN.txt S1+S2 pairs (primary)**; **[v3.3] SpaceNet 6 MSAW with generated captions (C31)** — the only sub-metre cross-modal supervision we have; SARLANG-1M (incl. single-channel), BigEarthNet-MM | G5 *(bonus — D1 still carries the gate)* |

**VRSBench remains in §6 as an evaluation set** — reporting grounding acc@0.5 / acc@0.7 on it is exactly what the PS asks for, and evaluating on it creates no deliverable exposure.

### 5.3a Two data decisions the PS forces **[v3.2]**

**Object grounding — partially recovered, and honestly scoped (C32 → C36/C39).** The PS asks for *"land-cover **and major objects**"*. BEN.txt's referring annotations target CORINE **land-cover class instances**, not objects.

The first licence sweep rejected DIOR, FAIR1M, RSICD and NWPU-Captions — all Google Earth-derived — and concluded that object grounding was impossible on a shippable licence. **That conclusion was too broad.** The error was searching for "object datasets" as a category rather than working backwards from licence-clean imagery bases. A second sweep found two:

| Capability | Source | Licence |
|---|---|---|
| **Aircraft** | RarePlanes — 14,700 hand-annotated real aircraft at 30 cm, plus ~630k synthetic | CC BY-SA 4.0 |
| **Buildings** | SpaceNet 6 (~48k footprints) + OpenEarthMap-SAR (178.5k building segments) | CC BY-SA 4.0 / CC BY 4.0 |
| **Land-cover regions** | BEN.txt referring LULC + point detection | CDLA-Permissive 1.0 |
| **Ships** | LS-SSDD-v1.0 — 6,015 expert boxes on Sentinel-1 SAR | **Apache-2.0 — CLEARED (C53)** |
| Vehicles, tanks, bridges, harbours | — | **None clean. Out of scope.** |

**Decision: G3 ships as referring grounding over land-cover regions, buildings, aircraft and ships (C53).** This is a narrower claim than "all objects" and a wider one than v3.3's region-only. Crucially it is a promise we can keep.

**Still scoped in three places, and all three must exist before Week 2:**

1. **In this plan** — here and in the §1.1 coverage matrix.
2. **In the validator** (§4.5.3) — a query for an *uncovered* class routes to `object_box_fallback` (§4.6.9), returns boxes flagged as deterministic proposals, and says which classes are supported. Partial credit, never a confident wrong box.
3. **In the deck and `../../CREDITS.md`** — *we ground land-cover regions, buildings and aircraft, trained only on permissively-licensed data. For other object classes we fall back to classical vision and label those boxes as deterministic proposals, because every dataset covering them is Google Earth-derived and we weren't willing to train your deliverable on imagery we can't license.*

**Do not reopen this by reintroducing VRSBench, DIOR or xView3.** The reasoning that excluded them has not changed, and xView3 specifically ships composites marked rights-reserved.

**SAR-only single images (C26).** The PS permits *"one optical/multispectral **or SAR** image"* for VQA, captioning and grounding. But BEN.txt always supplies co-registered S1+S2 — the source paper feeds both branches — so a SAR-only single image at inference was out of distribution for `rs_vqa`. Two fixes, both cheap:

1. Add SARLANG-1M question–answer pairs to the `rs_vqa` mix.
2. **Modality dropout:** train ~20% of BEN samples with S2 withheld, so the adapter has genuinely answered questions from SAR alone. This mirrors the polarisation-dropout logic in §4.2 — never let inference be the first time the model meets an input configuration.

Merge `rs_vqa` + `optsar_fusion` if evaluation shows no conflict — same architecture now (C1 makes this legal again), same source data, saves ~25% compute. This is triage-ladder rung 2, not a default.

**[v3.3] Resolution honesty for `optsar_fusion` — revised (C28 → C31).** Our bulk cross-modal supervision is BEN.txt at 10 m Sentinel, against a metre-class Cartosat/RISAT eval. **SpaceNet 6 MSAW (§5.3b) now supplies genuine sub-metre co-registered optical–SAR**, which v3.2 wrongly asserted did not exist — but it is one city, one class, and aerial rather than spaceborne SAR geometry, so it narrows the gap rather than closing it. SARLANG-1M helps the SAR side but is SAR image–text, **not** cross-modal pairs.

Therefore, in plain terms: **the deterministic path carries G5. `optsar_fusion` is a bonus we cannot guarantee.** D1 — spectral index versus backscatter threshold, reconciled by a physical rule table — is resolution-agnostic and transfers to any sensor pair, which is exactly the property the learned model lacks. GSD conditioning and multi-scale augmentation narrow the learned gap; they cannot manufacture sub-metre cross-modal texture.

This is a design decision, not a shortfall, and it should be said out loud before a judge finds it: *we put the guaranteed capability on physics and the upside on learning, rather than betting the mandatory gate on a model trained 5–10× away from the test distribution.*

**Change training data (C46) — why we generate our own.** Nearly all change-detection research runs on one dataset, SECOND, which carries **no licence statement at all**. CDVQA, SECOND-CC and QAG-360K all descend from it, so what looked like three independent sources was one source three times, and G4 had no clean training data.

Rather than negotiate that, we rebuilt the change mix from clean sources — **SpaceNet 7 (C59) is the primary expert-labelled source**, HRSCD (C60) adds sub-metre semantic change, and we generate our own Sentinel data for Indian domain match:

1. **Acquire** bi-temporal Sentinel-1/Sentinel-2 pairs over Indian AOIs — reusing the India holdout v0 machinery, so no new tooling.
2. **Label** change by index differencing (NDVI/NDWI/NDBI deltas, σ⁰ deltas) with the same bimodality-gated thresholding as §4.6.2, so labels inherit a method we already trust and can explain.
3. **Caption** via the BEN.txt template-plus-augmentation recipe.
4. **Supplement** with OSCD (24 clean Sentinel-2 pairs, plus its S1 extension — the only optical–SAR *change* data in the inventory).

**The self-generated portion is an upgrade, not a compromise.** Copernicus grants unlimited volume and a clean licence, and the imagery is **Indian** rather than 0.5 m aerial photography of Hangzhou. It closes a licence gap and a domain gap simultaneously. Cost: roughly one week of the geospatial engineer's time.

**Honest limit:** index-differenced labels are weaker than SECOND's expert pixel annotations. Mitigate by keeping CDVQA as the *evaluation* set — it is PS-nominated, so evaluating on it is compliance — and by treating `change_map` (which may still use SECOND/LEVIR/QAG under the C33 mask rule) as the higher-fidelity spatial output.

### 5.3b SpaceNet 6 — how we actually use it **[v3.3, C31]**

SpaceNet 6 has no image–text annotations, so it is not drop-in VLM data. Four uses, in priority order:

1. **D1 validation with ground truth (highest value, zero training).** ~48,000 building footprints over 120 km² at ~0.5 m give us real masks to score optical-vs-SAR agreement IoU against. Until now `Cross-modal agreement rate` in §6.2 was a *reported* number with nothing to check it. Now it is a *measured* one.
2. **X-band and single-pol stress testing (C2, C26).** RISAT-2B/2BR1 are X-band; everything else we hold is C-band Sentinel-1. The Capella data is X-band quad-pol at sub-metre, so we can (a) tune the X-band σ⁰ threshold and sanity tables on real data rather than literature guesses, and (b) synthesise genuine single-pol from real quad-pol to validate the pol-dropout strategy. **These assumptions were previously untestable.**
3. **`optsar_fusion` training via generated captions.** Apply the BEN.txt recipe — template captions built from the footprint reference maps, then LLM linguistic augmentation — to produce cross-modal instruction data at 0.5 m. This is the only sub-metre cross-modal supervision available to us.
4. **Demo material.** A metre-scale optical–SAR pair with verifiable ground truth is a far stronger live demo than a 10 m Sentinel patch.

**Limits, stated:** one city (Rotterdam), one class (buildings), aerial rather than spaceborne SAR geometry, and CC BY-SA means attribution plus share-alike on any redistributed derivative of the data itself. Generated captions must be released under compatible terms if we distribute them.

### 5.3c OpenEarthMap-SAR — the second cross-modal source **[v3.4, C37]**

Complements SpaceNet 6 rather than duplicating it, on three axes that matter:

| Axis | SpaceNet 6 | OpenEarthMap-SAR |
|---|---|---|
| Geography | 1 city (Rotterdam) | **35 regions, 3 countries** (JP/FR/US) |
| Polarisation | Quad-pol | **VV or HH single-pol** — the RISAT FRS case directly |
| Labels | Buildings only | **8 classes** — building, water, road, tree, agriculture, rangeland, developed, bareland |
| Platform | Aerial | Satellite (Umbra Spotlight) |

Together they give us metre-scale cross-modal data that is geographically diverse *and* covers both quad-pol and single-pol. Uses: sub-metre region-grounding supervision, D1 validation on water and building masks, single-pol stack tuning, and generated cross-modal captions for `optsar_fusion` via the BEN.txt template-plus-augmentation recipe.

> **Three limits to hold in mind, and one Phase 0 check.**
> 1. **It is semantic segmentation, not object detection.** It supplies *segments*, so it strengthens region grounding and building localisation — it is not a source of aircraft, ships or vehicles.
> 2. **Most labels are pseudo-labels** generated by optical models, with only ~20 manually annotated images per region. Mean agreement with real labels is ~68%, and Bareland is effectively unusable (0.02 IoU). **Train on the manually-labelled subset where label quality matters; treat pseudo-labels as weak supervision only.**
> 3. **SAR-only segmentation is much weaker than optical** in the paper's own baselines (~35 vs ~57 mIoU). That is a *finding*, not a defect — it independently corroborates our S3 thesis that SAR alone under-performs and must be fused rather than substituted.
> 4. **Verify the SAR band in Phase 0.** Umbra operates an X-band constellation, which would make this a second X-band source alongside SpaceNet 6 — but the paper does not state the band, so confirm it from the Umbra open-data catalogue before relying on it for X-band tuning.

## 5.4 Data curation

- **[v3] Manifest first (C13, revised by C29):** select the patch-ID manifest from metadata *before* downloading anything; download only manifest patches + official benchmark splits; stage as **three or four** private Kaggle Datasets (~100–140 GB total) in the priority order in §2.2, so Phase 1 training starts before staging finishes
- **[v3.2] Modality dropout (C26):** ~20% of BEN samples train with S2 withheld, giving `rs_vqa` genuine SAR-only experience
- **[v3.2] GSD stratification (C27):** RSVQA-HR at 15 cm, LEVIR at ~0.5 m and **SpaceNet 6 at ~0.5 m** are our sub-metre sources. Ensure they are not diluted to noise by the 10 m Sentinel majority — sample them at a floor of ~15% of each relevant mix, and report per-GSD-band accuracy on the smoke-eval set
- Balance by task type and LULC class (BEN is heavily skewed toward a few CORINE classes)
- ❌ **Drop Country, Climate Zone, Season MCQ tasks.** Derived from ten European countries and Köppen-Geiger maps. "Which country is this" trained on Europe is an actively harmful prior for Indian imagery
- Hold out the official BEN.txt benchmark split (1,082 pairs, 15,029 annotations, answer-balanced)
- Keep a fixed **2k smoke-eval set** runnable in 5 minutes after every run
- **GSD-conditioned prompting:** inject real ground sample distance into the text context at train *and* inference, so the model treats scale as a variable rather than a hidden constant
- **Multi-scale augmentation:** randomly resample across a GSD range
- **[v3] Answer-format discipline in training data:** targets use each benchmark's canonical answer strings (closed vocabularies where they exist), so the formatter (§6.4) is normalising, not translating

## 5.5 Compute — **[v3.5] A100-80GB budget (C41)**

**Confirmed access: ~50 hours on a single A100-80GB.** This replaces the T4 free-tier plan as the primary compute path. The T4 tables and triage ladder are retained below as the fallback if access falls through.

### The budget

Per adapter, 4B base, 60–80k samples, 1 epoch. **Pre-measurement estimates — replaced by the Phase 0 200-step timing test.**

| Item | A100-80GB | Notes |
|---|---|---|
| Smoke tests + 200-step timing + debug | **3 h** | Non-negotiable. Spend first |
| `rs_vqa` | 4 h | **Run first** — cheapest adapter, exercises the whole data → train → serve path |
| `change_vqa` | 6 h | |
| `optsar_fusion` | 6 h | Includes SpaceNet 6 / OEM-SAR generated captions |
| `rs_ground_caption` | 12 h | ~40% of cost is image size. Uncuttable (C17) |
| **Scheduled total** | **31 h** | |
| **Reserve** | **19 h** | Reruns, one failed config, one grounding re-do |

Roughly 60/40 planned work to recovery. **That ratio is the point.** 50 hours is enough for one clean pass *plus* the reruns you will actually need; it would not be enough for one pass plus reruns if the schedule consumed 45 of them. Nobody's first training run is their last.

### What this changes

**The triage ladder is effectively obsolete.** At 50 A100-hours we sit at rung 1: 4B base, 60–80k samples, four separate adapters. No drop to 2B, no forced merge of `rs_vqa` + `optsar_fusion`. Keep the ladder on paper only for the access-falls-through case.

**Config changes the A100 unlocks:**
- **bf16 on** (blocked on Turing) — more stable than the fp16 + fp32-LoRA workaround
- **FlashAttention-2 on** (blocked on Turing)
- **Drop QLoRA** — plain bf16 LoRA; a 4B base fits 80 GB trivially
- **Raise batch size** until utilisation saturates, reducing gradient-accumulation overhead

> **What does NOT change: `preprocessing.yaml`.** Extra VRAM is not a reason to relax `max_pixels` or any input representation. Train on exactly what the deployed system will see. Violating this reintroduces the S3 failure mode by the back door.

**One decision now has a visible price.** The three-composite optical input (C22) costs an estimated **5–8 of the 50 hours**, since it raises image tokens 2–3× on multispectral samples. **Ablate it on a short run before committing** rather than paying for it blind across every multispectral adapter. If two composites match three, that is 5–8 hours returned to reserve.

### Guard rails — this budget is non-renewable

1. **Never start a full run without a 200-step smoke on that exact config.** Ten minutes to protect twelve hours.
2. **Checkpoint every ~500 steps** (adapter + optimizer state, to the private HF repo). A crash costs minutes, not a run.
3. **`rs_vqa` runs first.** If the data format or LoRA wiring is broken, find out at hour 4, not hour 16.
4. **Re-derive this entire table from the Phase 0 200-step measurement.** Do not trust the estimates above past that point.
5. **Log hours consumed against this table weekly.** The integration lead owns the burn-down; it is the single number that predicts whether Phase 3 happens.

### Reserve allocation rule (C42)

If the first pass runs clean and ≥ 15 hours remain at end of Phase 2, the reserve is **not** spent on more reruns. It goes to **raising the sample count on `rs_ground_caption`.**

Rationale: referring detection is our weakest §6.1 target (≥45 mIoU against RS-InternVL's 65.84), and it is weak for exactly one reason — RS-InternVL fine-tuned per-task on ~100% of a ~347k-pair corpus and we train on ~2%. More data is the only lever that moves that number. No amount of hyperparameter tuning substitutes for it.

Second priority if grounding is already at target: raise `optsar_fusion` sample count using the sub-metre cross-modal data (C31/C37).

### Fallback — free-tier T4 plan (retained from v3)

*Applies only if A100 access falls through.* Per adapter on T4, 60–80k samples, 1 epoch:

| Adapter | 4B | 2B |
|---|---|---|
| `rs_vqa` | 15–35 h | 10–22 h |
| `optsar_fusion` | 25–50 h | 16–32 h |
| `change_vqa` | 30–60 h | 19–38 h |
| `rs_ground_caption` | 60–120 h | 38–76 h |

**Total with reruns: 200–320 T4-hours (4B) / 130–210 (2B).**

**The triage ladder (C3)** — within 48 h of the Phase 0 timing test, the integration lead projects total hours and applies the first matching rung:

| Projected total | Decision |
|---|---|
| ≤ 220 T4-h | Proceed: 4B, 60–80k samples, four adapters |
| 220–300 | 4B, cut to ~40k samples, merge `rs_vqa`+`optsar_fusion` |
| 300–400 | Drop to 2B across the board (bake-off numbers quantify the cost) |
| > 400 | 2B + ~30k samples, **`rs_ground_caption` trained at reduced scale but never dropped**. Cap `max_pixels` hardest on grounding |

> **[v3.1] Grounding is never a fallback casualty (C17).** The PS says the ISRO set carries "bounding boxes, or masks, as applicable", so grounding is scored on the hidden data whichever G3 option we nominally elect. Electing captioning would satisfy the gate and forfeit the points.

**Insurance (fallback path only):** pre-authorise **$60–100 of spot L4/A10**. Turns the 120 h `rs_ground_caption` run into ~15 h.

**T4 caveats:** Turing = **no bf16** (fp16 with LoRA layers in fp32; watch the first 200 steps for NaNs), **no FlashAttention-2** (SDPA/xformers). On Kaggle's 2×T4, run two separate experiments rather than DDP.

### Account & relay policy (C16)

- **One adapter per person, on that person's own account** — applies to the T4 fallback path.
- **Checkpoint relay is for resuming your *own* interrupted sessions.** Relaying a single run across multiple people's accounts to defeat per-account quotas violates Kaggle/Colab terms and risks bans mid-project. We don't do it.

**Cost lever ranking:** (1) cap `max_pixels`, (2) subsample harder, (3) merge adapters. `rs_ground_caption` is ~40% of the budget purely from image size.

## 5.6 Stretch experiment — RS-InternVL multi-encoder **[v3 — optional, off the critical path]**

The dual frozen BEN-ViT + projection recipe that took a 1B model from 31.73 → 65.84 mIoU remains scientifically attractive. It runs **only if** Phase 3 is green on every exit criterion, **only** for `optsar_fusion`, and is served as a separate transformers process (§4.9). It requires the BEN ViT licence to have cleared Phase 0 review (substitutes: torchgeo weights, DOFA, SatMAE). If it beats the plain-LoRA `optsar_fusion` on the holdout, it becomes an ablation slide; it never becomes a demo dependency. First item on the cut list.

---

# PART 6 — Evaluation

## 6.1 Targets

**[v3.1] Read the BEN.txt paper's tables correctly (C23).** Its "Qwen" row is Qwen3-VL **8B evaluated zero-shot** — a *baseline*, not a bar. Fine-tuning beats it almost automatically. The actual bar is **RS-InternVL**, which the paper fine-tuned **separately per task** on the combined train+val set (~347k pairs, millions of annotations), taking ~2 days on 4×H200. **We train on roughly 2% of that corpus.** Targets below are set accordingly, and every row now carries an honest confidence.

| Benchmark | Zero-shot Qwen | RS-InternVL (full data) | Our target | Confidence |
|---|---|---|---|---|
| BEN.txt captioning (BLEU-4) | 0.57 | 34.04 | **≥ 25** | **High** — captions are template-derived then paraphrased, so this is largely format learning and very sample-efficient. Cheapest headline number in the whole benchmark |
| BEN.txt binary VQA | 61.96 | 73.29 | **≥ 70** | **Likely** — expect 68–73 |
| BEN.txt MCQ *(full task set)* | 37.55 | 51.49 | **≥ 45** | Likely, *reported for comparability only* |
| BEN.txt MCQ *(our restricted set)* | — | — | **≥ 45, reported as our honest figure** | See caveat below |
| BEN.txt ref. detection (mIoU) | 15.60 | 65.84 | **≥ 45**, stretch 55 | **Lowest** — largest jump in the table, earned on ~50× our data. Plan for 35–50 |
| CDVQA test1 | Qwen3-VL-4B: AA 67.86 / OA 74.08 | — | ≥ AA 68, **test1−test2 gap < 2** | Likely |
| CDVQA test2 | Qwen3.5-2B: AA 69.56 | — | ≥ AA 68 | Likely |
| CDVQA "smallest change" | **32–37% every model** | — | **> 60% via `change_stats`** | High — arithmetic (D4), and **[C59] now validatable against SpaceNet 7 tracking IDs** |
| RSVQA-LR | Specialists ~92 avg | — | ≥ 88 | Likely |
| **[v3.2] RSVQA-HR** | Specialists ~90 avg | — | **≥ 85** | Likely — **and the single best public proxy for Cartosat-scale performance we have** (C27) |
| VRSBench grounding | — | — | report acc@0.5, acc@0.7 | Eval-only set (C20) |
| **[v3.4] Object grounding — aircraft** | — | — | **In scope (C36).** Report acc@0.5 on a RarePlanes real held-out split | Trained capability |
| **[v3.4] Object grounding — buildings** | — | — | **In scope (C31/C37).** Report acc@0.5 / mask IoU on SpaceNet 6 + OpenEarthMap-SAR | Trained capability |
| **[v3.8] Object grounding — ships (SAR)** | — | — | **In scope (C53).** Report acc@0.5 on an LS-SSDD held-out split | Trained capability; SAR-native |
| **[v3.4] Object grounding — other classes** | — | — | **Out of scope.** Report `object_box_fallback` acc@0.25/0.5 separately, labelled deterministic | Partial credit by design |

> **[v3.1] The MCQ comparability trap — report both numbers.** We drop the Country, Season and Climate Zone MCQ tasks (§5.4), correctly: a Europe-trained country prior is poison for Indian imagery. But those are precisely Qwen's *strongest* MCQ subtasks — Country 47.76 and Climate Zone 45.93, against Presence 34.41 and Counting 29.00 — and almost certainly RS-InternVL's strongest too, since country and climate zone are highly memorisable from spectral signature plus geography. So the 51.49 headline is inflated by exactly what we exclude, and our restricted-set number is **harder than it looks, not easier**. Always publish both, and say why we dropped them. That reads as rigour, not as a shortfall.

> **[v3.1] What is NOT covered by any row above.** Every number here is on 120×120 Sentinel-2 patches at 10 m. Hitting all of them says nothing about the Cartosat/RISAT hidden set. Benchmark performance and domain transfer are two independent risks and the plan treats them separately (§6.2 India holdout, Part 10).

## 6.2 System-level metrics (nobody else will report these)

| Metric | How |
|---|---|
| **Routing accuracy** | 300-query set (built Phase 0), confusion matrix over the task enum, split by `router_path` |
| **Invalid-config catch rate** | Deliberately malformed inputs — does the validator refuse? Includes [v3] pan-only, single-pol, X-band cases |
| **Calibration** | ECE + reliability diagram on the fitted end-to-end calibrator (§4.7.3), held-out ~150 |
| **Cross-modal agreement rate** | IoU between optical and SAR decisions. **[v3.3] Now scored against SpaceNet 6 building footprints (C31)** — previously reported with nothing to verify it |
| **Disagreement-cause accuracy** | Hand-labelled disagreement cases |
| **[v3.3] X-band / single-pol robustness** | σ⁰ tables and pol-dropout validated on real SpaceNet 6 X-band quad-pol, not literature estimates |
| **[v3.3] Object-fallback partial credit** | acc@0.25 / acc@0.5 of `object_box_fallback` on any object-level queries, reported separately and labelled deterministic |
| **India holdout** | v0 proxy (~50 samples) from Week 2; v1 real Bhoonidhi when granted; human-judged |
| **Latency** | [v3] Reported per SLA: prep-job p50/p95 per scene **and** query p50/p95 per task type |
| **[v3.1] Parameter-gate conformance** | % of executed steps whose parameters validated against the manifest; count and cause of rejections. Target: 100% of *executed* steps pass, with rejections visibly logged rather than absent |
| **[v3.1] Mask export conformance** | Every produced mask is a GeoTIFF, in the source CRS, at full scene resolution, loadable in QGIS. Automated check in CI — masks are graded outputs (C18) |

> **Watch the CDVQA test1−test2 gap on every single run.** Test2 has a deliberately shifted answer distribution. A large gap means the model learned the answer prior, not the task — and normalised scoring punishes that twice.

## 6.3 The baselines that matter most — all Week 1

1. **Zero-shot Qwen on RSVQA and CDVQA.** The "before" that proves G1 mattered; the first thing a judge asks for; doubles as an end-to-end pipeline test.
2. **[v3] D2 zero-shot ablation:** base model on ~200 grounding samples, with vs without injected centroid prior. Decides whether D2 is a claimed accuracy differentiator or an evidence-anchoring feature (§1.3). Costs one afternoon, saves a phase. **[v3.1] Expect the reframe branch to fire (C24)** — the paper's two referring tables use *different sample sets*, and a point prior helps some models while actively hurting others: LLaVa 23.04→17.27 and EarthMind-rgb 21.90→9.35 both *drop*. On Qwen specifically the gain is the smallest of any model that gained, 15.60→20.42 (+4.8), which sits right on our ≥5 mIoU confirmation threshold.
3. **[v3] Router zero-shot:** rules alone / LLM alone / hybrid on the 300-query set. If the hybrid is < 90% on unambiguous cases, the rules get fixed *now*, in Week 1, not discovered in Week 5.

## 6.4 Scoring integrity **[v3 — new]**

- **Official scorer scripts only.** Download and run each benchmark's own evaluation code in Phase 1. Our harness wraps them; it never reimplements the metric. Any gap between our numbers and the official script's is a P0 bug.
- **Per-benchmark answer formatter:** constrained decoding / post-normalisation onto each benchmark's closed answer vocabulary (yes/no, single word, canonical class names, canonical number formats). "The water body covers approximately 3.4 km²" scores zero where the scorer wants "yes" — format compliance is worth more points than a better model, and it costs two days.
- Formatter unit-tested against each benchmark's published examples.
- **[v3.1] The paper validates this emphasis.** Its tables carry an "IF" column recording whether an unambiguous answer could be extracted every time. Qwen passes on every task; most RS-specialist models fail. A large share of Qwen's lead over GeoChat, LHRS-Bot and SkyEyeGPT is instruction-following rather than vision — which is exactly the property the formatter layer builds on, and a second reason the base-model choice was right.

---

# PART 7 — Repository structure

```
satquery/
├── docker-compose.yml
├── ../../CREDITS.md                    ← every source + citation + licence
├── configs/
│   ├── preprocessing.yaml        ← FROZEN Phase 0: SAR chain incl. single-pol
│   │                                stack + pol-dropout, band harmonisation,
│   │                                threshold tables, sanity ranges, max_pixels
│   ├── trace_schema.json         ← FROZEN in Phase 0
│   └── training/*.yaml
├── satquery/
│   ├── ingest/                   P1  reader, modality, bands, radiometry
│   ├── sar/                      P2  SNAP graphs, dB, stretch, stacks
│   ├── coreg/                    P3  phase corr, mutual info, AROSICS
│   ├── tiling/                   P4  partition, geo-index, scorer
│   ├── agent/                    P5  enum, rules router, LLM tiebreak,
│   │                                 validator, executor, trace
│   ├── tools/                    P6  one module per tool + manifests
│   ├── fusion/                   P7  decision fusion, disagreement
│   ├── confidence/               P7  calibrator fit/apply, ECE      [v3]
│   ├── report/                   P8  overlays, GeoTIFF export, PDF
│   ├── serving/                  P9  vLLM multi-LoRA
│   ├── evalcli/                  §4.11 headless eval mode           [v3]
│   └── api/                      FastAPI routes, queue, workers
├── training/
│   ├── data/                     manifest builder, curation, formatting
│   ├── train_lora.py
│   └── eval/                     wrappers around OFFICIAL scorers + formatter
├── frontend/                     P10 React + MapLibre
└── tests/
```

---

# PART 8 — Implementation sequence

## Phase 0 — Foundations (Week 1)

**Nothing downstream can start until items 1–4 are done.**

1. Freeze `trace_schema.json` and `task_enum.py` — **and build the 300-query routing eval set in the same sitting** (same artifact, same mental model)
2. **Resolve every ⚠ licence** in Part 2 (integration lead, one day). Name confirmed substitutes for anything restricted. Nothing trains on ⚠ data. **[v3.4] Object-grounding review is COMPLETE (C36–C40): RarePlanes and OpenEarthMap-SAR cleared, xView3 rejected, LS-SSDD pending. Remaining Phase 0 licence work is RSVQA-HR annotations, SARLANG-1M per-subset (C35), and LS-SSDD (C40)**
3. **Obtain the submission/packaging spec** for the hidden set (integration lead). Until answered, §4.11 assumes the most restrictive case
4. Freeze `preprocessing.yaml` — SAR chain **including the single-pol stack, pol-dropout rate, per-band threshold and sanity tables**, band harmonisation, `max_pixels`
5. **Define the tool manifest format as an enforceable schema** (§4.6) — permitted parameter names plus ranges/enums and band prerequisites, machine-readable. This is the data the §4.5.4 gate validates against, so it is a contract, not documentation
6. Repo + Docker Compose + CI; one dummy tool emitting a valid trace **including a populated `graded` block and a passing `parameter_check`**
7. **Serving smoke test:** base + two dummy LoRA adapters hot-swapping under vLLM **on the named demo machine** (name the machine today). If serving forces an architecture change, it happens now — before any training data is formatted
8. **Manifest-first data staging:** build the 60–80k patch-ID manifest from metadata, download only manifest + benchmark splits, publish the private Kaggle Dataset
9. **Zero-shot baselines** — Qwen3-VL-4B *and* Qwen3.5-2B on RSVQA + CDVQA + a grounding set. Commit to one base
10. **D2 zero-shot ablation** (~200 samples, prior vs no prior) → decide D2's framing (§1.3)
11. **Router zero-shot measurement** on the 300-query set (rules / LLM / hybrid)
12. **200-step timing test** on real data → **apply the triage ladder within 48 h** (§5.5)
13. Apply for Bhoonidhi access (Day 1) **and start assembling India holdout v0 from open data** — no approvals needed
14. Set up W&B; test own-session checkpoint resume; decide the paid-GPU insurance budget
15. Build P1 ingest + validation, P3 co-registration check

**Exit:** a GeoTIFF pair uploads and produces a valid trace. Baselines + D2 ablation + router accuracy recorded. Base model chosen. Triage rung applied. Two adapters hot-swap on the demo machine. All licences resolved. Holdout v0 underway.

## Phase 1 — Vertical slice (Weeks 2–3)

16. Train `rs_vqa` (owner's own account — see §5.5 account policy)
17. P5 rules-first router + validator gate routing a single task
18. **[v3.1] Parameter enforcement gate live (§4.5.4)** with the first three tool manifests, plus a rejection test suite — deliberately malformed parameters must be refused, not clamped
19. **[v3.1] Mask conformance check in CI** — every mask-producing tool emits a full-scene-resolution GeoTIFF in the source CRS; automated assert, since masks are graded outputs. **[v3.2] Plus synthetic GeoTIFF ingest fixtures (C30)** — multi-band, 12-bit, unusual CRS, nodata regions, single-pol SAR, so the mandatory format path is tested against more than reBEN
20. **Headless eval mode (§4.11) working and in CI** — batch JSONL in, answers + traces out, CPU path green
21. Deterministic tools live: `spectral_index` (with bimodality gate), `sar_backscatter`, `texture_seg`, **`object_box_fallback` (§4.6.9, ~2 days)** — router sends only *uncovered-class* queries here (C39)
22. **Official scorer scripts integrated + answer formatter built and unit-tested** (§6.4)
23. P2 SAR chain end to end — both pol stacks
24. **D3 band-availability routing live** — tested on 4-band Cartosat-style, RGB-only, and pan-only inputs
25. Frontend v1: upload → prep progress → ask → answer → trace panel
26. **India holdout v0 assembled** (~50 samples, open data); swap plan for Bhoonidhi v1 written. **[v3.6] Same acquisition pipeline generates the `change_vqa` bi-temporal training pairs (C46)** — one piece of tooling, two deliverables

**Exit:** end-to-end single-image VQA on a real (or proxy) Cartosat-style tile, with a trace, scored by official scripts, reproducible headlessly.

## Phase 2 — Breadth + the centrepiece (Weeks 4–6)

27. Train `change_vqa`; fine-tune `change_map`
28. Build `change_stats` (D4) and wire the quantitative CDVQA question types through the formatter
29. Train `rs_ground_caption` on regions + **buildings + aircraft (C36/C37)**; productionise `centroid_prior` per the Week-1 D2 verdict; generate referring expressions from RarePlanes/SpaceNet6/OEM-SAR boxes via the BEN.txt template-plus-augmentation recipe
30. **D1 deterministic decision-level fusion + full disagreement rule table — demoable by Week 4** (moved up from Phase 3). **[v3.3] Scored against SpaceNet 6 building footprints (C31)** — real ground truth, not a self-reported number
31. **[v3.3] Tune X-band σ⁰ threshold and sanity tables on SpaceNet 6 Capella data; validate pol-dropout by synthesising single-pol from real quad-pol (C2, C26, C31)**
32. Multi-tool DAG execution, parallel where independent
33. P8 evidence overlays + geo-referenced mask export
34. **Collect the ~500 labelled end-to-end outputs** (routing set + benchmark val) for the calibrator
35. Routing confusion matrix re-run with trained components

**Exit:** all six gates minimally demonstrable **including the cross-modal disagreement demo**. All CDVQA question types answered, quantitative ones via tools. Labelled calibration data in hand.

## Phase 3 — Upgrade & polish (Weeks 7–8)

36. Train `optsar_fusion` — an upgrade over the D1 floor, never a dependency
37. Fit the end-to-end calibrator; reliability diagram + ECE (§4.7.3)
38. P4 GeoPixel-style tiling for full scenes
39. PDF report generation
40. Ablations: multi-sensor vs RGB-only; point-prior vs none (confirming Week 1 at scale); tool vs VLM on change ratios; single-pol stack vs naive repeat; **[v3.1/v3.5] two- vs three-composite optical input (C22) — run this EARLY in Phase 0, not here, since it costs 5–8 A100-h (C43)**; **[v3.3] `optsar_fusion` with vs without SpaceNet 6 generated captions (C31) — does sub-metre cross-modal supervision transfer?**
41. §5.6 stretch experiment **only if every prior item is green**

**Exit:** the cross-modal demo nobody else has, with ablation numbers to back it — and it already existed in Phase 2; this phase made it better.

## Phase 4 — Hardening (Weeks 9–10)

42. India holdout robustness pass (v1 if Bhoonidhi landed, else v0) — including X-band/single-pol synthetic stress cases
43. Failure modes, graceful refusals, edge cases (validator catch-rate suite)
44. Latency per the two SLAs: prep ≤ 5 min/scene, query < 20 s p95 on prepared bundles, demo scenes pre-warmed
45. **Fully offline demo** — headless mode + local Docker Compose, pre-cached scenes, quantised models on disk. Assume no internet and no cloud GPU at the venue. Rehearse on the named demo machine, twice
46. `../../CREDITS.md` final, README, tests, demo video, deck

---

## The scheduling rule

**Training must never be on the demo's critical path.** Adapters hot-swap. Build the entire app against the zero-shot base from Week 2 and upgrade as each adapter lands. Adapters train in parallel — one per teammate, each on their own account — while the backend and frontend proceed independently.

If a bad training week can kill your demo, the schedule is wrong.

## The Week-6 cut list **[v3 — agreed now, while calm]**

Reviewed at the end of Phase 2. If the integration lead judges the project > 1 week behind, cuts apply **in this order, no debate**:

1. §5.6 stretch experiment (auto-cut)
2. SMARTIES/DOFA band projection upgrade
3. Third optical composite (SWIR) → two-composite input only
4. `lulc_classifier` (its calibrator feature is replaced by mask statistics)
5. `change_map` **fine-tune** → pretrained CD backbone **zero-shot**. The tool itself is *not* cut — it still emits a geo-referenced mask, which is a graded output type (C18)
6. Full-scene tiling → global downsample + one query-relevant detail crop (benchmarks unaffected)
7. PDF report → styled HTML print/screenshot export

**Never on this list:** the six gates, the differentiators D1/D3/D4, **grounding (C17)**, **any tool's mask output (C18)**, and **the parameter enforcement gate (C19)** — the last three are graded artifacts, not polish.

---

# PART 9 — Team

**[v3] Capacity assumption, stated:** the schedule assumes **~25–30 focused hours per person per week**. This is a hackathon run alongside coursework — if actual capacity is materially lower, the integration lead re-baselines the phase lengths at the end of Week 2 rather than letting the plan silently slip. The triage ladder and cut list are the levers.

| Role | Owns | Phases |
|---|---|---|
| **Geospatial engineer** | P1–P4: GDAL/rasterio, SAR chain (both pol stacks), co-registration, tiling, band harmonisation, holdout v0 assembly | 0–3 |
| **ML — single image** | `rs_vqa`, `rs_ground_caption`, D2 ablation + point prior, RSVQA/VRSBench harness, **[v3] + `change_map` fine-tune** | 1–3 |
| **ML — temporal & cross-modal** | `change_vqa`, `optsar_fusion`, CDVQA harness, D1 rule table, P7 fusion | 2–3 |
| **Agent/backend** | P5 (rules router + validator), P6 registry, P9 serving, §4.11 headless mode, FastAPI, queue, trace emitter, **[v3] + `change_stats` (D4 — pure arithmetic, belongs with the tool layer)** | 0–3 |
| **Frontend** | P10, P8 UI side, trace viewer, report export | 1–4 |
| **Integration/eval lead** | Licences, submission spec, triage ladder, India holdout, calibrator, routing eval, ablations, demo, deck. **Veto power on scope; owns the cut list.** | 0–4 |

*[v3] Rebalance rationale: the temporal/cross-modal seat previously carried four trainings plus D1, D4, and P7 — roughly double any other seat. `change_map` moves to the single-image ML seat (their heavy training lands earlier) and `change_stats` moves to backend (it's arithmetic + a manifest, not ML).*

Everyone writes tests for their own layer. The integration lead owns the demo.

**Work discipline:** if anyone spends more than one day debugging a stranger's `requirements.txt`, pull them off it and reimplement the method instead. This is the most common way hackathon teams lose a week.

---

# PART 10 — Risks — **[v3] now decisions, not techniques**

Every row has a trigger you can observe, a threshold, an owner, and a deadline. A mitigation without a trigger is a hope.

| Risk | Sev | Trigger / threshold | Decision when triggered | Owner | Check by |
|---|---|---|---|---|---|
| Domain gap → hidden Cartosat/RISAT set | **Crit** | Holdout v0 accuracy > 15 pts below benchmark accuracy | Escalate GSD conditioning + aug; shift eval effort to holdout; consider SARLANG-heavy rebalance | Integration lead | End of each phase |
| Hidden SAR is single-pol / X-band | **Crit** | *(pre-mitigated by design — C2)* Validator stress suite fails on single-pol/X-band synthetic inputs | Fix before Phase 2 exit; these cases are in the catch-rate suite | Geospatial | Phase 1 exit |
| Serving/architecture mismatch | **Crit** | *(pre-mitigated — C1)* Phase 0 item 7 smoke test fails | Architecture decision revisited **before** data formatting | Backend | Phase 0 exit |
| **[v3.5] Compute overrun** | Med *(was High — A100 access reduces it)* | Burn-down shows > 35 h consumed before Phase 3 starts | Stop reruns; ablate away the third composite (C43); if still over, merge `rs_vqa`+`optsar_fusion` | Integration lead | **Weekly burn-down** |
| **[v3.5] A100 access falls through** | High | Access lost or unavailable at any point | Revert to the retained T4 fallback plan and triage ladder in §5.5. Everything needed is still documented | Integration lead | Continuous |
| **[v3.8] `change_map` below published F1/IoU** | Low–Med *(much reduced by C59/C60)* | F1 more than ~15 pts below TinyCD/ChangeFormer published figures | Raise self-generated pair count using A100 reserve (C42); lean on `change_stats` for quantitative answers; present the licence rationale rather than apologising | ML single-image | Phase 3 |
| **[v3.8] A restricted checkpoint reaches the deliverable** | High | Any weight in the ship bundle lacking a CLEAR row in CREDITS | Automated check: every shipped `.pt`/`.safetensors` must map to a CREDITS entry marked CLEAR. TinyCD/ChangeFormer checkpoints are blocklisted by filename | Integration lead | Phase 4, and CI |
| **[C61] DynamicEarthNet used before licence clears** | Med | Any DynamicEarthNet sample appears in a manifest | Blocked by the C45 provenance gate. Commercial Planet Fusion core — must clear verification first | Integration lead | Phase 0 |
| **[C60] HRSCD 2006 imagery redistributed** | Med | Any HRSCD imagery in a shipped artifact or public repo | Train-only. Never redistribute. CREDITS records the split-licence status | Geospatial | Continuous |
| **[v3.6] Self-generated change labels too weak** | Low *(C59 restores expert labels)* | `change_vqa` CDVQA-eval accuracy > 10 pts below the v3.5 projection | Increase pair count (Copernicus is unlimited); tighten index thresholds; lean on `change_stats` (D4) for quantitative question types | ML temporal | Phase 2 exit |
| **[v3.6] A dataset enters a manifest without provenance tracing** | High | Any manifest entry lacking a full chain in CREDITS | C45 is a hard gate — manifest build fails without it | Integration lead | Continuous |
| **[v3.5] Reserve burned on reruns instead of grounding data** | Med | Phase 2 ends with < 15 h left | Enforce C42 — reserve goes to `rs_ground_caption` sample count, the only lever on our weakest target | Integration lead | Phase 2 exit |
| Bhoonidhi access delayed/denied | High | No grant by end of Week 3 | Ship on holdout v0 permanently; stop chasing | Integration lead | Week 3 |
| Multi-sensor input *degrades* performance (EarthDial failure) | High | RGB-only ablation beats multi-view on any run | Investigate input representation before next run; never ship the degraded config | ML seats | Every run |
| CDVQA test2 collapse | High | test1−test2 AA gap > 2 on any run | Class-balanced resampling; re-check answer-prior leakage | ML temporal | Every run |
| `optsar_fusion` underdelivers | Med *(was High — D1 now ships first)* | Adapter ≤ D1 floor on holdout | Demo runs on D1; adapter becomes an ablation slide | ML temporal | Phase 3 |
| Router misroutes | Med | Hybrid routing < 90% on unambiguous 300-set cases | Extend rules; shrink LLM's jurisdiction | Backend | Week 1, then Phase 2 |
| Answer-format mismatch vs official scorers | Med | Our harness ≠ official script on any benchmark | P0 bug; formatter fix same day | Integration lead | Phase 1 |
| **[v3.1] Grounding underperforms on the hidden bbox portion** | High | Ref. detection mIoU < 40 on holdout | Spend the insurance GPU budget on more grounding data before anything else — it cannot be cut (C17) | ML single-image | Phase 2 exit |
| **[v3.1] Masks not scoreable as delivered** | Med | CI mask-conformance check fails, or a mask lands at tile rather than scene resolution | Fix before phase exit; masks are graded outputs (C18) | Geospatial | Every phase |
| **[v3.1] Parameters logged but not enforced** | Med | Any executed step lacks `parameter_check.passed` | Gate is on the never-cut list; block the phase exit | Backend | Phase 1 exit |
| **[v3.1] Benchmark shortfall vs RS-InternVL** | Med | Any §6.1 row below its target after its adapter lands | Expected on ref. detection — we train on ~2% of their corpus. Present the data-efficiency framing, not an apology; escalate only if binary VQA or captioning miss | Integration lead | Phase 2–3 |
| **[v3.1] Judging weights unpublished** | Med | PS placeholder still unfilled by end of Week 2 | Optimise for breadth, since normalised scoring rewards it; revisit if the table appears | Integration lead | Week 2 |
| ~~No permissive object-grounding source clears~~ **PARTIALLY REVERSED** | — | *(first sweep failed; second sweep succeeded)* | **Decided (C36/C39):** aircraft + buildings now trained on CC BY-SA / CC BY sources; remaining classes use `object_box_fallback`; scoped in plan, validator and deck | Integration lead | ✅ Closed |
| **[v3.4] OpenEarthMap-SAR pseudo-labels degrade training** | Med | Sub-metre region grounding worse than BEN.txt baseline | Train on the ~20-per-region manual subset only; demote pseudo-labels to weak supervision. Never use Bareland (0.02 IoU) | ML single-image | Phase 2 |
| **[v3.4] Umbra band assumption wrong** | Low | Umbra data turns out not to be X-band | X-band tuning falls back to SpaceNet 6 Capella alone; OpenEarthMap-SAR still serves single-pol and geographic diversity | Geospatial | Phase 0 |
| **[v3.3] Hidden set boxes are object-level, not region-level** | Med | Object queries appear in the holdout stress suite | `object_box_fallback` is the mitigation; measure its acc@0.25 and report honestly. Do not train on restricted data to chase this | ML single-image | Phase 3 |
| **[v3.3] LEVIR licence questioned by a judge** | Low | Any challenge to the change-detection data | Point at CREDITS: LEVIR used for `change_map` masks only, never for shipped weights (C33). If the CD checkpoint itself would be distributed, fall back to zero-shot backbone | Integration lead | Continuous |
| **[v3.3] SARLANG-1M subset licence unresolved** | Med | Any subset enters a manifest before its check | Per-subset review (C35). SpaceNet 6 portion is clean; DFC2023 / OpenEarthMap-SAR / SARDet-100K each need their own | Integration lead | Phase 0 |
| **[v3.2] SAR-only input unhandled by `rs_vqa`** | Med | SAR-only smoke-eval accuracy > 10 pts below optical | Raise modality-dropout rate; add SARLANG-1M weight | ML single-image | Phase 2 exit |
| **[v3.2] Staging overruns free-tier disk** | Med | Any staged dataset exceeds the per-dataset limit | Split further; drop to the priority-1 set and train `rs_vqa` first | Geospatial | Phase 0 |
| **[v3.2] GeoTIFF path only exercised on two sources** | Med | CI has no GeoTIFF fixture test | Synthetic fixtures — multi-band, 12-bit, odd CRS, nodata, single-pol SAR (C30) | Geospatial | Phase 1 exit |
| Full-scene inference OOM / slow | Med | Prep job > 5 min or query > 20 s p95 | GeoPixel top-k tighter; cut-list item 5 | Geospatial | Phase 3 |
| Venue has no internet | Med | *(assumed true)* Offline rehearsal fails | Headless mode is the fallback demo; rehearse twice | Integration lead | Phase 4 |
| **DOTA licence** (VRSBench images) | Low *(was "low but real" — C20 largely closes it)* | Any VRSBench sample appears in a training manifest | **Eval-only.** Never in training, never in the shipped weights, never in the public demo or repo. Automated check on the training manifest | Integration lead | Continuous |
| SNAP GPL-3.0 | Low | — | External process only, never linked | Geospatial | Continuous |
| Free-tier account action | Low *(was unlisted)* | Any ToS warning on any account | Confirm per-person-per-account policy holds; move affected adapter to insurance GPU | Integration lead | Immediate |

---

# PART 11 — Credits discipline

Maintain `../../CREDITS.md` from day one. Every model, dataset, library, and method with its citation and licence. Costs an hour total.

When an ISRO judge asks "what's yours and what isn't" — and they will — you answer by pointing at a file rather than improvising. That reads as rigour.

**The answer to have ready:**

> *"SAR preprocessing is ESA SNAP — it's the calibrated reference and reimplementing it would be worse, not better. The base model is Qwen. BigEarthNet.txt is our adaptation dataset as specified, and it's CDLA-Permissive, so the weights we hand you are unencumbered. VRSBench, RSVQA and CDVQA we use for evaluation — those are the ones you nominated. We kept VRSBench out of training entirely, and LEVIR, SECOND and everything derived from them we dropped entirely — including from change detection, once we realised a fine-tuned change model is itself a shipped weight. Our change pipeline trains on Sentinel pairs we generated ourselves over India, on MIT-licensed TorchGeo backbones. That's cleaner licensing *and* closer to your test data. We also excluded the TinyCD and ChangeFormer checkpoints, which are academic-use only. Object grounding we train for buildings and aircraft, from RarePlanes and SpaceNet 6 and OpenEarthMap-SAR — all CC BY or CC BY-SA. For the remaining object classes every dataset is Google Earth-derived, so we scoped those out and fall back to classical vision, labelling those boxes as deterministic proposals in the trace. What's ours is the four fine-tuned adapters, the geospatial validation layer, the agent and trace system, the calibrated confidence layer, and five places where we found a non-learned tool beats or safeguards a learned one: point priors for grounding, exact arithmetic for change ratios, band-availability routing when SWIR — or even NIR — is missing, physically-explained cross-modal disagreement, and a bimodality gate that tells us when our own thresholds can't be trusted. And every parameter in the trace was validated against the tool's manifest before it ran — you can check that in the file."*

**[v3.1] The benchmark question you will be asked, and the answer:**

> *"RS-InternVL reports higher referring-detection numbers than we do. They fine-tuned per task on the full BigEarthNet.txt corpus — roughly 347,000 pairs, two days on four H200s. We trained on about 2% of that on free-tier hardware. We match or approach them on captioning, binary VQA and MCQ, and we're honest about where the data gap shows. We also report the MCQ number twice — with and without the country, season and climate-zone tasks — because we dropped those deliberately: a model that has learned to guess European countries is learning the wrong thing for Indian imagery."*

---

# Appendix — Phase 0 checklist

- [ ] `trace_schema.json` frozen
- [ ] `task_enum.py` frozen
- [ ] **300-query routing eval set built** (same sitting as the enum)
- [ ] **Every ⚠ licence resolved; substitutes named where needed**
- [ ] **Submission/packaging spec obtained (or most-restrictive assumption documented)**
- [ ] `preprocessing.yaml` frozen and versioned — **incl. single-pol SAR stack, pol-dropout, per-band threshold + sanity tables (warning-only)**
- [ ] **Tool manifest format defined as an enforceable schema** — permitted names, ranges/enums, band prerequisites
- [ ] **Dummy trace carries a populated `graded` block and a passing `parameter_check`**
- [ ] **Team briefed: grounding and mask outputs are graded on the hidden set and are never cut**
- [x] **[v3.3] Object-grounding review COMPLETE — all candidates rejected; G3 scoped to region grounding (C32)**
- [ ] **[v3.3] G3 scoping written into all three places: plan ✅, validator rule, deck line**
- [ ] **[v3.3] SpaceNet 6 MSAW downloaded from AWS S3 and staged (C31)**
- [ ] **[v3.4] RarePlanes real split downloaded from AWS Open Data and staged (C36)**
- [ ] **[v3.4] OpenEarthMap-SAR downloaded from Zenodo and staged (C37); manual-label subset identified**
- [ ] **[v3.4] Umbra SAR band confirmed from the open-data catalogue (X-band?) (C37)**
- [x] **[v3.8] LICENCE VERIFICATION COMPLETE — all items resolved, records in CREDITS (C53–C58)**
- [ ] **[v3.8] AROSICS pinned to >=1.0.0 in requirements (C58)** — pre-1.0 was GPL-3.0
- [ ] **[v3.8] TinyCD / ChangeFormer checkpoints blocklisted in CI; CD backbone switched to TorchGeo MIT weights (C55)**
- [ ] **[v3.8] LEVIR / SECOND / LEVIR-MCI / QAG-360K purged from ALL manifests, including `change_map` (C56)**
- [ ] **[C59] SpaceNet 7 / MUDS downloaded from AWS Open Data and staged; footprint-differencing pipeline built**
- [ ] **[C60] HRSCD staged — 2012 from DataPort, 2006 direct from IGN; train-only flag set, split licence in CREDITS**
- [ ] **[C61] DynamicEarthNet licence sent to the verification job — NOT staged until it clears**
- [ ] **[C59] SpaceNet 7 tracking IDs wired into `change_stats` validation (counting / ratio question types)**
- [ ] **[v3.6] Provenance chain recorded in CREDITS for EVERY manifest dataset (C45)** — hard gate on manifest build
- [ ] **[v3.6] BEN.txt annotation-layer licence VERIFIED, not assumed (C50)** — do this first; it is the primary source
- [ ] **[v3.6] RSVQA-LR annotation layer, pretrained-weight licences, AROSICS licence family checked (C50)**
- [ ] **[v3.6] SARLANG-1M filtered to SpaceNet6 + OEM-SAR portions only; DFC2023 and SARDet-100K excluded (C47)**
- [ ] **[v3.6] OSCD + S1 extension downloaded (C49)**
- [ ] **[v3.6] Self-generated Sentinel change-pair pipeline built over Indian AOIs (C46)** — reuses holdout v0 machinery
- [ ] **[v3.4] xView3 remains excluded; recorded in CREDITS with the reason (C38)**
- [ ] **[v3.3] SARLANG-1M per-subset licence check (C35)** — SpaceNet6 / DFC2023 / OpenEarthMap-SAR / SARDet-100K
- [ ] **[v3.3] ../../CREDITS.md records the C33 LEVIR rule explicitly** — `change_map` masks only, never shipped weights
- [ ] **[v3.2] RSVQA-HR licence verified and staged**
- [ ] **[v3.2] Staging split into 3–4 datasets in priority order; priority-1 set live**
- [ ] **[v3.2] Synthetic GeoTIFF fixtures in CI** — multi-band, 12-bit, unusual CRS, nodata regions, single-pol SAR
- [ ] **[v3.2] Asked ISRO/organisers whether the hidden set contains bi-temporal pairs at all**
- [ ] Repo + Docker Compose + CI running
- [ ] Dummy tool emits a valid trace end-to-end
- [ ] **Serving smoke test passed: base + 2 adapters hot-swapping under vLLM on the *named* demo machine**
- [ ] **Demo machine named and specced**
- [ ] **Patch-ID manifest built; manifest-only downloads started; private Kaggle Dataset published; full-corpus downloads banned**
- [ ] Zero-shot baselines recorded (both candidate bases, 3 benchmarks)
- [ ] **D2 zero-shot ablation run; D2 framing decided**
- [ ] **Router zero-shot accuracy measured (rules / LLM / hybrid)**
- [ ] Base model committed
- [ ] **[v3.5] 200-step timing measured on the A100; §5.5 budget table re-derived from real numbers (C41)**
- [ ] **[v3.5] bf16 + FlashAttention-2 confirmed working; QLoRA dropped; batch size tuned to saturation**
- [ ] **[v3.5] Two- vs three-composite ablation run before committing to C22 (C43)**
- [ ] **[v3.5] Hour burn-down tracker live; integration lead owns weekly update (C44)**
- [ ] *(fallback only)* 200-step timing on T4; triage-ladder rung applied within 48 h
- [ ] **Paid-GPU insurance budget decided; institute GPU access request sent**
- [ ] Bhoonidhi access requested (Day 1) **and India holdout v0 assembly started from open data**
- [ ] W&B project live; own-session checkpoint resume tested
- [ ] **Per-person-per-account training policy acknowledged by the team**
- [ ] `../../CREDITS.md` started
