# Future changes — deferred, not forgotten

Things we decided **not** to do now, with enough context to pick each up cold.
Nothing here is a bug in flight; everything here is a considered deferral.

Ordered by value per hour of work, not by when it came up.

---

## 1. Captioning — multiple length modes

**Now:** one length. `STRONG_PROMPT` asks for *"about 47 words across 3
sentences"*, because that is VRSBench's median.

**Wanted:** the caller picks. Concise ~50 words, detailed ~100.

**Why it matters more than it looks.** The graded hidden set is ISRO's, and
**we do not know their caption length.** A model locked to 47 words is locked to
VRSBench's convention; a length-conditioned one adapts at inference. That is the
same reasoning that removed the provenance clause — do not hard-code another
lab's convention into a system graded by someone else.

**Already half-built.** `scripts/gen_captions.py` emits both modes and the
corpus carries them; only the served prompt is fixed at one length. The design
was settled earlier: a *request*, not a contract, and deliberately **no 150-word
mode** — our annotations cannot fill it without inventing.

**Shape of the work.** A `length` enum on `configs/tools/rs_ground_caption.yaml`
(the parameter gate then validates it, as it does `mode`), the router passing it,
and `served_prompt` swapping the word-count clause. Perhaps an hour.

**Measure it.** `scripts/eval_captioning.py` at both lengths against the blind
floor. Length is the largest lever measured on this task — **130 → 47 words moved
ROUGE-L from 0.195 to 0.252** — so a second mode is not cosmetic, and a longer
caption on a sparse scene will score worse, not better.

---

## 2. Grounding — return multiple objects when asked

**Now:** one box. `PRECISE_PROMPT` asks for the single best match, and
`_parse_boxes` keeps the first valid box.

**Wanted:** *"find **all** the aircraft"* returns every match; *"locate the
aircraft"* still returns one.

**Why it is not free.** **62.7% acc@0.5 was measured single-box**, on VRSBench
referring, where each item has exactly one ground-truth box. Multi-box changes
the task — the right metric becomes precision/recall or mAP, not acc@0.5 — so
the headline number does not transfer and would need re-measuring against a
different reference set.

**The router already knows.** `_GROUNDING_TERMS` contains `find all`, `find any`,
`find every` as distinct entries from `find the`. The plural intent is captured
at routing and then discarded downstream.

**Watch for:** **58% of VRSBench targets are non-unique**, and non-unique is
already the weak slice (55.4% against 64.7% unique). Returning several boxes may
*help* there — a recall-oriented answer suits a set query — which makes this
worth measuring rather than assuming.

**Shape of the work.** A `max_boxes` parameter on the manifest, a prompt variant
that permits a list, `_parse_boxes` keeping all valid boxes, and the box asset
already handles a list. The measurement is the real cost, not the code.

---

## 3. Evidence scoping — layers accumulate across queries

**Now:** every query's masks and boxes stay in the layers panel, so labels stack
up (`8 CANDIDATES`, `9 BOX PROPOSALS`, …) from questions asked minutes ago.

**Wanted:** evidence scoped to the selected query, with history reachable
deliberately.

Cosmetic, but it makes a demo look confused, and it was the thing that made the
box misalignment hard to read.

---

## 4. Confidence label wording

The header shows e.g. **"27% HEURISTIC"** beside an answer, which reads as
*"27% sure there are trees"*. It is not: it is a **pipeline-health heuristic**
over agreement IoU, threshold-fallback fraction, router path and warning count.
`confidence_basis: "heuristic"` already declares this honestly, and the system
page says plainly that *"no calibrator is fitted yet… an ordering, not a
probability"*.

So this is wording, not computation. Either label it as pipeline confidence, or
fit the 4.7.3 calibrator and earn a real number.

---

## 5. Real imagery for SAR and change — **CLOSED**

This was called "the largest honesty gap in the demo": both capabilities were
measured, and neither had real imagery you could put through the UI.

**The testing-corpus gallery closed it.** 50 reBEN patches (S1 radar, and S2
optical beside it on the fused rows) and 50 real CDVQA Val pairs, all held-out, all with published gold answers, all replayed through the
live API — SAR **37/50** against a benchmarked 74.95%, change **47/50**. See
[08-testing-corpora.md](08-testing-corpora.md).

Doing it also proved the gap was not cosmetic. Running real radar through the
product surfaced that **BIFOLD had never executed on a single real patch**:
Sentinel-1 names no bands, so the tool refused with `needs band 'VH'` and the
VLM answered in its place. A synthetic fixture would not have found that,
because the synthetic files carried band descriptions.

What remains open is only the third bullet below — a genuinely cross-modal
*co-registered* source. The reBEN fused rows pair optical and radar of the same
patch, which exercises the path, but they are not the sub-metre structural match
to Cartosat + RISAT that SpaceNet 6 would be.

Original options, for the one still outstanding:

- **Sentinel-1 GRD tiles** — C-band VV/VH in dB, exactly BIFOLD's training
  domain. Makes 74.95% testable rather than theoretical.
- **A CDVQA pair** — 0.5 m bi-temporal, the corpus `change_vqa` was
  actually trained on. Note the imagery carries no published licence, which was
  accepted for training but is worth restating for a demo.
- **SpaceNet 6 `SAR-Intensity`** — 0.5 m Capella X-band co-registered with Maxar
  optical. The structural match to the ISRO set (Cartosat optical + RISAT SAR),
  and the only way to exercise the cross-modal route on real data. A subset of
  41 GB; the optical half is already staged.

The first two have landed. Say plainly that the cross-modal route is exercised on
paired-but-not-co-registered reBEN rather than on a Cartosat/RISAT-shaped source.

**And that the optical half of a reBEN pair is a texture proxy, not a spectral
index.** reBEN's S2 patches name no bands and their channel order could not be
resolved -- neither candidate fits the published per-band statistics, and the
optical classifier is at chance at every order, so the one experiment that could
decide it carries no signal. No band map was written; guessing one would put a
plausible-looking NDVI on the screen with nothing able to tell it was wrong.
Reasoning in [08-testing-corpora.md](08-testing-corpora.md). SpaceNet 6's
`SAR-Intensity` is what closes this properly.

---

## 6. In-memory state on the deployment

Scenes, bundles and queries live in process memory, so a container scaledown
(5 min idle) loses them. `max_containers=1` fixed the *random* losses — requests
were round-robining across replicas that shared no state — but an idle gap still
clears everything.

Fix is to move those three dicts onto the Volume or a real store. Only worth it
if more than one person uses the demo at once, which is also when
`max_containers=1` stops being right.

---

## 7. Cold start ~45 s

First question per container pays the 4B weight load; everything after is
sub-second. `min_containers=1` removes it and bills continuously. A demo-day
decision, not an engineering one.

---

## 8. Corpus and measurement debts

- **`smallest_change` at 29.3%** — below the published 32–37% band. Cause is
  understood (flat gold distribution, 22.4% blind ceiling, genuine class
  confusion rather than a formatter bug), no fix attempted.
- **The shuffle control has never been run on the CDVQA adapter.** It is the
  decisive check for how much of 68.0% is vision — the SpaceNet 7 adapter
  measured a 9.9-point contribution. One eval run; the harness exists.
- **CDVQA Test1/Test2 are staged and never scored.** Those are the graded
  splits; 55.3% baseline and 68.6% SOTA are Test1 numbers, so our Val figure is
  not directly comparable. `cdvqa_test_balanced.jsonl` (500/type) is already on
  the Volume.
- **Official scorers still not wrapped** (C8) — RSVQA, VRSBench and CDVQA are all
  scored with the in-repo comparator. Every report carries the caveat. Note
  CDVQA ships **no evaluation code at all**, so there is nothing to defer to
  there; our AA definition already matches the paper's.
- **The caption fine-tune test was never run.** Whether training on our corpus
  clears the 0.252 prompted baseline is still unknown, and it is a few hundred
  steps on a slice.

---

## 9. Deferred by decision, listed so nobody reopens them

| | why it is closed |
|---|---|
| `change_map` / Siamese CD | F1 0.2931, and CDVQA asks about land cover while the detector finds buildings. Wrong task, not just weak weights. `TEAM_CONTEXT.md` §5. |
| `optsar_fusion` adapter | Cross-modal is graded only on the hidden set, so any training is unmeasurable. BIFOLD + D1 covers the mandatory bullet. |
| `object_box_fallback` in grounding | The base model handles out-of-vocabulary targets and abstains correctly. Blob proposals beside a learned box are worse than nothing. |
| Fine-tuning on our caption corpus | 16-word median against a 47-word target; training would break a length the prompted base already gets right. |
| A second base model | C1 commits to one shared base so several LoRAs serve from one set of weights. |
