# 7. SAR and cross-modal fusion — the capability we bought instead of building

## 7.1 Result

**reBEN held-out validation, 6,000 rows, binary land-cover questions.**

| | accuracy |
|---|---|
| **`resnet50-s1` on radar (BIFOLD)** | **74.95%** |
| majority-answer floor | 50.1% |
| the same model with the band order reversed | **49.9%** — chance |

Threshold 0.5, which is the out-of-the-box operating point. A sweep found a
better threshold but it was chosen on the same split it was scored on, so 0.5 is
what is reported.

**No adapter was trained for this.** `optsar_fusion` was a planned fourth LoRA
and it does not exist.

## 7.2 Why no fusion model — the published numbers close the question

BIFOLD is the reBEN authors' own baseline set, published alongside the dataset
(Clasen et al., IGARSS 2025, arXiv 2407.03653). Their average precision:

```
optical  0.714
radar    0.628
fused    0.711     <- worse than optical alone
```

**Early fusion adds nothing.** A trained cross-modal adapter would have been
attempting to beat a fusion result that the dataset's own authors could not make
better than a single modality. There was no GPU spend left to justify on that
path, and the project's existing decision-level choice (D1, §4.7) was confirmed
rather than replaced.

So the SAR story ships as: **`resnet50-s1` as a registered tool, plus D1 as the
deterministic floor.** That covers the one mandatory cross-modal bullet with two
measured components and no unmeasured ones.

## 7.3 Radar over optical, and the number that decided it

The obvious design is to run whichever classifier matches the input. The
measurement says otherwise. On **4,000 held-out cross-modal rows, both models
answering the same questions**:

```
radar classifier    0.7525
optical classifier  0.4968      <- chance
agreement rate      0.6542
when they disagree  radar right 86.98%   optical right 13.02%
```

The optical arm is **at chance on this data**, and when the two disagree radar is
right seven times out of eight. So `lulc_classifier` is scoped to radar and the
manifest says `required_modalities: [sar]`. Optical land cover is served by
`spectral_index` and the VLM instead — both measured, both working.

That decision was recorded in the manifest, in the router, and in the plan, and
**the tool still contained a live optical fallback** until it was removed very
late. It was unreachable behind two guards, but unreachable is not safe: relax
either and the system serves a chance-level classifier behind a warning nobody
reads. It now refuses, and three layers of test pin it ([11](11-failure-atlas.md)).

## 7.4 The band order — measured, and counter-intuitive

Sentinel-1 GeoTIFFs carry no band descriptions. The order had to be established
empirically, and the intuitive answer is wrong:

```
s1 = [VH, VV]    accuracy 0.817     <- correct
s1 = [VV, VH]    accuracy 0.499     <- chance
```

**Band 1 is VH.** A wrong-order control scoring *exactly* chance is the cleanest
possible confirmation, and it is why the order is pinned in
`configs/preprocessing.yaml` with the numbers beside it — this is precisely the
kind of value a later reader "corrects" on instinct.

The same problem on the S2 side could **not** be resolved, and that is documented
as unresolvable rather than guessed ([13](13-limitations.md)): neither candidate
order fits reBEN's published per-band statistics, and the optical model is at
chance at every order, so the one experiment that could decide it carries no
signal.

## 7.5 The corpus, and three iterations to an honest floor

`optsar_fusion.jsonl`: **85,096 rows** from 13,683 verified reBEN pairs, across
three arms (optical, radar, fused).

The floor was driven to exactly 50% over three passes:

```
60.6%  ->  53.9%  ->  50.0%
```

A benchmark where the majority answer scores 60% makes a 75% result look far
better than it is. Getting every arm to an exact 50.0% majority-answer floor
means the 74.95% is 25 points of real signal, not 15.

One regression on the way is worth recording: an intermediate fix took forest
from 77.5% to 56% of the radar arm and dropped the floor from ~60.6% to 53.9% —
a good change whose effect on the metric had to be measured rather than assumed.

## 7.6 D1 — decision-level fusion

When an optical mask and a SAR mask both exist for the same target, they are
reconciled rather than averaged:

```
IoU >= 0.60   consistent  ->  union of both, confidence raised
IoU <  0.30   conflict    ┐
0.30 - 0.60   partial     ┘ ->  consult a five-rule physical table
```

The table, evaluated in order, first match wins:

| cause | winner | physical justification |
|---|---|---|
| cloud over water | **SAR** | cloud is opaque to optical, transparent to C- and X-band radar |
| wet or smooth soil | optical | specular soil mimics water on radar; the spectrum separates them |
| radar shadow | optical | the dark return follows terrain facing away — geometry, not surface |
| wind-roughened water | optical | wind raises backscatter; the water is still water |
| dry smooth sand | optical | specular sand reads as dark as water on radar |

**SAR wins exactly one case** — the one where optical is physically blind.
Optical wins four, all of them known radar failure modes.

**If no rule matches, neither wins.** The system returns the *intersection* — the
agreed extent — with confidence multiplied by 0.6 and a warning saying no rule in
the table explains the disagreement. Choosing a modality silently is the failure
this design exists to prevent.

Disagreement always costs confidence: the winner's own confidence × 0.6, never
the raw value. A single-modality answer costs less but still costs — × 0.9, for
lack of corroboration.

## 7.7 D2 and D3

**D2 — centroid prior.** A mask's centroid can be offered to the grounding model
as a point prior. The evaluation concluded it should be *"a trace artifact and
not injected into prompts"* at this stage, so it is recorded as evidence rather
than fed to the model.

**D3 — band gating.** `BandInventory` decides which tools may enter a plan at
all. A target whose index needs SWIR on a source with no SWIR does not get a
`spectral_index` step that would fail the parameter gate — it gets the SAR path,
or `texture_seg`, and the substitution is recorded as a routing note. Planning a
step the gate would refuse yields a failed trace instead of an answer, which is
why the gate and the planner agree by construction rather than by convention.

## 7.8 What was rejected

| option | why |
|---|---|
| **a trained `optsar_fusion` adapter** | published fusion AP (0.711) is *below* optical alone (0.714); cross-modal is graded only on the hidden set, so any training would be unmeasurable |
| **SpaceNet 6 for training** | best labels available (48k surveyed footprints, quad-pol) but **41 GB** and optical-only staged; deferred, still the right long-term answer |
| **SSL4EO / `B2_vits16_mae` fine-tune** | a real candidate *after* the deadline — fine-tune on our reBEN S1 and see if it beats `resnet50-s1` |
| **SARLO captions** | captions produced by a VLM reading the wrong sensor |
| **OEM-SAR pseudo-labels** | in places simply wrong — Bareland IoU 0.02 |
| **SARChat** | non-commercial |

## 7.9 The honest limit

The 74.95% is measured on **reBEN held-out val** with **our binary questions**
against **our verified 50% floor**. It is not the same number as the authors'
published 0.628 macro AP on their own metric — different task, different
scoring. Both are stated; neither is used to stand in for the other.

And until very recently, **none of it ran in the product**: Sentinel-1 names no
bands, so the tool refused with `needs band 'VH'` and the VLM answered in its
place — the 0.4968 component speaking over the 0.7525 one. That is
[11](11-failure-atlas.md), and it is the strongest argument in this project for
testing components through the router rather than only in isolation.
