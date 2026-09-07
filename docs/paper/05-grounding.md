# 5. Grounding — the capability we won by *not* training

## 5.1 Result

**VRSBench referring, acc@0.5** (the predicted box must overlap the truth by at
least 50% IoU), random unstratified sample, matching the published protocol:

| system | acc@0.5 | trained on VRSBench? |
|---|---|---|
| **ours — base Qwen3-VL-4B + `PRECISE_PROMPT`** | **62.7%** | **no** |
| GeoChat (published) | 60.6% | yes, fine-tuned |
| base Qwen, plain prompting | 61.3% stratified / 60.7% random | no |
| zero-shot Qwen, first attempt | 15.60 mIoU | no |

Mean IoU **0.5506**. Instruction-following on box format: **100%**, unparsable
**0.0%**.

**We beat a published fine-tuned baseline with no training at all**, on the same
benchmark, using one Apache-2.0 model and a better-written prompt. That is the
single most surprising result in the project, and it was reached by trying to
train and discovering training was the wrong lever.

## 5.2 Why we did not train — the corpus could not have worked

The plan called for a trained grounding adapter. Two measurements killed it
before a single step was run.

**Our grounding corpus covers 9% of VRSBench's classes.** VRSBench referring is
**28% `vehicle`**, plus harbours, bridges, storage tanks and tennis courts —
precisely the classes ruled out at C39 because *"every other public source is
GoogleEarth-derived and cannot ship"*. We had RarePlanes aircraft and SpaceNet 6
buildings. Training on those would have taught the model nine percent of the
benchmark.

**BEN boxes are the wrong scale entirely.** BigEarthNet grounding rows are 10 m
*region* referring against a 0.3 m *object* referring benchmark. The median BEN
box covers **20% of the image**, and **zero of 20,000 are small objects**.

So the corpus was not merely thin, it was categorically mismatched. Its job
changed: it stopped being "the thing that makes grounding work" and became
"something to test against 62.7%". That test — does fine-tuning beat the
prompted base — was never run, and remains open.

## 5.3 The detector pipeline, measured and rejected

A text-prompted detector needs no training data for the missing classes at all,
which made Grounding DINO the obvious alternative. It was measured properly
rather than adopted on reputation.

```
Grounding DINO ceiling, 9 candidates      65.3%
Grounding DINO ceiling, 26 candidates     70.3%   (later, honestly: 63.7%)
after correct reweighting                 62.5%   (down from 65.3%)
a realistic selector at 70–80% of ceiling ~45–51% actual
dino_vlm, actually built and scored       34.7%
```

Every one of those is a **ceiling** — they assume a selection stage that picks
the right candidate out of the proposals. The one time that stage was actually
built and scored end to end, it managed **34.7% against `qwen_precise`'s
62.7%**.

The decision writes itself: a pipeline whose *ceiling* is 63.7% and whose
*measured* form is 34.7% loses to a single prompted model at 62.7%.

One honest note on the way there: Text2Seg's advertised *"31–225% over vanilla
SAM"* is improvement over a baseline that does essentially nothing, and tells
you almost nothing in absolute terms. Beating that by 225% is entirely
compatible with a low absolute score. The rule adopted was to demand an absolute
number on our own graded benchmark before integrating anything.

## 5.4 `PRECISE_PROMPT` — what a prompt is actually doing here

The prompt is not decoration; it encodes measured properties of the annotations:

- the **smallest enclosing rectangle**, not a loose bound;
- **full diagonal extent** for elongated objects;
- **no shadows** included;
- targets are **usually under 3% of the image**;
- the **0–1000 coordinate grid**, which is Qwen's native convention;
- the exact JSON shape `[{"bbox_2d": [x1, y1, x2, y2]}]` the parser expects.

The gain over plain prompting is **62.7% against 61.3%** — modest, and reported
as modest. The larger contribution of the prompt work is that it made the output
*parseable at all*: instruction-following went to 100% with 0% unparsable.

An 11-variant sweep sits behind this; the full table is in
[`docs/segments/02-grounding.md`](../segments/02-grounding.md).

One variant was rejected for a reason worth stating: `qwen_sam` wins on acc@0.5
but is **worst of the qwen family at actually finding the object (83.3%)**,
because SAM's trimming can pull the box centre off target. It helps at threshold
0.5 and hurts at 0.7. **The ISRO set's scoring threshold is undisclosed**, so a
change that helps at one threshold and hurts at another is a bet on something we
do not know. Declined.

## 5.5 Three coordinate bugs, each of which looked like a broken model

This is the clearest example in the project of format masquerading as capability.

| bug | symptom | cause |
|---|---|---|
| **divide by 512** | 75% "unparsable" | anything above 512 clamped to 1.0 and collapsed into a degenerate box |
| **comma as decimal separator** | *every* Qwen-format box rejected | `_NUMBER` read `(560,190)` as the single value `560.190` |
| **`bbox_2d` misread** | boxes shifted | the `2` in the key name parsed as a coordinate |

None of these was a modelling failure. All three produced numbers that looked
exactly like a model that could not localise. They are why the served path now
parses boxes with **the same formatter the benchmark scores with** — so what a
user sees is what acc@0.5 measured.

## 5.6 A second metric, kept permanently

acc@0.5 is a cliff: a box that finds the right object but is slightly loose
scores zero, identically to a box on the wrong side of the image. Breaking the
150-row run down:

```
94  correct at 0.5
33  right object, loose box
23  genuinely lost
```

So **84.7% found the object**, against 62.7% acc@0.5. Both are reported, because
they answer different questions and the ISRO threshold is unknown. If the hidden
set is scored leniently, or by mask overlap, **84.7% is closer to what we would
see than 62.7%**.

Mean IoU 0.5506 sits barely above the 0.5 line, which says the same thing from
the other direction: a lot of answers are hovering right at the boundary.

## 5.7 The abstention property, found by accident

`PRECISE_PROMPT` asks the model to *find this object*. VRSBench only ever asks
about objects that **are** present — the benchmark has no abstention case. So a
user asking for a green car in an image with no green car is off-distribution
for the measured number.

Tested directly: the base model **abstains correctly**, replying that there is no
such object rather than returning a box. That behaviour is not covered by 62.7%
and is reported separately.

This is also why `object_box_fallback` was removed from the served grounding
plan. It offered eight deterministic blob proposals beside one learned box, on
classes the base model handles open-vocabulary. Its gate was written when
`rs_ground_caption` was going to be a *trained* adapter with a fixed vocabulary;
the shipped component is the base model, which has no such limit. Sending the
benchmark's single most common class — `vehicle`, 28% of it — to a texture-blob
proposer instead of the model measured at 62.7% is a straightforward downgrade.

## 5.8 Serving parity

The number was measured by feeding `PRECISE_PROMPT` **the whole referring
sentence** (`eval_grounding_pipelines.py`: `phrase = row["question"].rstrip(". ")`).
The served path originally passed an extracted noun instead, which throws away
the half of the sentence that disambiguates one aircraft from another in the
same frame. Fixed, with the interrogative case still passing the noun, since
*"Find this object in the satellite image: Where is the white ship?"* is not a
phrase and is not what was measured either.

Separately, the router did not recognise VRSBench referring rows as grounding at
all — they are *statements*, not questions — and sent them to the VQA adapter,
which answered "yes." That is [11](11-failure-atlas.md), and it took grounding
from 12% to 80% on held-out rows when fixed.

## 5.9 Open

- **Does fine-tuning on our corpus beat 62.7%?** Never run. The corpus covers 9%
  of the classes, so the expectation is no, but it is an expectation.
- **SAR ship grounding** is explicitly in scope (C53) with a graded acc@0.5, and
  it rests on **LS-SSDD** — which `CREDITS.md` records as NOT STAGED after an
  earlier finding of *"research/teaching only"*, with imagery behind a portal
  whose terms nobody has read. That capability is currently unbuildable on clean
  data.
