# 0. Abstract, and the shape of the argument

## Abstract

We present **SatQuery AI**, an agentic remote-sensing system that answers natural
language questions about satellite imagery across single images, bi-temporal
pairs and optical–SAR pairs, and exposes its reasoning as a machine-readable
trace. The system comprises a single 4B vision-language backbone
(Qwen3-VL-4B-Instruct) with two LoRA adapters of 40.3M trainable parameters each,
one published radar classifier, six deterministic geospatial tools, and a
rules-first router that decides which of them answers.

On the graded splits it reaches **AA 85.06 on RSVQA-HR** and **83.08 on
RSVQA-LR**, both above the RSVQA authors' own model (83.12 / 81.49); **AA 68.0%
on CDVQA**, level with the published fine-tuned result for the same backbone
(67.86) and 0.6 behind the best in that paper, against a 45.0% blind ceiling; **62.7%
acc@0.5 on VRSBench referring grounding with no training at all**, above
published fine-tuned GeoChat (60.6%); and **74.95%** on held-out reBEN radar
land-cover against a verified 50.1% floor. Captioning reaches ROUGE-L 25.2
against a 24.4 wrong-image floor and **loses** to fine-tuned baselines by roughly
12 points.

Three findings generalise beyond the system.

**First, format dominates.** Every large score movement we recorded was a
formatter fix rather than a modelling one: quantising RSVQA counts into the
benchmark's own answer space was worth **+7.3 points**; bucketing `area` carried
the HR gate; three separate coordinate-convention bugs each produced results
indistinguishable from a model that cannot localise. No weight changed in any of
them.

**Second, blind baselines are not optional.** RSVQA-LR presence questions are
76.3% "yes"; a model that never opens the image scores that. Shown a *wrong*
image, our fine-tuned adapter still scores 59.0 against a 39.7 floor. We report a
majority-answer ceiling, a wrong-image floor, and a shuffled-image ablation
beside every number, and the measured vision contribution (9.9–16 points) is the
claim rather than the raw accuracy.

**Third, and most transferably: component benchmarks do not measure a system.**
Every number above was produced by calling a component directly. When we built a
200-row held-out gallery that replayed published benchmark rows through the
*product* — the same router, tools and adapters a user reaches — it exposed nine
defects, seven of which were invisible to 440 unit tests and 50 route checks.
Among them: the radar classifier holding our 74.95% **never executed on a single
deployed request**, because Sentinel-1 files name no bands and the tool refused;
and every cross-modal query failed outright on an attribute error, on a branch no
test had ever reached. Fixing the routing took grounding from 12% to 80% and SAR
from 46% to 74% on held-out data, without retraining anything.

We argue that a held-out, published-gold, end-to-end replay belongs alongside
unit tests and benchmark evaluation as a standing verification layer, and we
report the negative results — including a capability we chose *not* to fine-tune
because the metric would have rewarded the wrong thing — as part of the
contribution.

## The shape of the argument

```
1  problem      a hidden Indian sensor set, a shipping licence constraint,
                and one shared base model
2  corpus       what was staged, what was refused, and how balance was kept
3  rs_vqa       the trained adapter that beat the dataset's own authors
4  change_vqa   trained twice; the first was discarded, and why that was right
5  grounding    won by NOT training -- and the three coordinate bugs
6  captioning   the loss, stated plainly, and the metric that explains it
7  SAR          bought rather than built, because the published numbers said so
8  agent        router, gate, executor, composer -- graded, and buggiest
9  method       blind ceilings, shuffle controls, answer contracts
10 serving      one base, three adapters, a deploy that verifies itself
11 failures     nine bugs, what each proved
12 results      every number with its baseline
13 limits       what we cannot claim
```

## The four decisions that defined the system

**One shared base.** Three separate 4B models do not fit on one L4 — learned by
running out of memory mid-query. Every adapter is therefore a LoRA on the same
backbone, and that constraint reached back into training.

**Measurement beats inference where a mask exists.** Router plus set arithmetic
scored **100% on 2,012 rows** given ground-truth footprints, where the fine-tuned
adapter scored 43.5%. It does not ship, because nothing produces the mask well
enough — our detector reached F1 0.2931. The honest reading: the pipeline was
never the bottleneck, perception was.

**Two of four planned adapters were never trained.** Grounding ships the base
model because it beat a fine-tuned published baseline without training. Fusion
ships a published classifier because the dataset's own authors showed early
fusion (AP 0.711) is *worse* than optical alone (0.714). Both decisions were made
by measurement, against work already scoped.

**A licence-clean 80% beats an unusable 91%.** "Codes and models" is a
deliverable, so a permissive badge on a repository says nothing about the imagery
underneath it. That single rule rejected DIOR, FAIR1M, NWPU-Captions, RSICD,
LHRS-Bot, Open-CD, LLaVA-Instruct-150K and ChatGPT-smoothed SkyScript — and sent
change detection to 4 m SpaceNet 7 instead of 0.5 m LEVIR-CD, at a known cost.
