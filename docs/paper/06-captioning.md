# 6. Captioning — the one that plateaued, and the honest reason

Every other capability in this project has a favourable comparison. This one
does not, and the document says so in its first paragraph rather than its last.

## 6.1 Result

**VRSBench captioning, 300 rows.**

| system | ROUGE-L |
|---|---|
| fine-tuned LLaVA-1.5 (published) | **36.9** |
| Mini-Gemini (published) | 36.8 |
| GeoChat (published, fine-tuned) | 35.2 |
| GPT-4V (published) | 30.1 |
| **ours — `STRONG_PROMPT` on base Qwen3-VL-4B** | **25.2** |
| ours, two-pass (inventory then narrate) | 25.8 |
| ours, 8B base (partial run, 50/300) | 26.8 |
| ours, plain prompting | 19.5 |
| **wrong-image blind floor** | **24.4** |
| zero-shot GeoChat | 13.2 |

**We lose to published fine-tuned models by roughly 12 ROUGE-L points.** That is
the headline and it is not softened anywhere in this project's reporting.

## 6.2 The blind floor is the whole story

Describing **the wrong image entirely** scores **ROUGE-L 0.244** on this
benchmark. Generic aerial vocabulary — *image, area, buildings, road, visible,
surrounded* — overlaps with almost any reference.

So:

```
plain prompting   0.195   BELOW the floor
STRONG_PROMPT     0.252   clears the floor by 0.008
```

**Our best result beats "write about a different photograph" by eight
thousandths.** Read plainly, ROUGE-L on this benchmark is measuring *register*,
not sight.

That the model can see is not in doubt — the same weights localise objects at
**84.7%** on the same imagery. The metric simply does not reward it.

## 6.3 Two metric findings worth keeping

**CIDEr separates where ROUGE-L cannot.** CIDEr weights informative words by
inverse document frequency, so generic aerial vocabulary is discounted. Our
CIDEr sits at roughly **four times its floor** while ROUGE-L barely moves. When
a caption scores 0.15 on ROUGE-L you cannot tell whether it is a bad caption or
a good one worded differently; CIDEr at least tries.

**CIDEr-D returns 0.0 for an identical caption when there is one reference** —
with a single document, every n-gram has IDF `log(1/1) = 0`. That is a property
of the metric, not a bug in the implementation, and it was verified by scoring
perfect predictions (1.00 / 1.00 / 10.0) against wrong-image ones
(0.05 / 0.24 / 0.04).

## 6.4 Length is the largest lever measured

```
130 words  ->  ROUGE-L 0.195
 47 words  ->  ROUGE-L 0.252
```

Almost the entire gain from prompt engineering came from **writing fewer
words**. VRSBench's median reference is 47 words across 3 sentences; the base
model naturally writes 130. Plain prompting scored near the floor *"almost
entirely because it writes 130 words where 48 are wanted"*.

This is why `STRONG_PROMPT` asks for a word count at all, and why a
length-conditioned mode is the top item in
[`07-future-changes.md`](../segments/07-future-changes.md): the graded ISRO set
is not VRSBench, and we do not know its caption length. A model locked to 47
words is locked to one benchmark's convention.

## 6.5 Why we did not fine-tune

The published table settles the architecture question — every top scorer is a
fine-tuned modern VLM — so fine-tuning is clearly the lever that works. We
declined anyway, for a reason specific to our corpus:

**Our caption corpus has a 16-word median against a 47-word target.** Training on
it would teach the model to write *shorter* than the prompted baseline already
does, breaking the one property that got us over the floor.

Worse, our captions are template-generated. **A model trained on templates learns
the template rather than the image — and n-gram metrics reward exactly that**,
because BLEU and ROUGE score shared words. We would very likely have produced a
higher number and a worse model, with no way to tell from the metric.

So captioning ships as the prompted baseline, and the fine-tune test —
does training on our corpus clear 0.252? — is recorded as **never run**.

## 6.6 What the metric punishes, concretely

The single most useful artefact from this work is a high-scoring model output
that scores badly:

> The model counted the turbines correctly, described the terrain correctly, and
> added positions the reference omits — and scored **0.14** for saying *wind
> turbines* instead of *windmills*, and for not saying *GoogleEarth*.

**86% of VRSBench references open with a sensor phrase** — *"This aerial image
from GoogleEarth shows…"*. Matching that opener is worth real ROUGE-L.

We deliberately removed it. `STRONG_PROMPT` used to instruct the model to open
with *"The image, sourced from GoogleEarth,"* and that clause was deleted by
project decision: the graded set is ISRO's, the imagery is Cartosat, and
hard-coding another lab's provenance convention into a system graded by someone
else is teaching the model to lie about where its pixels came from.

**That decision costs ROUGE-L and we took it anyway.** When the held-out gallery
was scored later, captioning items clustered just under the floor — `0.2045`,
`0.2000`, `0.1961` against `0.2218` — and the missing opener is worth roughly
that margin. It is a measured cost of an honesty decision, recorded as such.

## 6.7 Rejected

| option | why |
|---|---|
| **Gemini-generated captions** | Google's API terms restrict using output to develop competing models — the same defect that disqualified LLaVA-Instruct-150K and ChatGPT-smoothed SkyScript. And we never measured that Gemini is *good* at 0.3 m aerial imagery; only that Qwen is not |
| **an 8B base** | partial run reached 0.268 against 4B's 0.252 — **+0.016 for double the weights**, and it voids the grounding result, which was measured on the 4B |
| **fine-tuning on our corpus** | 16-word median against a 47-word target; would teach the template |
| **two-pass inventory-then-narrate** | best ROUGE-L of four prompts at 0.258, but costs double inference and *drops* CIDEr (0.092 vs 0.135) and BLEU-1 — better on the metric that measures register, worse on the one that measures content |

## 6.8 How it is reported in the product

Captioning is the only capability in the testing-corpus gallery marked
**`scored`** rather than **`verified`**. A caption is not right or wrong, and
presenting a ROUGE-L as a verdict would misrepresent what was measured. Each
item carries its score *and* the run's blind floor, so a reader can see the eight
thousandths for themselves.
