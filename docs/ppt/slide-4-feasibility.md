# Slide 4 — FEASIBILITY AND VIABILITY

**Required sub-headings:** feasibility analysis · potential challenges and risks ·
strategies for overcoming them.

**The framing that wins this slide:** most entries argue feasibility. We report
it. Then we name our own weaknesses before a judge finds them — which is itself
the strongest possible feasibility signal.

---

## Block A — Feasibility: already demonstrated

> ### Not "can this be built" — it is built, measured and deployed.

| evidence | measured |
|---|---|
| Training cost | `rs_vqa` **3.0 h** against a 4.2 h budget · `change_vqa` **2.4 h** |
| Hardware | **one L4 GPU**, 19.13 GB peak — not a cluster |
| Model footprint | **40.3M** trainable params per adapter (**0.899%**) |
| Inference | sub-second to ~2.3 s warm · adapter **17.9 samples/s** |
| Verification | **440** tests · **50** route checks · **200** held-out rows replayed |
| Deployment | self-verifying pipeline, ~11 s redeploy on a cached image |
| Corpora | **117,023** VQA/grounding rows · **85,096** cross-modal rows, generated |

**Cost reality:** the whole system trains in **under 6 GPU-hours** and serves from
a single L4. This is affordable to run, and affordable to *re*-train when ISRO
supplies Cartosat/RISAT data.

---

## Block B — Risks, named honestly

> Four tiles. Each: **the risk → the number → what we already did.**

**① Domain gap — the graded set is Cartosat-2S + RISAT**
Every public training source is Western. **Unclosable with public data.**
→ *Mitigated:* every prompt carries the scene's GSD (scale conditioning); three
spectral composites instead of RGB; transfer measured against the RSVQA paper's
own **5.6-point** cross-split drop as a yardstick.
→ *Residual risk: accepted and stated.* Re-training on ISRO data is ~3 GPU-hours.

**② Counting is our weakest capability**
RSVQA-LR `count` **56.2%**; vision contribution only **2.9 points** — the model
answers partly from question phrasing.
→ *Mitigated:* counting is routed to **deterministic arithmetic** wherever a mask
exists — that path scored **100% on 2,012 rows**.
→ *Known fix, costed:* bucket the training manifest so the model emits ranges
directly. One re-train.

**③ Licence — "codes and models" is a deliverable**
A permissive badge on a repository says nothing about the imagery underneath.
→ *Mitigated:* every source cleared and recorded in `CREDITS.md`; the barrier is
**enforced by a test that fails the build**, not by convention.
→ *Accepted cost:* we trained change detection on 4 m imagery rather than a 0.5 m
academic-only set. **A licence-clean 80% beats an unusable 91%.**

**④ Benchmark scores are not official-scorer scores**
Ours use an in-repo comparator.
→ *Mitigated:* the caveat is printed on **every** report, not buried.
→ *Next step:* wrap the official scorers — a named deliverable.

---

## Block C — Strategy: the discipline that de-risks all of it

Three practices, each with the failure it prevents:

**Never quote a score without its blind baseline.**
RSVQA presence is **76.3% "yes"** — a blind model scores that. We report the gap.
*Prevents: shipping a prior as a capability.*

**Measure before building; delete when the measurement says so.**
Two of four planned adapters were **never trained** — grounding because the base
model already beat a fine-tuned baseline, fusion because the dataset's own authors
showed fusion (AP 0.711) is *worse* than optical alone (0.714).
*Prevents: spending the budget on work that cannot win.*

**Test the product, not just the parts.**
Every benchmark number came from calling a component directly. A **200-row
held-out replay through the live system** found **9 defects that 440 unit tests
missed** — including a radar classifier that scored 74.95% in the lab and **had
never run on a single deployed request**. Fixing routing moved grounding
**12% → 80%** and SAR **46% → 74%**, with no retraining.
*Prevents: a demo that is green everywhere and wrong in production.*

---

## Layout direction

```
┌── FEASIBILITY: PROVEN ─────────────────────────────────┐
│  6 GPU-hours total  ·  one L4  ·  440 tests  ·  live   │
└────────────────────────────────────────────────────────┘
┌ RISK ①      ┬ RISK ②      ┬ RISK ③      ┬ RISK ④      ┐
│ domain gap  │ counting    │ licence     │ scorers     │
│ 5.6pt yard  │ 56.2% → arith│ build-gated │ caveated    │
└─────────────┴─────────────┴─────────────┴─────────────┘
┌── WHY IT HOLDS ────────────────────────────────────────┐
│ blind baselines · measure-then-build · test the product│
└────────────────────────────────────────────────────────┘
```

Use a **traffic-light column** on the risk tiles: risk in amber, mitigation in
green. Judges scan for whether risks were considered — make that visible at a
glance.

---

## Speaker notes

- Open this slide with: **"Let me tell you what's wrong with it first."** That
  single move converts scepticism into trust, and everything after lands harder.
- The 12% → 80% story is the best thirty seconds in the deck. It shows the team
  finds its own bugs *and* has a method that catches them.
- Do not soften ① — the domain gap is real, unclosable with public data, and
  saying so is more credible than a hand-wave.

## If a judge pushes

**"Your grounding beat a fine-tuned model without training — is that luck?"**
> It is measured on the published protocol against the published number. And we
> tested the alternative properly: a detector pipeline whose *ceiling* was 63.7%
> scored **34.7%** when actually built. We shipped the thing that measured better.

**"What happens when we give you Cartosat data?"**
> Roughly three GPU-hours per adapter, and the corpus generator already emits the
> same three composites, so Indian data drops into the existing convention rather
> than being a special case.
