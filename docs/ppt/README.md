# SIH 2025 — Idea Presentation, six slides

One markdown per slide. Each file carries **what goes on the slide**, **how to lay
it out**, and **what to say if a judge pushes back**.

| slide | file | the single thing it must land |
|---|---|---|
| 1 | [`slide-1-title.md`](slide-1-title.md) | this is not an idea — it is running |
| 2 | [`slide-2-solution.md`](slide-2-solution.md) | it beats the people who built the benchmarks |
| 3 | [`slide-3-technical.md`](slide-3-technical.md) | one 4B model, four capabilities, every answer traced |
| 4 | [`slide-4-feasibility.md`](slide-4-feasibility.md) | feasibility is proven, not argued — plus the risks named first |
| 5 | [`slide-5-impact.md`](slide-5-impact.md) | minutes to seconds, and an answer you can audit |
| 6 | [`slide-6-references.md`](slide-6-references.md) | every claim traceable to a source |

## The template's own rules, which we obey

- **Six slides maximum**, including the title.
- **No paragraphs.** Points, diagrams, infographics, pictures.
- Precise, easy to understand, unique and novel.
- Do not change the template's section headings.
- Export to **PDF** — the portal takes nothing else.

## The strategic read

Most SIH entries at this stage present an *idea*. We present a **measured, running
system** with published baselines beaten. That is the whole leverage, so every
slide should carry at least one number a judge could go and check.

**The four numbers to repeat until they stick:**

```
85.06  RSVQA-HR   above the dataset authors' own model (83.12)
62.7%  grounding  above fine-tuned GeoChat (60.6%) -- with NO training
68.0   CDVQA      level with the published result for the same backbone
74.95% SAR        against a verified 50.1% floor
```

**The one-sentence version:** *a 4-billion-parameter model, on one GPU, that
answers questions about satellite imagery better than the teams who built the
benchmarks — and shows its working for every answer.*
