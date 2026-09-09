# Final Presentation

**Project:** SatQuery AI
**PS ID:** SIH26167 — Vision-Language Model for Remote Sensing Imagery
Interpretation and Analysis
**Organisation:** ISRO / Space Applications Centre (SAC)

## Deck

| | |
|---|---|
| **File** | *(add `SatQuery-AI-SIH26167.pdf` here, or the viewer link below)* |
| **Viewer link** | *(Google Drive / OneDrive link, if the file is too large for GitHub)* |
| **Slides** | 6 |
| **Format** | 16:9, exported as PDF |

> If the PPT exceeds GitHub's file size limit, upload it to Drive/OneDrive and
> put an **accessible** viewer link in the table above — set sharing so a
> reviewer without an account can open it.

## Structure

Six slides, one per required heading.

| # | slide | carries |
|---|---|---|
| 1 | **Title** | PS ID, theme, category, team — plus a one-line hook strip: a working system, with its four headline numbers |
| 2 | **Proposed Solution** | detailed explanation · how it addresses the problem · innovation and uniqueness. Four capability tiles, the orchestration band, the blind-baseline strip |
| 3 | **Technical Approach** | technologies · methodology. The full routing tree is the hero of this slide |
| 4 | **Feasibility and Viability** | challenges and risks · strategies · feasibility analysis. Reordered so it reads as one argument: shippable data is scarce, we held to a commercial licence standard anyway, and beat the benchmark authors |
| 5 | **Impact and Benefits** | impact on the target audience · social, economic and environmental benefits |
| 6 | **Research and References** | datasets, models and methods, each with its licence |

## The claim the deck is built around

Every figure on every slide is measured on a **held-out public benchmark** and
shown beside the score a system gets **without looking at the image**.

| | ours | blind baseline | published comparison |
|---|---:|---:|---|
| RSVQA-HR | **85.06** | 62.6 | dataset authors' own model 83.12 |
| VRSBench referring | **62.7%** | — | GeoChat, fine-tuned, 60.6% |
| CDVQA Val | **68.0** | 45.0 | same backbone, fine-tuned, 67.86 |
| reBEN SAR | **74.95%** | 50.1 | — |

Full comparison landscape: [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

## Before export

- [ ] Team name matches the portal registration **character for character**
- [ ] Team ID filled in
- [ ] PS ID reads **SIH26167**, category **Software**, theme **Space Technology**
- [ ] Screenshots are our own console output, not dataset samples or stock imagery
- [ ] Exported as **PDF**
- [ ] Viewer link is accessible without an account
