# Slide 5 — IMPACT AND BENEFITS

**Required sub-headings:** potential impact on the target audience · benefits
(social, economic, environmental, etc.).

**The framing:** the impact is not "AI for satellites". It is **removing the
analyst bottleneck without removing the analyst's ability to check the answer.**

---

## Block A — The problem being removed

```
TODAY                                    WITH SATQUERY
─────                                    ─────────────
analyst opens QGIS                       types a question
picks bands, computes an index           answer in seconds
chooses a threshold by eye               threshold chosen and RECORDED
writes it up                             evidence exported as GeoTIFF
                                         trace shows every step

minutes to hours per scene    ──────>    seconds per scene
expertise required to ask                anyone can ask
result depends on the analyst            result is reproducible
```

**The multiplier:** ISRO archives grow faster than they can be read. The
constraint is not imagery — it is **questions asked per analyst-hour**.

---

## Block B — Target audience and use

| who | what they ask | what changes |
|---|---|---|
| **ISRO / SAC analysts** | *"What land cover is in this radar scene?"* | archive triage at scale; SAR read without a SAR specialist |
| **Disaster response** | *"What changed between these two dates?"* | flood and damage extent in seconds, with the mask as a GeoTIFF |
| **Agriculture** | *"How much of this scene is vegetation?"* | measured NDVI coverage, not an impression |
| **Urban planning** | *"Where is the built-up area?"* | change over time, auditable |
| **Non-specialists** | plain language | **the expertise barrier drops** — no band maths required |

**Cross-modal is the one only we do properly:** optical is blind through cloud,
radar sees through it. The system knows *which sensor to believe and why* — cloud
over water goes to radar because cloud is opaque to optical and transparent to
C-band. **That is monsoon-season India, stated as physics.**

---

## Block C — Benefits

**⚙️ Operational**
- **Minutes → seconds** per question
- Runs on **one L4 GPU** — deployable on-prem, no cluster, no data egress
- Four capabilities from **one 4B model**; adapters are tens of MB
- **Every answer auditable**: tool, parameters, threshold + reason, RMSE in pixels
- Masks export as **GeoTIFF straight into QGIS** — output is usable, not a screenshot

**💰 Economic**
- Entire system trains in **under 6 GPU-hours**
- Retraining on Indian sensors: **~3 GPU-hours per adapter**
- **Zero licence liability** — every source cleared and recorded; enforced by a
  build-failing test
- No per-query API cost, no vendor lock-in — Apache-2.0 backbone, self-hosted

**🌍 Social and environmental**
- **Faster disaster assessment** — flood and change extent in seconds
- **Crop and vegetation monitoring** at archive scale
- **Democratised access** — a district officer can ask what previously needed a
  remote-sensing analyst
- **Small model = small footprint**: 4B on one L4, not a 70B on a cluster

**🇮🇳 Strategic**
- Handed over as **code and weights**, not a hosted service
- Built to be **retrained on Cartosat-2S and RISAT** the day that data exists
- No dependency on a foreign API for inference

---

## Block D — The differentiator, in one box

> ## Trust is the product.
>
> An answer you cannot check is not usable in an operational setting.
>
> SatQuery reports **what it measured**, **which tool measured it**, **what
> threshold was used and why**, and **how confident it is** — and where two
> sensors disagree and no physical rule explains it, it says **"they disagree"**
> and reports only the agreed extent, at reduced confidence.
>
> **It is built to be checked, and it says when it does not know.**

---

## Layout direction

- **Top strip:** the TODAY → WITH SATQUERY comparison as two columns with a bold
  arrow. This is the emotional beat — keep it visual, minimal words.
- **Middle:** audience table, five rows, icons per row.
- **Lower:** four benefit tiles (⚙️ 💰 🌍 🇮🇳), three bullets each maximum.
- **Bottom:** the "Trust is the product" box, boxed and accented — the last thing
  a judge reads before references.

Do **not** put numbers on this slide beyond `6 GPU-hours`, `one L4`, `3 hours`.
Slides 2–4 carry the evidence; this slide carries the *meaning*.

---

## Speaker notes

- The strongest line to deliver aloud: **"An answer you cannot check is not
  usable operationally."** Then point at the trace panel in the demo.
- The cloud/radar example is worth ten seconds — it is concrete, physical, and
  obviously Indian-relevant in monsoon season.
- Close on: *"code and weights, retrainable on your sensors in three GPU-hours."*
  That is what a handover-focused evaluator wants to hear.

## If a judge pushes

**"Who actually uses this day to day?"**
> The analyst triaging an archive. Today every question is a manual GIS session.
> This makes asking cheap, and keeps the evidence so the answer survives review.

**"Why not a commercial API?"**
> Data never leaves the premises, there is no per-query cost, and you receive the
> weights. An Apache-2.0 backbone on your own hardware is a strategic asset; an
> API key is a dependency.
