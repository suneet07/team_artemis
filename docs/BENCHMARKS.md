# Benchmarks and comparisons

Every benchmark this system is scored on, with **every model we compare against**
and the blind baseline that says whether the score means anything.


## How to read these numbers

**Blind baselines are the floor, not a courtesy.** A VQA score is compared
against the majority-answer ceiling — what a model scores by ignoring the image
and always answering the most common class. A captioning score is compared
against the wrong-image floor: the same model captioning a *different* scene.
A number that does not clear its baseline has not demonstrated sight.



---

## 1. RSVQA — single-image VQA

Average accuracy (AA), the metric RSVQA reports.

### RSVQA-HR · 0.30 m aerial

| system | AA |
|---|---|
| **ours — Qwen3-VL-4B + LoRA** | **85.06** |
| RSVQA authors' own model | 83.12 | 
| *blind — majority class* | *62.6* | 

### RSVQA-LR · 10 m Sentinel-2

| system | AA |  
|---|---|
| **ours — Qwen3-VL-4B + LoRA** | **83.08** | 
| RSVQA authors' own model | 81.49 | 
| *blind — majority class* | *55.8* | 

Clears the published model but **misses our internal target of 88**. The whole
shortfall is one type: `count` at **56.2%**, whose vision contribution is 2.9
points — largely answered from question phrasing rather than pixels.

---

## 2. BigEarthNet VQA — the RS-specific VLM field

The widest comparison set in the project: this is where every remote-sensing
specialist model can be lined up on the same rows.

### BEN binary

| system | AA | notes |
|---|---|---|
| **ours — Qwen3-VL-4B + LoRA** | **76.78** | |
| RS-InternVL | 73.29 | 1B, fine-tuned, RS-specialised |
| zero-shot Qwen | 61.96 | our backbone, untrained |
| GPT | 60.39 | |
| *blind — majority class* | *52.6* | — |
| GeoChat | 50.82 | RS-specific VLM |
| SkyEyeGPT | 48.87 | RS-specific VLM |
| LHRS | 48.23 | RS-specific VLM; licence-barred for us anyway |

**Three RS-specific VLMs score at or below the blind baseline here.** That is the
finding that settled the architecture question: adopting one would have cost a
second model and bought nothing.

### BEN MCQ

| system | AA |
|---|---|
| **ours — Qwen3-VL-4B + LoRA** | **73.62** |
| RS-InternVL | 51.49 |
| zero-shot Qwen | 37.55 |
| GPT | 34.93 |
| *blind — majority class* | *29.3* |

---

## 3. CDVQA — bi-temporal change VQA


| system | AA |
|---|---|
| Qwen3.5-2B + LoRA — best in the published paper | 68.59 |
| **ours — Qwen3-VL-4B + LoRA** | **68.0** |
| Qwen3-VL-4B + LoRA — same paper, our exact backbone | 67.86 | 
| VisTA — prior SOTA, specialised architecture | 65.9 | |
| SOBA | 60.3 | |
| CDVQA paper baseline | 55.3 | |
| *blind ceiling — majority answer per type* | *45.0* | — |

**Level with the published fine-tuned result for the same backbone** (68.0 vs
67.86), 0.6 under the paper's best, and 2.1 over the prior specialised SOTA.
Instruction-following 100%.

---

## 4. VRSBench referring — grounding

acc@0.5: the predicted box must overlap ground truth by ≥50% IoU. Random
unstratified sample, matching the published protocol.

| system | acc@0.5 | trained on VRSBench? |
|---|---|---|
| **ours** | **62.7%** | **no** |
| base Qwen, plain prompting | 61.3% stratified / 60.7% random | no |
| GeoChat (published) | 60.6% | **yes, fine-tuned** |
| zero-shot Qwen, first attempt | 15.60 mIoU | no |



**This capability was won by not training.** A published fine-tuned baseline is
beaten by one Apache-2.0 model and a better-written prompt.





---

## 6. reBEN — SAR land cover


| system | accuracy |
|---|---|
| **`resnet50-s1` on radar (BIFOLD)** | **74.95%** |
| BIFOLD optical arm (`resnet50-s2`) on the same rows | 0.4968 — chance |
| *majority-answer floor* | *50.1* |
| the same radar model, band order reversed | 49.9% — chance |


---



## Scoreboard

| benchmark | ours | blind | best comparison | verdict |
|---|---|---|---|---|
| RSVQA-HR | **85.06** | 62.6 | 83.12 | **win** |
| RSVQA-LR | **83.08** | 55.8 | 81.49 | **win** (internal target missed) |
| BEN binary | **76.78** | 52.6 | 73.29 | **win** |
| BEN MCQ | **73.62** | 29.3 | 51.49 | **win** |
| CDVQA | **68.0** | 45.0 | 68.59 / 67.86 | **draw** |
| VRSBench referring | **62.7%** | — | 60.6% | **win, untrained** |
| reBEN SAR | **74.95%** | 50.1 | — | **win** |


