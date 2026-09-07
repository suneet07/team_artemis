# Phase 0 — batch saturation at the worst-case sequence length

`NVIDIA A100-SXM4-80GB`, attention `flash_attention_2`, checkpointing 60/60.

Inputs: **512px x 6 composites = 1536 vision tokens/sample**, against 42 for a 120px BEN chip.

| micro-batch | peak VRAM | s/step | samples/s |
|---|---|---|---|
| 1 | 12.37 GB | 1.2275 | 0.815 |
| 2 | 16.02 GB | 2.1006 | 0.952 |
| 4 | 23.3 GB | 3.7657 | 1.062 |
| 8 | 37.86 GB | 6.9572 | 1.15 |

**Largest that fits: 8. Fastest: 8.**

Synthetic tiles. Memory and step time only -- no loss here is evidence of anything. Valid for this sequence length; a shorter one is cheaper and a longer one may not fit.
