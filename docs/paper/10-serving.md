# 10. Serving — one base, three adapters, and a deploy that verifies itself

## 10.1 The architecture, and the constraint that forced it

```
Qwen3-VL-4B-Instruct  (one set of weights, loaded once)
   |
   +-- rs_vqa             LoRA adapter    40.3M params
   +-- change_vqa         LoRA adapter    40.3M params
   +-- rs_ground_caption  no adapter -- the base model, prompted
```

PEFT multi-adapter serving: one base, `load_adapter` / `set_adapter` /
`disable_adapter_layers`. Adapter weights are tens of megabytes each; the base is
several gigabytes.

This is not an optimisation, it is a correctness requirement discovered by
failure. `_RUNNERS` was originally keyed by `(base, adapter)`, which loaded **a
separate full 4B model for every registered tool** — three of them on a 22 GB L4.
CUDA ran out of memory mid-query, the executor recorded it as a tool failure, and
the symptom that reached a user was *"change_vqa never runs"*. Keying by base
alone fixed it, and latency went from a timeout to **2.3 s**.

## 10.2 The three Modal accounts

Training and evaluation ran across three separate workspaces, and **the test
split lives on the same account as the training run that produced it**:

| account | test corpora | checkpoints | segment |
|---|---|---|---|
| `suneet-sharan-ug25` | `vrsbench_val`, `rsvqa_hr`, `rsvqa_lr`, `sn2`, `rareplanes`, `sn6` | `rs_vqa`, `change_vqa` | grounding, captioning, `rs_vqa` — **and serving** |
| `suneetsharan14` | `cdvqa_qa`, `second`, `sn7`, `india` | `change_vqa` + step snapshots | `change_vqa` |
| `revanshu2473` | `reben` | — | SAR / BIFOLD |

Which `change_vqa` checkpoint serves was settled **by hash, not by name**,
because two accounts hold a directory with the same name and only one is the
CDVQA run. Full reasoning in
[`docs/segments/08-testing-corpora.md`](../segments/08-testing-corpora.md).

## 10.3 Deploy verification — `scripts/deploy_verify.sh`

Ad-hoc `modal deploy` followed by a guessed `sleep` produced two failure modes
that look identical from outside: a warm container still serving the old code,
and a container that never started. The script distinguishes them.

```
1. run the full test suite         -- refuses to deploy on a red tree
2. snapshot the working tree       -- so edits cannot race the upload
3. deploy, bounded at 15 minutes
4. poll for the expected build id  -- computed locally, content-addressed
5. exercise every route and task   -- 50 checks including error paths
```

Three of those steps exist because of a specific failure:

**The snapshot.** Modal hashes the payload as it uploads and aborts with *"was
modified during build process"* if any file changes underneath it. Four deploys
died that way — twice on frontend build artifacts, twice because work continued
in `scripts/` while the upload ran. Excluding directories fixed the first pair
and **could not** fix the second: the payload *is* the code being deployed.
Copying the tree first makes editing during a deploy simply allowed.

**The content-addressed build id.** `_build_id` originally hashed file sizes and
**mtimes**, which made it strictly local — copying a tree into an image rewrites
every mtime, so a served id could never match a locally computed one. The symptom
was a deploy touching only `scripts/` reporting `WARN still serving <old id>`:
true, correct, and indistinguishable from a failure. Now it hashes **contents**,
cached once per process, so the deploy script computes the expected id and waits
for exactly that.

**Route verification.** `verify_routes.py` drives all 20 endpoints including the
error paths, checks contract fields against the frontend's own `types.ts`, and
drives all 8 `Task` branches. A missing *required* contract field does not
degrade the console, it blanks it — that is how `bundle.provenance` once took the
whole workspace down, and how a gallery endpoint later did it again.

It also now **executes one crossmodal query**, because everything above only
*planned*. That gap shipped an outage ([11](11-failure-atlas.md)).

## 10.4 Serving parity — the hard part

The benchmark numbers were produced by an evaluation harness. The product is a
different code path. Every difference between them is a silent error, and there
were four.

| what the benchmark did | what serving did | cost |
|---|---|---|
| three distinct composites (true-colour, false-colour, short-wave) | one image repeated three times | NIR and SWIR never reached the adapter |
| B04/B03/B02 into R/G/B | first three bands in file order — B02/B03/B04 | red and blue swapped on every scene |
| view count per adapter (3 / 6 / 2) | a hardcoded 3 | a duplicated date on every change pair |
| `PRECISE_PROMPT` fed the whole referring sentence | fed an extracted noun | not the input 62.7% was measured on |

The composite fix is now verified by **pixel equality against the corpus PNGs the
adapter trained on** — not a shape check, because the stretch is per channel and
the 20 m bands are upsampled, and either drifting would leave shapes intact while
changing what the model sees.

The rule that came out of this, now enforced by test:

> `composites` is a **sequence-length budget**, not a forced count. One view
> repeats up to the budget; several real views pass through untouched.

Reading it as "always send N" pads duplicates; reading it as "truncate to N"
hands a change model a single date and it converges on the answer prior.

## 10.5 The answer path

Serving applies **no answer contract**. The benchmark scores `format_answer(raw,
contract)`; the product shows the raw generation.

For a well-formed question this is invisible — the adapter's instruction
following is 100%. For a malformed one it is not: a garbled *"is it a urban area
of rural"* produced `"no."`, an answer outside that question's `{urban, rural}`
answer space, where the contract would have mapped it in.

This is a known, scoped gap rather than a bug: serving cannot infer a benchmark
name from free text. The **gallery** can, because every item carries its
`contract`, which is how 175 held-out items were scored through the identical
formatter that produced the published numbers.

## 10.6 Operational facts

```
cold start          ~45 s   (first question per container pays the 4B weight load)
warm query          sub-second to ~2.3 s
max_containers      1       -- state lives in process memory
scaledown_window    5 min
gallery imagery     on the Volume, not in the image (53 MB would slow every deploy)
```

`max_containers=1` is a correctness fix, not a cost one: scenes, bundles and
queries live in process memory, and requests were round-robining across replicas
that shared no state, so uploads randomly vanished. Moving that state to the
Volume is the fix, and it is only worth doing when more than one person uses the
demo at once — which is also when `max_containers=1` stops being right.

A Modal Volume is **snapshotted at container start**. Anything written to it
afterwards — by `modal volume put`, which is how the gallery is published — stays
invisible until the mount is reloaded. The symptom is quietly wrong rather than
broken: the endpoint keeps serving the manifest that existed at boot. There is
now an explicit reload.
