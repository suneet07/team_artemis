# Six gap closures — where they landed

The completeness audit left six documented shortfalls against the master plan.
They were first drafted in a standalone `satquery_extensions/` folder; that
folder is gone and each one now lives in the module that owns its concern,
because a parallel package would have been a second implementation of things the
pipeline already does, sitting next to the first and diverging from it.

| Gap | Where it lives now | Tests |
|---|---|---|
| AROSICS escalation for non-translation residuals (§4.3 step 5) | `satquery/coreg/arosics_backend.py`, escalated from `register_pair` | `tests/test_extensions.py` |
| Tiling in the query path, top-k + mosaic (§4.4 step 5) | `satquery/agent/tiled.py` | `tests/test_extensions.py` |
| One box convention (TEAM_CONTEXT §10) | `satquery/qgen/boxes.py`, used by `gen_bbox` and `object_box_fallback` | `tests/test_extensions.py` |
| Polarisation dropout actually applied (§4.2) | `satquery/qgen/dropout.py` | `tests/test_extensions.py` |
| CLIP tile scorer (§4.4 step 4) | `satquery/tiling/clip_scorer.py`, plugs into `score_tiles` | `tests/test_extensions.py` |
| `radar_shadow` reachable (§4.7.2) | `slope_mask_from_dem` in `satquery/fusion/fusion.py` | `tests/test_extensions.py` |

## What the first drafts got wrong, and why it is worth recording

Every one of the six imported cleanly and did the wrong thing quietly. That is
the pattern to watch for, not the individual bugs.

- **Silent fallback stubs.** Five of the six wrapped their real import in
  `try/except ImportError` and substituted a dummy. `dropout_pipeline` imported
  `apply_pol_dropout` from the wrong module and fell through to a stub that
  mutated a field; `tiled_pipeline` fell through to a `plan_tiles` returning one
  hardcoded tile. Both imported fine and reported success. **A missing
  dependency must be reported, never substituted.**
- **Wrong claims in a docstring are a defect.** The AROSICS draft said it
  handled "scale, rotation, and translation". `COREG` solves a single global
  translation and `COREG_LOCAL` fits a grid of local shifts; neither estimates
  rotation or scale. A co-registration everyone believes is correcting geometry
  it never touches is precisely how change detection reports building outlines
  as change.
- **Adding a format is not reconciling formats.** The box draft appended
  `bbox_1000` while leaving the existing field in place, producing three
  conventions where there were two, and it appended prose to the graded `answer`
  string. It also assumed a y-first axis order; published Qwen-VL conventions are
  x-first. That axis order is still **unverified** — see
  `satquery/qgen/boxes.py`, where `BOX_CONVENTION_VERIFIED` blocks training-data
  generation until someone checks the model card.
- **Interfaces matter more than the algorithm.** The CLIP draft was a class with
  its own `score(query, tiles, image_path)` signature, so it could not be passed
  to `score_tiles`; it also re-opened the scene from disk once per tile. It now
  implements the `TileScorer` callable and scores a whole scene in one forward
  pass.
- **A capability probe must be cheap.** `clip_available()` originally imported
  torch to answer a yes/no question — tens of seconds on a hot path. Both probes
  use `importlib.util.find_spec` now.
- **A refusal in an unregistered tool refuses nothing.** The radar-shadow draft
  returned an error dict from a function no registry knew about. The reasoning
  behind it was right and is preserved: shadow needs a DEM and a look direction,
  and proxying slope from optical texture would put a fabricated physical cause
  into a graded trace. So the row is now genuinely reachable — with a DEM, via
  `slope_mask_from_dem` — and genuinely unreachable without one.
