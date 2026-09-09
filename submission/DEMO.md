# Demo Video

**Project:** SatQuery AI · **PS ID:** SIH26167

| | |
|---|---|
| **Video link** | *(YouTube / Google Drive link — set sharing so a reviewer without an account can open it)* |
| **Length** | |
| **Recorded against** | build *(paste the `build` field from `/api/v1/meta/health`)* |

A demo video is optional for SIH but recommended.

## What the recording should show

The point of the demo is that the system is **running**, and that every answer
carries its evidence. In order:

1. **Ask a question of your own imagery.** Upload a GeoTIFF in Receiving, then
   ask it in the workspace. Show the answer arriving with a confidence chip.
2. **Open the trace.** Every tool that ran, its parameters checked against the
   manifest, and the threshold chosen — this is the part a general VLM cannot do.
3. **A cross-modal disagreement.** Optical and SAR marking different extents, and
   the system reporting reduced confidence rather than picking a winner it cannot
   justify.
4. **A refusal.** Ask something the imagery cannot support and show the system
   saying so and naming the fix — a refusal is a graded deliverable, not an error.
5. **The testing corpus.** 200 rows from public test splits the model was never
   trained on, asked live through the same router.

## Notes for recording

- The backend cold-starts in about **90 seconds**. Warm it with one query before
  recording, or the first answer looks broken.
- `RUNBOOK.md` covers starting and stopping the GPU backend, and what each state
  costs.
