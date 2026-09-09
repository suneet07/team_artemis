# Screenshots

Console screenshots for the submission and the deck. **Our own output only** — a
dataset sample or stock image signals the opposite of a working system.

## Naming

`NN-short-description.png`, two-digit prefix so they sort in narrative order.

```
01-landing.png              the console on open, service budgets visible
02-receiving-upload.png     a GeoTIFF being uploaded, roles designated
03-workspace-answer.png     an answer with its confidence chip
04-trace.png                the execution trace: tools, parameters, thresholds
05-grounding-box.png        a referring expression resolved to a box
06-disagreement.png         optical and SAR marking different extents
07-refusal.png              the system declining a question it cannot support
08-testing-corpus.png       the 200-row held-out gallery
```

## Rules

- **PNG**, full window, no browser chrome cropping out the header.
- Capture against a **warm** backend — a cold start shows an unreachable-API
  banner that reads as a broken system.
- No credentials, tokens, account names or Modal URLs visible in the frame.
- If a screenshot shows a number, that number must be reproducible: prefer rows
  from the testing corpus over one-off uploads.
