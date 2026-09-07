# Modal accounts — which work runs where

Modal **volumes, secrets and deployed apps are per-account**. Nothing crosses
between them. An account that has not staged a dataset does not have it, and an
account that has not run training does not have the ~9 GB HuggingFace model
cache either — the first GPU job on a fresh account pays for that download
before it trains anything.

That is the whole reason this file exists. "Is the BEN corpus on this account?"
is a question with a real cost attached to guessing wrong, and the CLI will not
answer it for you: there is **no per-command `--profile` flag** on `modal run`,
so the active profile is stateful and silent.

## Before any command that spends money

```bash
modal profile current
```

One line, unambiguous, and it lands in the transcript ahead of the spend. Do
this rather than assuming, because `modal profile activate` persists across
sessions and the account you used yesterday is the account you are on today.

## The accounts

| Profile | Adapter | Volume | What is staged |
|---|---|---|---|
| `suneet-sharan-ug25` | `rs_vqa` (G1/G2) | `satquery-data` | BEN.txt chips + bench, RSVQA-LR train/test, RSVQA-HR train/test/plan_test_phili, merged corpus, trained adapter, gate + ablation reports |
| `suneetsharan14` | `change_vqa` (G4) | `satquery-data` | SpaceNet 7 — 60 AOIs, 2,846 files, `plan.json` |
| `revanshu2473` | `rs_ground_caption` (G3) | `satquery-data` | nothing — volume created 2026-09-04, no data staged |

`rs_vqa` is trained and evaluated; its adapter lives at
`/data/checkpoints/rs_vqa/adapter` on `suneet-sharan-ug25` and is **not**
reachable from any other account.

`revanshu2473` belongs to a teammate, not to the repo owner. Treat anything on
it as theirs: it is the account for G3's staging and training, and nothing else
should be run there.

## Switching

```bash
modal profile list                  # all profiles, active one marked
modal profile activate <name>       # stateful — persists until changed
```

**Prefer the environment variable over activating.** `MODAL_PROFILE` overrides
the active profile for one command and changes no state:

```bash
MODAL_PROFILE=revanshu2473 modal volume list
```

This is the safer form and it is what an earlier version of this file said did
not exist — the claim was that the absence of a `--profile` flag on `modal run`
left no per-command option. There is no such flag, but the variable does the
same job. It matters because `modal profile activate` persists: the account you
switched to in order to check one listing is the account your next GPU job
bills, and that mistake is silent and expensive. Reserve `activate` for when
you are genuinely moving your work to another account, and use the variable for
everything you are only looking at.

A new account is added with `modal token new` while logged into that account in
the browser. Omit `--profile` and Modal names the profile after the workspace,
which is what keeps profile name and account name the same thing. Prefer
`--no-activate --verify`: verify proves the credentials work, and not
activating means the switch stays a decision rather than a side effect.

## Why not one account

The four adapters are independent training runs of 4.2 h to 15.7 h. Running
them on separate accounts is parallelism, not credit farming — each teammate
owns the account their adapter trains on. The cost is that every account needs
its own staging pass, which is CPU time and bandwidth rather than GPU spend.
