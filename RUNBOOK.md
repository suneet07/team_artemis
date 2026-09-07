# Demo runbook

Starting, checking and stopping the deployed system. Written because the
expensive mistakes here are not dramatic — nothing crashes, you simply leave a
GPU reachable and find out later.

## What costs money, and what does not

| state | billing |
|---|---|
| app **deployed**, 0 tasks | **nothing** — containers are scaled to zero |
| app deployed, a request arrives | an L4 spins up: ~90 s cold start, then bills while warm |
| app deployed, idle after a request | bills for `scaledown_window` = **5 minutes**, then stops |
| app **stopped** | nothing, and the URL returns an error |

A deployed app with no traffic is free. What you are managing is not the
deployment, it is **who can send it traffic**.

Both GPU classes are capped at `max_containers=1`, so concurrent load queues
rather than multiplying the bill. The worst case of leaving it up is a slow
demo, not a surprise invoice — but "worst case" assumes the cap, so do not
raise it without re-reading this table.

## Workspace

`suneet-sharan-ug25`. It is the only one carrying both adapters
(`/data/checkpoints/rs_vqa/adapter`, `/data/checkpoints/change_vqa/adapter`)
plus `gallery/`. The other two profiles are incomplete:

- `suneetsharan14` — `change_vqa` only, no `rs_vqa`
- `revanshu2473` — no `checkpoints/` at all

There is no `rs_ground_caption` adapter and there is not meant to be one:
grounding runs on the base model, which is where its 62.7% acc@0.5 was
measured. An empty grounding slot is correct, not a missing file.

This workspace is also the one with limited credits. That is not a reason to
deploy elsewhere — elsewhere cannot serve the system — it is a reason to stop
the app when you are not demoing.

```bash
modal profile activate suneet-sharan-ug25
```

## Start

```bash
modal deploy scripts/modal_phase0.py
```

Takes a few minutes. It builds images and registers every function; only the
web endpoints become reachable, and none of them run until called.

The API lands at:

```
https://suneet-sharan-ug25--satquery-phase0-ml-api-web.modal.run
```

Prefer the scripted path when you have changed code — it runs the suite first,
deploys from a snapshot, and waits out the scaledown window so you are not
verifying the build you just replaced:

```bash
scripts/deploy_verify.sh
```

## Check it is actually up

Health is deliberately outside the API-key gate, so it answers even when the
deployment is locked down:

```bash
curl -s https://suneet-sharan-ug25--satquery-phase0-ml-api-web.modal.run/health
```

The first call pays the ~90 s cold start. A timeout on the first request is
usually the model loading, not a failure — retry once before concluding
anything.

Confirm the adapters actually loaded, which is the difference between the real
system and a deterministic-only fallback that still returns 200:

```bash
curl -s https://suneet-sharan-ug25--satquery-phase0-ml-api-web.modal.run/api/v1/meta/health
```

## Stop

```bash
modal app stop satquery-phase0-ml
```

This is the only measure that is certain. The URL stops resolving to a running
app, so no request from anyone can start a container.

Do this whenever you are not actively demoing. It costs one command to start
again.

## State

```bash
modal app list
```

`State: deployed` with `Tasks: 0` means reachable but idle — **free right now,
billable the moment anyone calls it**. `Tasks: 1` means a container is warm and
the clock is running.

```bash
modal app logs satquery-phase0-ml
```

## Locking it down

The API is open by design: a demo anyone can open is the point. If you want to
narrow it for a window, both of these default to today's behaviour and change
nothing unless set:

```bash
# require a shared key on everything except the health checks
modal secret create satquery-api SATQUERY_API_KEY=<value>

# restrict browser origins to the deployed frontend
SATQUERY_ALLOWED_ORIGINS=https://<your-app>.vercel.app
```

Understand what the key buys before relying on it. The frontend reads it from
`VITE_API_KEY`, and Vite inlines `VITE_*` into the built bundle, so the key is
readable by anyone who opens the site. It stops someone holding only the API
URL. It does not stop a visitor.

The Modal URL is not a secret either. It is in the bundle, it is in the
browser's network tab, and Modal URLs are structured
(`<workspace>--<app>-<class>-<method>.modal.run`) rather than random. Plan on
it being public.

## If a judge sees an error

1. `modal app list` — is it deployed at all?
2. `curl .../health` — if this times out twice, the container is failing to
   start; check `modal app logs`.
3. `curl .../api/v1/meta/health` — 200 here with adapters missing means it is
   serving deterministic evidence only. The answer will be honest but will not
   show the trained system.
4. Cold start is ~90 s. Warm it with one query before the demo starts rather
   than letting the first judge pay for it.
