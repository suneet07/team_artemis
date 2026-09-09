#!/usr/bin/env bash
# Deploy the API and prove it works, with every step time-stamped and bounded.
#
# Ad-hoc `modal deploy` followed by guessed sleeps kept producing two failure
# modes that look identical from outside: a warm container still serving the old
# code, and a container that never started. This does the whole loop and says
# which one happened.
#
#   scripts/deploy_verify.sh [--skip-deploy] [--profile NAME]
#
# Every stage is bounded. Nothing here waits forever.
set -uo pipefail

# No default: a workspace name identifies an account and does not belong in a
# public repository. Set MODAL_PROFILE, or pass --profile.
PROFILE="${MODAL_PROFILE:-}"
if [ -z "$PROFILE" ]; then
  echo "set MODAL_PROFILE (or pass --profile NAME) to the workspace to deploy to" >&2
  exit 2
fi
APP="satquery-phase0-ml"
BASE="https://${PROFILE}--${APP}-api-web.modal.run/api/v1"
SKIP_DEPLOY=0
# Containers hold the previous build until they scale down. scaledown_window is
# 5 minutes, so anything less than that risks verifying the code you just
# replaced -- which has happened repeatedly and reads as "the fix didn't work".
CYCLE_WAIT="${CYCLE_WAIT:-420}"
SCENE="${SCENE:-data/chips/geotiff/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57.tif}"

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-deploy) SKIP_DEPLOY=1 ;;
    --profile) PROFILE="$2"; BASE="https://${PROFILE}--${APP}-api-web.modal.run/api/v1"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- 1. tests
say "running the suite before spending a deploy"
if ! python -m pytest -q >/tmp/dv_tests.txt 2>&1; then
  tail -5 /tmp/dv_tests.txt
  say "FAIL: tests are red, not deploying"
  exit 1
fi
say "OK   $(tail -1 /tmp/dv_tests.txt | tr -d '\r')"

# ---------------------------------------------------------------- 2. deploy
if [ "$SKIP_DEPLOY" -eq 0 ]; then
  # Whatever is serving now, so we can tell when it has been replaced. Empty is
  # fine -- nothing running means the first response is already the new build.
  BEFORE=$(curl -s -m 60 "$BASE/meta/health"            | python -c "import json,sys;print(json.load(sys.stdin).get('build',''))" 2>/dev/null)
  # Deploy from a snapshot, not from the working tree.
  #
  # Modal hashes the payload as it uploads and aborts with "was modified during
  # build process" if any file changes underneath it. Four deploys have died
  # that way -- twice on frontend build artifacts, twice because work continued
  # in scripts/ while the upload ran. Excluding directories fixed the first
  # pair and cannot fix the second: the payload *is* the code being deployed,
  # so there is nothing left to exclude.
  #
  # Copying first gives the upload a tree nothing else can touch, which makes
  # editing during a deploy simply allowed. The copy skips what the image
  # already ignores, which is also what keeps it quick.
  SNAP="$(mktemp -d)/satquery"
  say "snapshotting the tree so edits cannot race the upload"
  mkdir -p "$SNAP"
  # tar takes the POSIX path; modal is a Windows Python and reads "/tmp/..."
  # as "C:\tmp\...", which does not exist. MSYS_NO_PATHCONV is set on the
  # deploy line, so the translation has to happen here.
  SNAP_ARG="$(cygpath -w "$SNAP" 2>/dev/null || echo "$SNAP")"
  tar -cf - --exclude=.git --exclude=node_modules --exclude=__pycache__ \
      --exclude='*.pyc' --exclude=data --exclude=logs --exclude=frontend \
      --exclude=docs --exclude=checkpoints --exclude=graphify-out \
      --exclude=sample_dataset --exclude=.pytest_cache --exclude=.ruff_cache \
      . | (cd "$SNAP" && tar -xf -)

  say "deploying to $PROFILE (bounded at 15m; current build ${BEFORE:-none})"
  if ! MSYS_NO_PATHCONV=1 MODAL_PROFILE="$PROFILE" timeout 900 \
        modal deploy "$SNAP_ARG/scripts/modal_phase0.py" >/tmp/dv_deploy.txt 2>&1; then
    grep -iE "error|traceback" /tmp/dv_deploy.txt | head -5
    say "FAIL: deploy did not finish"
    exit 1
  fi
  say "OK   $(grep -o 'App deployed in [0-9.]*s' /tmp/dv_deploy.txt | head -1)"

  # What the local tree hashes to. The id is content-addressed, so the container
  # serving the right code reports exactly this -- which is the difference
  # between "the deploy landed" and "the id happens to have changed".
  #
  # It also makes a no-op deploy legible. A change confined to scripts/ or
  # tests/ leaves the served package identical, so the id is *supposed* to stay
  # put; comparing against BEFORE alone reported that as "WARN still serving
  # <old>", which reads like a failed deploy and is not one.
  WANT=$(python -c "import sys; sys.path.insert(0, '.'); from satquery.api.server import build_id; print(build_id())" 2>/dev/null)
  say "waiting for build ${WANT:-unknown} to answer (was ${BEFORE:-none}, bounded at ${CYCLE_WAIT}s)"
  DEADLINE=$(( $(date +%s) + CYCLE_WAIT ))
  while :; do
    NOW_ID=$(curl -s -m 120 "$BASE/meta/health"              | python -c "import json,sys;print(json.load(sys.stdin).get('build',''))" 2>/dev/null)
    if [ -n "$WANT" ] && [ "$NOW_ID" = "$WANT" ]; then
      if [ "$NOW_ID" = "$BEFORE" ]; then
        say "OK   serving $NOW_ID -- matches the local tree; served code unchanged by this deploy"
      else
        say "OK   new build serving: $NOW_ID"
      fi
      break
    fi
    # No local id to compare against (python missing, import failed): fall back
    # to the old "did it change" heuristic rather than blocking the deploy.
    if [ -z "$WANT" ] && [ -n "$NOW_ID" ] && [ "$NOW_ID" != "$BEFORE" ]; then
      say "OK   new build serving: $NOW_ID"
      break
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
      say "WARN serving ${NOW_ID:-nothing} but the local tree is ${WANT:-unknown}, after ${CYCLE_WAIT}s; verifying anyway"
      break
    fi
    sleep 15
  done
fi

# ------------------------------------------------- 3. every route, every task
# One exerciser rather than three hand-picked questions: all 20 endpoints
# including the error paths, contract shapes checked against the frontend's own
# types.ts, and all 8 Task branches. Exits non-zero if anything failed.
say "verifying every route and every task"
if python scripts/verify_routes.py --base "$BASE" --scene "$SCENE" --stub-adapters 2>&1      | grep -viE "symlink|developer mode|warnings.warn"; then
  say "done -- all green"
else
  say "FAIL: see the failures above. Container errors, if any:"
  bash scripts/modal_errors.sh "$APP" "$PROFILE" 20
  exit 1
fi
