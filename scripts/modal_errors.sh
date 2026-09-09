#!/usr/bin/env bash
# Tracebacks and crash lines from a deployed Modal app, nothing else.
#
# `modal app logs` interleaves image-build chatter ("Downloaded numpy", a
# hundred lines of it) with the one traceback that matters, so the signal is
# easy to scroll past. This filters to the lines that mean something broke.
#
#   scripts/modal_errors.sh [app-name] [profile]
set -u
APP="${1:-satquery-phase0-ml}"
PROFILE="${2:-${MODAL_PROFILE:-}}"
if [ -z "$PROFILE" ]; then
  echo "usage: $0 <app> <workspace>   (or set MODAL_PROFILE)" >&2
  exit 2
fi
MODAL_PROFILE="$PROFILE" modal app logs "$APP" 2>&1 \
  | grep -viE "DeprecationWarning|asyncio\.set_event|^\s*Downloaded |^\s*Building |^\s*Built " \
  | grep -iE "error|traceback|exception|crash|failed|modulenotfound|File \"|^\s+[a-z_]+\(|Runner failed" \
  | tail -"${3:-40}"
