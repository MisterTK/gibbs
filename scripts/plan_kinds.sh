#!/usr/bin/env bash
# Metal plan kinds of MODEL's prompt run, one per line: scripts/plan_kinds.sh BIN MODEL
set -uo pipefail
GIBBS_DUMP_PLAN=1 "$1" -m "$2" --probe-top5 hi 2>&1 | awk '/^plan /{print $NF}'
