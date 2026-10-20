#!/usr/bin/env bash
# Tune-guarded full lib test run through the central scheduler: tests may rewrite the host tune
# file, so it is snapshotted before and restored after. Usage: scripts/tg.sh LABEL [cargo test args...]
set -u
cd "$(dirname "$0")/.."
label=${1:-lib}; shift || true
python3 ~/.cache/gibbs/central_scheduler.py run --label "t-$label" -- bash -c 'T=$(ls ~/.cache/gibbs/host/tune-*.json); cp -p "$T" /tmp/tune.guard; cargo test --release --lib -q "$@"; rc=$?; cp -p /tmp/tune.guard "$T"; exit $rc' _ "$@"
