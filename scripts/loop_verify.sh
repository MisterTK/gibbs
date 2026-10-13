#!/usr/bin/env bash
# Frozen autoresearch Verify: paired ABBA run of the last kept binary (base) against HEAD (cand).
# Prints delta_pp (candidate minus base geomean roofline %, same ceiling for both) on success.
set -u
cd "$(dirname "$0")/.."
BASE=target/loop-base/gibbs
if [ ! -x "$BASE" ]; then echo "metric-error: $BASE missing" >&2; exit 1; fi
if ! git diff --quiet HEAD -- src scripts bench; then echo "metric-error: uncommitted changes in src/scripts/bench" >&2; exit 1; fi
cargo build --release -q || exit 1
mkdir -p target/loop
run() {  # tag binary
  "$2" --bench bench/loop.json > "target/loop/$1.out" 2> "target/loop/$1.err"
  code=$?
  case $code in 0|3) ;; *) echo "metric-error: $1 exited $code" >&2; exit 1 ;; esac
}
run base1 "$BASE"
run cand1 target/release/gibbs
run cand2 target/release/gibbs
run base2 "$BASE"
python3 scripts/loop_score.py target/loop
