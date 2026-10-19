#!/usr/bin/env bash
# Sprint 3 Phase 0 GPU probes (plans/research-sprint-3.md, P0.1/P0.3/P0.4/P0.5). Every GPU command
# runs through the central scheduler on a quiet GPU; outputs land in target/sprint3/.
#   scripts/sprint3_probes.sh p01   kernel profile, one-model suites for M0 M1 M2 M7 M8
#   scripts/sprint3_probes.sh p03   llama-bench refresh on the same GGUFs (M0-M4, M7, M8 at depth)
#   scripts/sprint3_probes.sh p04   Wave 11 ABBA: scripts/loop_verify.sh, 5de3057 (cand) vs 4ff5e60 (base)
#   scripts/sprint3_probes.sh p05   Gibbs on bench/reference-7b.json (Llama 2 7B Q4_0 / Q8_0)
# P0.2 (plan dumps) needs no GPU: cargo run --release --example plan_dump -- bench/loop.json <row> --ctx <n>
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=$ROOT/target/sprint3
SUITE=bench/loop.json
HUB=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
mkdir -p "$OUT"

sched() {  # label cmd...: one exclusive, quiet-GPU job
  local label=$1
  shift
  python3 "$HOME/.cache/gibbs/central_scheduler.py" run --quiet --label "s3-$label" -- "$@"
}

suite_row() {  # row field: a field of bench/loop.json's models[row]
  python3 - "$SUITE" "$1" "$2" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["models"][int(sys.argv[2])].get(sys.argv[3], ""))
PY
}

one_model_suite() {  # row tag: target/sprint3/suite-<tag>.json holding only models[row]
  local path=$OUT/suite-$2.json
  python3 - "$SUITE" "$1" "$path" <<'PY'
import json, os, sys
suite, row, path = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3]
m = dict(suite["models"][row])
if "prompt_file" in m:  # relative to the suite's directory, which is no longer bench/
    m["prompt_file"] = os.path.abspath(os.path.join(os.path.dirname(sys.argv[1]), m["prompt_file"]))
json.dump({"defaults": suite.get("defaults"), "models": [m]}, open(path, "w"), indent=2)
PY
  echo "$path"
}

gguf() {  # row: the cached GGUF file of models[row] ("org/repo/file.gguf")
  local id org repo file f
  id=$(suite_row "$1" id)
  org=${id%%/*}
  repo=${id#*/}
  file=${repo#*/}
  repo=${repo%%/*}
  for f in "$HUB/models--$org--$repo"/snapshots/*/"$file"; do
    [ -e "$f" ] && { echo "$f"; return; }
  done
  echo "missing $id under $HUB" >&2
  return 1
}

p01() {
  cargo build --release -q
  local spec tag row suite rc
  for spec in m0:0 m1:1 m2:2 m7:7 m8:8; do
    tag=${spec%%:*}
    row=${spec#*:}
    suite=$(one_model_suite "$row" "$tag")
    rc=0
    echo "commit $(git rev-parse HEAD)" > "$OUT/p01-$tag.log"
    sched "p01-$tag" env GIBBS_PROFILE_KERNELS=1 GIBBS_PROFILE_ROWS=40 \
      target/release/gibbs --bench "$suite" >> "$OUT/p01-$tag.log" 2>&1 || rc=$?
    echo "p01 $tag: exit $rc -> $OUT/p01-$tag.log"
  done
}

p03() {
  # The ctx4k prompt (bench/prompts/long-4k.txt, chat-templated) is 5777 tokens: every ctx4k row of the
  # 2026-10-18 loop_verify logs (target/loop/{base,cand}{1,2}.out) prints "Prefill: 5777 tokens".
  local depth=5777 spec tag row f rc
  brew list --versions llama.cpp ggml > "$OUT/p03-versions.txt" 2>&1 || true
  sched p03-help llama-bench --help > "$OUT/p03-help.txt" 2>&1 || true
  if ! grep -qE -- '-d, --n-depth' "$OUT/p03-help.txt"; then
    echo "p03: llama-bench has no -d/--n-depth (see $OUT/p03-help.txt)" >&2
    return 1
  fi
  for spec in m0:0 m1:1 m2:2 m3:3 m4:4 m7:7 m8:8; do
    tag=${spec%%:*}
    row=${spec#*:}
    f=$(gguf "$row")
    rc=0
    if [ "$tag" = m8 ]; then
      sched "p03-$tag" llama-bench -m "$f" -p 0 -n 128 -r 3 -d "$depth" > "$OUT/p03-$tag.md" 2>&1 || rc=$?
    else
      sched "p03-$tag" llama-bench -m "$f" -p 0 -n 128 -r 3 > "$OUT/p03-$tag.md" 2>&1 || rc=$?
    fi
    echo "p03 $tag: exit $rc -> $OUT/p03-$tag.md"
  done
}

p04() {
  # loop_verify.sh runs HEAD's build (cand) against target/loop-base/gibbs (base), each model
  # base, cand, cand, base. Both commits get a detached worktree here; the base binary is copied in.
  # loop_verify.sh, loop_score.py and bench/loop.json are identical at 5de3057 and 9045f1c.
  # Its PAUSE file lands in the cand worktree, so this checkout's matrix runner is paused here too.
  local base=$OUT/p04-base cand=$OUT/p04-cand rc=0
  [ -d "$base" ] || git worktree add --detach "$base" 4ff5e60
  [ -d "$cand" ] || git worktree add --detach "$cand" 5de3057
  (cd "$base" && cargo build --release -q)
  mkdir -p "$cand/target/loop-base"
  cp "$base/target/release/gibbs" "$cand/target/loop-base/gibbs"
  (cd "$cand" && cargo build --release -q)  # loop_verify rebuilds too; this keeps the GPU slot short
  mkdir -p "$ROOT/target/matrix" && touch "$ROOT/target/matrix/PAUSE"
  trap 'rm -f "$ROOT/target/matrix/PAUSE"' EXIT
  echo "base 4ff5e60 cand 5de3057" > "$OUT/p04-verify.log"
  sched p04-abba bash -c "cd '$cand' && scripts/loop_verify.sh" >> "$OUT/p04-verify.log" 2>&1 || rc=$?
  rm -rf "$OUT/p04-loop"
  cp -R "$cand/target/loop" "$OUT/p04-loop"
  echo "p04: exit $rc -> $OUT/p04-verify.log, $OUT/p04-loop/"
}

p05() {
  cargo build --release -q
  local rc=0
  # The first run downloads both GGUFs (3.8 + 7.2 GB) inside the scheduler slot.
  echo "commit $(git rev-parse HEAD)" > "$OUT/p05-ref7b.log"
  sched p05-ref7b target/release/gibbs --bench bench/reference-7b.json >> "$OUT/p05-ref7b.log" 2>&1 || rc=$?
  echo "p05: exit $rc -> $OUT/p05-ref7b.log"
}

case "${1:-}" in
  p01 | p03 | p04 | p05) "$1" ;;
  *)
    echo "usage: $0 p01|p03|p04|p05" >&2
    exit 2
    ;;
esac
