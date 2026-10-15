#!/usr/bin/env bash
# Frozen autoresearch Verify: paired ABBA run of the last kept binary (base) against HEAD (cand),
# model by model (base, cand, cand, base back to back for each suite entry), so thermal/clock drift
# over the minutes a whole suite takes cancels inside each model. Prints delta_pp on success.
set -u
cd "$(dirname "$0")/.."
BASE=target/loop-base/gibbs
if [ ! -x "$BASE" ]; then echo "metric-error: $BASE missing" >&2; exit 1; fi
# Quiet machine: the matrix runner abandons its in-flight item while this file exists.
rm -f target/loop/verify.json  # an aborted run must not leave the previous verdict readable
mkdir -p target/matrix && touch target/matrix/PAUSE && trap 'rm -f target/matrix/PAUSE bench/.loop-model-*.json' EXIT
# PAUSE only abandons a bench, so stop a running model download too (precaution: an unexplained ~13 %
# slower machine state, E2B QAT 106-108 vs 122-125 tok/s, appeared in several sessions, also without it).
pkill -f 'curl.*matrix-cache' || true
sleep 5
if ! git diff --quiet HEAD -- src scripts bench; then echo "metric-error: uncommitted changes in src/scripts/bench" >&2; exit 1; fi
cargo build --release -q || exit 1
mkdir -p target/loop
rm -f target/loop/{base1,cand1,cand2,base2}.{out,err}
n=$(python3 - <<'PY'
import json
s = json.load(open("bench/loop.json"))
for i, m in enumerate(s["models"]):
    # prompt_file is relative to the suite file's directory: keep the single-model suites in bench/.
    json.dump({"defaults": s.get("defaults"), "models": [m]}, open(f"bench/.loop-model-{i}.json", "w"))
print(len(s["models"]))
PY
)
quiet() {  # wait (<= 15 s) for an idle GPU, always log the utilization seen (other apps hold the GPU
  # at 40-99 % for long stretches here, so a hard gate would never pass; the ABBA noise term carries it)
  for _ in 1 2 3 4 5; do
    u=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
    [ "${u:-0}" -lt 5 ] && break
    sleep 3
  done
  echo "before $1: device utilization ${u:-?}%" >> target/loop/contention.log
}
contended() {  # tag: abort (exit 4, nothing recorded) when E2B QAT decodes < 100 tok/s in this run
  python3 - "target/loop/$1.out" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    if l.startswith("{"):
        r = json.loads(l)
        if "gemma-4-E2B-it-qat" in r.get("model", ""):
            # First E2B QAT row only (the plain-context entry, as loop_score.py's e2b_tok_s).
            if r.get("decode_tok_s", 1e9) < 100:
                print(f"contention-aborted: {r['model']} {r['decode_tok_s']:.1f} tok/s < 100", file=sys.stderr)
                sys.exit(4)
            break
PY
}
run() {  # tag binary suite
  quiet "$1 $3"
  "$2" --bench "$3" >> "target/loop/$1.out" 2>> "target/loop/$1.err"
  code=$?
  case $code in 0|3) ;; *) echo "metric-error: $1 $3 exited $code" >&2; return 1 ;; esac
  contended "$1" || exit 4
}
status=0
for ((i = 0; i < n; i++)); do
  s=bench/.loop-model-$i.json
  run base1 "$BASE" "$s" && run cand1 target/release/gibbs "$s" && run cand2 target/release/gibbs "$s" && run base2 "$BASE" "$s" || { status=1; break; }
done
rm -f bench/.loop-model-*.json
[ $status -eq 0 ] || exit 1
python3 scripts/loop_score.py target/loop
