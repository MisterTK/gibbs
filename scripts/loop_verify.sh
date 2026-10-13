#!/usr/bin/env bash
# Frozen autoresearch Verify: paired ABBA run of the last kept binary (base) against HEAD (cand),
# model by model (base, cand, cand, base back to back for each suite entry), so thermal/clock drift
# over the minutes a whole suite takes cancels inside each model. Prints delta_pp on success.
set -u
cd "$(dirname "$0")/.."
BASE=target/loop-base/gibbs
if [ ! -x "$BASE" ]; then echo "metric-error: $BASE missing" >&2; exit 1; fi
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
run() {  # tag binary suite
  "$2" --bench "$3" >> "target/loop/$1.out" 2>> "target/loop/$1.err"
  code=$?
  case $code in 0|3) ;; *) echo "metric-error: $1 $3 exited $code" >&2; return 1 ;; esac
}
status=0
for ((i = 0; i < n; i++)); do
  s=bench/.loop-model-$i.json
  run base1 "$BASE" "$s" && run cand1 target/release/gibbs "$s" && run cand2 target/release/gibbs "$s" && run base2 "$BASE" "$s" || { status=1; break; }
done
rm -f bench/.loop-model-*.json
[ $status -eq 0 ] || exit 1
python3 scripts/loop_score.py target/loop
