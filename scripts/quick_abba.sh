#!/usr/bin/env bash
# Short paired A/B, one model per process, each well under a minute.
#   scripts/quick_abba.sh ab LABEL "BASE_CMD" "CAND_CMD" ROW...
#       Per bench/loop.json row, one scheduled job runs base, cand, cand, base `--bench` legs at 64 decode
#       tokens (warm-up + 2 timed runs). CMDs are prefixes: "target/ab-base/gibbs",
#       "env GIBBS_DISABLE=pipeline target/release/gibbs". Output: target/quick/LABEL/summary.txt.
#   scripts/quick_abba.sh parity ROW...
#       `target/release/gibbs --bench-parity` on each row alone (marked reference).
set -uo pipefail
cd "$(dirname "$0")/.."
mode=$1; shift
suite() { # row out reference(0|1)
python3 - "$1" "$2" "$3" <<'PY'
import json, os, sys
row, out, ref = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
s = json.load(open("bench/loop.json"))
m = dict(s["models"][row]); m.pop("min_tok_s", None)
if "prompt_file" in m:
    m["prompt_file"] = os.path.abspath(os.path.join("bench", m["prompt_file"]))
d = dict(s["defaults"])
if ref: m["reference"] = True
else: d.update(decode_tokens=64, runs=2)
json.dump({"defaults": d, "models": [m]}, open(out, "w"))
PY
}
mkdir -p target/quick
for r in 0 1 2 3 4 5 6 7 8; do suite "$r" "target/quick/suite-$r.json" 0; done
if [ "$mode" = parity ]; then
  for row in "$@"; do
    suite "$row" "target/quick/parity-$row.json" 1
    python3 ~/.cache/gibbs/central_scheduler.py run --label "qp-$row" -- \
      target/release/gibbs --bench-parity "target/quick/parity-$row.json"
    echo "parity row $row rc=$?"
  done
  exit 0
fi
label=$1 base=$2 cand=$3; shift 3
d=target/quick/$label
rm -rf "$d" && mkdir -p "$d"
for row in "$@"; do
  for leg in base1 cand1 cand2 base2; do
    case $leg in base*) c=$base ;; *) c=$cand ;; esac
    echo "$c --bench target/quick/suite-$row.json > $d/r$row-$leg.out 2> $d/r$row-$leg.err"
  done > "$d/row-$row.sh"
  python3 ~/.cache/gibbs/central_scheduler.py run --label "qa-$label-$row" -- bash "$d/row-$row.sh"
done
python3 - "$d" "$@" <<'PY' | tee "$d/summary.txt"
import json, math, sys
d, rows = sys.argv[1], sys.argv[2:]
ratios, spreads = [], []
for row in rows:
    leg = {}
    for t in ("base1", "cand1", "cand2", "base2"):
        js = [json.loads(l) for l in open(f"{d}/r{row}-{t}.out") if l.startswith("{")]
        leg[t] = js[0] if js else {"error": "no result row"}
    bad = [t for t in leg if "error" in leg[t]]
    if bad:
        print(f"row {row}: ERROR {[(t, leg[t]['error']) for t in bad]}"); ratios.append(0.0); spreads.append(1.0); continue
    tok = {t: leg[t]["decode_tok_s"] for t in leg}
    b, c = (tok["base1"] + tok["base2"]) / 2, (tok["cand1"] + tok["cand2"]) / 2
    sp = abs(tok["base1"] - tok["base2"]) / b
    ratios.append(c / b); spreads.append(sp)
    g = lambda t, k: leg[t].get(k)
    print(f"row {row} {leg['base1']['model'][-46:]:46s} base {b:7.1f} cand {c:7.1f} x{c/b:.3f} spread {sp:.3f} "
          f"gpu {g('base1','gpu_ms')}->{g('cand1','gpu_ms')} host {g('base1','host_ms')}->{g('cand1','host_ms')} "
          f"disp {g('base1','dispatches')}->{g('cand1','dispatches')} rss {g('base1','rss_gb')}->{g('cand1','rss_gb')}")
gm = math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / len(ratios))
print(f"SUMMARY geomean x{gm:.3f} min x{min(ratios):.3f} max_spread {max(spreads):.3f} rows {len(rows)}")
PY
