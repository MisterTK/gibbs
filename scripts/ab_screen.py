#!/usr/bin/env python3
"""Paired alternating A/B screen of two gibbs binaries on selected bench/loop.json entries.

Usage: scripts/ab_screen.py BASE_BIN CAND_BIN [--rounds N] [--models i,j,...] [--env K=V ...]

For every round and model the two binaries run back to back (order alternates between rounds), so a foreign
GPU burst hits both arms of a pair about equally; the verdict per model is the median of the per-pair
cand/base tok/s ratios plus how many pairs favour the candidate. This is a SCREEN for machines where the
frozen ABBA verify (scripts/loop_verify.sh) keeps aborting on contention: it never replaces the keep rule.
Prints one JSON line per model and a summary of host/gpu/wall medians.
"""
import argparse, json, os, statistics, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("cand")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--models", default="0", help="comma-separated indexes into bench/loop.json models")
ap.add_argument("--suite", default="bench/loop.json")
ap.add_argument("--env", nargs="*", default=[], help="K=V pairs applied to the CANDIDATE only")
ap.add_argument("--base-env", nargs="*", default=[], help="K=V pairs applied to the BASE only")
a = ap.parse_args()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
suite = json.load(open(os.path.join(root, a.suite)))
files = {}
for i in (int(x) for x in a.models.split(",")):
    # Flat in bench/: prompt_file resolves relative to the suite file's directory.
    p = os.path.join(root, "bench", f".screen-m{i}.json")
    json.dump({"defaults": suite.get("defaults"), "models": [suite["models"][i]]}, open(p, "w"))
    files[i] = p

def run(binary, path, extra):
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in extra))
    r = subprocess.run([binary, "--bench", path], capture_output=True, text=True, cwd=root, env=env)
    for l in r.stdout.splitlines():
        if l.startswith("{"):
            row = json.loads(l)
            if "error" in row:
                sys.exit(f"{binary}: {row['error']}")
            return row
    sys.exit(f"{binary} printed no result row (exit {r.returncode}):\n{r.stderr[-800:]}")

res = {i: {"base": [], "cand": []} for i in files}
try:
    for rnd in range(a.rounds):
        for i, p in files.items():
            order = [("base", a.base, a.base_env), ("cand", a.cand, a.env)]
            if rnd % 2:
                order.reverse()
            for tag, binary, extra in order:
                res[i][tag].append(run(binary, p, extra))
finally:
    for p in files.values():
        os.remove(p)

for i, r in res.items():
    ratios = [c["decode_tok_s"] / b["decode_tok_s"] for b, c in zip(r["base"], r["cand"])]
    med = lambda tag, k: statistics.median(x[k] for x in r[tag] if x.get(k) is not None)
    print(json.dumps({
        "model_index": i, "model": r["base"][0]["model"],
        "median_ratio": round(statistics.median(ratios), 4), "pairs_favouring_cand": sum(x > 1 for x in ratios),
        "pairs": len(ratios), "ratios": [round(x, 3) for x in ratios],
        "base_tok_s": round(med("base", "decode_tok_s"), 1), "cand_tok_s": round(med("cand", "decode_tok_s"), 1),
        "wall_ms": [round(med("base", "wall_ms"), 3), round(med("cand", "wall_ms"), 3)],
        "gpu_ms": [round(med("base", "gpu_ms"), 3), round(med("cand", "gpu_ms"), 3)],
        "host_ms": [round(med("base", "host_ms"), 3), round(med("cand", "host_ms"), 3)],
        "dispatches": [med("base", "dispatches"), med("cand", "dispatches")],
    }))
