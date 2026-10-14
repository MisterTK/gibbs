#!/usr/bin/env python3
"""Score an ABBA verify directory: prints delta_pp (two decimals), writes verify.json.

delta_pp = geomean roofline % of the candidate runs minus that of the base runs, where every
run is scored against the base binary's measured ceiling (the host profile is re-measured per
kernel-source hash, so each binary's own ceiling differs by run-to-run noise)."""
import json, math, sys, os

d = sys.argv[1]

def slug_rows(tag):
    # Rows keyed by order: the same suite yields the same sequence (names may repeat across variants).
    with open(os.path.join(d, tag + ".out")) as f:
        rs = [json.loads(l) for l in f if l.startswith("{")]
    for r in rs:
        if "error" in r:
            sys.exit(f"{tag}: {r['model']}: {r['error']}")
    if not rs:
        sys.exit(f"{tag}: no result rows")
    return rs

runs = {t: slug_rows(t) for t in ("base1", "cand1", "cand2", "base2")}
n = len(runs["base1"])
if any(len(v) != n for v in runs.values()):
    sys.exit("runs have different numbers of rows")
for i in range(n):
    if len({runs[t][i]["model"] for t in runs}) != 1:
        sys.exit(f"row {i}: model order differs between runs")

ceiling = [(runs["base1"][i]["ceiling_gbps"] + runs["base2"][i]["ceiling_gbps"]) / 2 for i in range(n)]

def pct(tag, i):
    r = runs[tag][i]
    return r["decode_tok_s"] * r["bytes_per_token"] / 1e9 / ceiling[i] * 100

def geomean(tag):
    return math.exp(sum(math.log(max(pct(tag, i), 0.1)) for i in range(n)) / n)

g = {t: geomean(t) for t in runs}
base = (g["base1"] + g["base2"]) / 2
cand = (g["cand1"] + g["cand2"]) / 2
mean_tok = lambda ts, i: sum(runs[t][i]["decode_tok_s"] for t in ts) / len(ts)
models = []
for i in range(n):
    b, c = mean_tok(("base1", "base2"), i), mean_tok(("cand1", "cand2"), i)
    r = runs["cand1"][i]
    # Separate processes of one binary differ by several % on small models (buffer placement), so a
    # model's allowed regression is 3% or twice its own base1/base2 spread, whichever is larger.
    spread = abs(runs["base1"][i]["decode_tok_s"] - runs["base2"][i]["decode_tok_s"]) / b
    allowed = max(0.03, 2 * spread)
    models.append({
        "model": r["model"], "base_tok_s": b, "cand_tok_s": c, "ratio": c / b,
        "base_spread": spread, "ratio_ok": c / b >= 1 - allowed,
        "cand_pct": sum(pct(t, i) for t in ("cand1", "cand2")) / 2,
        "wall_ms": r["wall_ms"], "gpu_ms": r.get("gpu_ms"), "run_ms": r.get("run_ms"),
        "host_ms": r.get("host_ms"), "weights_ms": r["weights_ms"], "dispatches": r.get("dispatches"),
    })
e2b = next((m["cand_tok_s"] for m in models if "gemma-4-E2B-it-qat" in m["model"]), None)
v = {"base_pct": base, "cand_pct": cand, "delta_pp": cand - base, "noise_pp": abs(g["base1"] - g["base2"]),
     "e2b_tok_s": e2b, "min_ratio": min(m["ratio"] for m in models),
     "ratios_ok": all(m["ratio_ok"] for m in models), "models": models}
# Contention tripwire: a machine this slow invalidates the run; exit 4, write nothing.
if base < 35 or (e2b is not None and e2b < 100):
    print(f"contention-aborted: base_pct {base:.1f}, e2b_tok_s {e2b}", file=sys.stderr)
    sys.exit(4)
json.dump(v, open(os.path.join(d, "verify.json"), "w"), indent=1)
print(f"{v['delta_pp']:.2f}")
