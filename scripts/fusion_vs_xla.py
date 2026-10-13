#!/usr/bin/env python3
"""Phase 4.9 table: gibbs fusion (GIBBS_FUSION=full / xla) vs XLA (jax-metal) on one decode block.

Runs the ignored `fusion_block_bench` test twice (under the GPU lock) and `scripts/xla_block_bench.py`
(it takes the lock itself), then prints `variant | kernels | us | ratio vs xla` and the 4.9 verdict.
Pass: Full kernels <= XLA kernels and us <= 0.8 * XLA us; with null XLA numbers (comparator
unavailable) the in-repo XlaClass row is the comparator: us <= 0.8x and kernels <= 0.6x.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gibbs(mode):
    env = dict(os.environ, GIBBS_FUSION=mode)
    r = subprocess.run(
        ["python3", "scripts/with_gpu_lock.py", "cargo", "test", "--release", "--lib", "fusion_block_bench",
         "--", "--ignored", "--nocapture"], cwd=ROOT, env=env, capture_output=True, text=True)
    for l in r.stdout.splitlines():
        if l.startswith("FUSION_BLOCK "):
            return json.loads(l[len("FUSION_BLOCK "):])
    sys.exit(f"fusion_block_bench ({mode}) printed no FUSION_BLOCK line:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def xla():
    py = "/tmp/xla-venv/bin/python"
    if not os.path.exists(py):
        return {"xla_us_per_block": None, "xla_kernels": None}
    r = subprocess.run([py, "scripts/xla_block_bench.py"], cwd=ROOT, capture_output=True, text=True)
    for l in reversed(r.stdout.splitlines()):
        if l.startswith("{"):
            return json.loads(l)
    return {"xla_us_per_block": None, "xla_kernels": None, "error": (r.stderr or r.stdout)[-500:]}


import statistics
# Alternate the variants (Full, Xla, Xla, Full, Full, Xla) and take medians: drift cancels, and with no F item
# landed the two rows are the same plan, so their spread is the fixture's noise floor.
runs = {"full": [], "xla": []}
for mode in ("full", "xla", "xla", "full", "full", "xla"):
    runs[mode].append(gibbs(mode))
med = lambda m, k: statistics.median(r[k] for r in runs[m])
full = {"us": med("full", "us"), "run_us": med("full", "run_us"), "dispatches": runs["full"][0]["dispatches"]}
xc = {"us": med("xla", "us"), "run_us": med("xla", "run_us"), "dispatches": runs["xla"][0]["dispatches"]}
x = xla()
xla_wall, xla_k = x.get("xla_us_per_block"), x.get("xla_kernels")
# Time comparator: XLA's host wall per call when it ran, else the in-repo XlaClass row (host wall). Kernel count:
# XLA's when its HLO was readable, else the XlaClass dispatch count.
ref_us, ref_name = (xla_wall, "XLA jax-metal") if xla_wall is not None else (xc["run_us"], "XlaClass (in-repo)")
ref_k, k_name = (xla_k, "XLA") if xla_k is not None else (xc["dispatches"], "XlaClass")
noise = abs(full["run_us"] - xc["run_us"]) / xc["run_us"] * 100
print(f"variant | kernels | gpu us | host-wall us | wall ratio vs {ref_name}")
for n, k, g, w in [("gibbs Full", full["dispatches"], full["us"], full["run_us"]),
                   ("gibbs XlaClass", xc["dispatches"], xc["us"], xc["run_us"]),
                   ("XLA jax-metal", xla_k, None, xla_wall)]:
    f = lambda v: "n/a" if v is None else f"{v:.1f}"
    print(f"{n} | {k if k is not None else 'n/a'} | {f(g)} | {f(w)} | {'n/a' if w is None else f'{w / ref_us:.2f}'}")
print(f"fixture noise (Full vs XlaClass wall, medians of 3 alternated runs): {noise:.1f}%")
ok = full["run_us"] <= 0.8 * ref_us and full["dispatches"] <= 0.6 * ref_k
print(f"4.9 verdict vs {ref_name} (kernels vs {k_name}, time = host wall): {'PASS' if ok else 'NOT ACHIEVED'}")
if "error" in x:
    print("xla error:", x["error"])
