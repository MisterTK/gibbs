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


full, xc, x = gibbs("full"), gibbs("xla"), xla()
ref_us, ref_k, ref_name = x.get("xla_us_per_block"), x.get("xla_kernels"), "xla"
if ref_us is None:
    ref_us, ref_k, ref_name = xc["us"], xc["dispatches"], "xla-class (in-repo)"
rows = [("gibbs Full", full["dispatches"], full["us"]), ("gibbs XlaClass", xc["dispatches"], xc["us"]),
        ("XLA jax-metal", ref_k if x.get("xla_us_per_block") is not None else None, x.get("xla_us_per_block"))]
print("variant | kernels | us | ratio vs " + ref_name)
for n, k, us in rows:
    print(f"{n} | {k if k is not None else 'n/a'} | {us if us is not None else 'n/a'} | "
          f"{us / ref_us:.2f}" if us is not None else f"{n} | n/a | n/a | n/a")
if x.get("xla_us_per_block") is not None:
    ok = full["dispatches"] <= ref_k and full["us"] <= 0.8 * ref_us
else:
    ok = full["us"] <= 0.8 * ref_us and full["dispatches"] <= 0.6 * ref_k
print(f"4.9 verdict vs {ref_name}: {'PASS' if ok else 'NOT ACHIEVED'}"
      + ("" if x.get("xla_us_per_block") is not None else " (XLA numbers unavailable: comparator is the in-repo XlaClass policy)"))
if "error" in x:
    print("xla error:", x["error"])
