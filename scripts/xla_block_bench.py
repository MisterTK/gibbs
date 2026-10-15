#!/usr/bin/env python3
"""XLA comparator for one Gemma-4-E2B-style sliding-layer decode block (plan 4.0 step 3).

f16 dense weights (XLA cannot read GGUF), hidden 1536, 8 heads, 1 KV head, head_dim 256,
intermediate 6144, KV length 128. Prints one JSON line {"xla_us_per_block", "xla_kernels"}.
Run with /tmp/xla-venv/bin/python (jax 0.4.35 + jax-metal). Takes the GPU lock.
"""
import fcntl, json, os, re, statistics, sys, time

lock = open(os.path.expanduser("~/.cache/gibbs/gpu.lock"), "a+")
fcntl.flock(lock, fcntl.LOCK_EX)

import jax, jax.numpy as jnp, numpy as np

H, NH, NKV, HD, FF, T = 1536, 8, 1, 256, 6144, 128
dt = jnp.float16
rng = np.random.default_rng(0)
w = lambda *s: jnp.asarray(rng.standard_normal(s) / np.sqrt(s[0]), dt)
P = dict(
    g1=jnp.ones(H, dt), g2=jnp.ones(H, dt), qn=jnp.ones(HD, dt), kn=jnp.ones(HD, dt),
    wq=w(H, NH * HD), wk=w(H, NKV * HD), wv=w(H, NKV * HD), wo=w(NH * HD, H),
    wg=w(H, FF), wu=w(H, FF), wd=w(FF, H),
)


def rms(x, g, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + eps)).astype(dt) * g


def rope(x, pos):  # x [heads, HD]
    half = HD // 2
    f = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) / half))
    a = pos * f
    c, s = jnp.cos(a).astype(dt), jnp.sin(a).astype(dt)
    x1, x2 = x[:, :half], x[:, half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def block(x, kc, vc, pos, P):
    h = rms(x, P["g1"])
    q = rope(rms((h @ P["wq"]).reshape(NH, HD), P["qn"]), pos)
    k = rope(rms((h @ P["wk"]).reshape(NKV, HD), P["kn"]), pos)
    v = (h @ P["wv"]).reshape(NKV, HD)
    kc = jax.lax.dynamic_update_slice(kc, k[:, None, :], (0, pos, 0))
    vc = jax.lax.dynamic_update_slice(vc, v[:, None, :], (0, pos, 0))
    kk, vv = jnp.repeat(kc, NH // NKV, 0), jnp.repeat(vc, NH // NKV, 0)
    s = jnp.einsum("hd,htd->ht", q, kk).astype(jnp.float32) / np.sqrt(HD)
    s = jnp.where(jnp.arange(T)[None, :] <= pos, s, -1e30)
    a = jax.nn.softmax(s, -1).astype(dt)
    o = jnp.einsum("ht,htd->hd", a, vv).reshape(-1) @ P["wo"]
    x = x + o
    h2 = rms(x, P["g2"])
    m = (jax.nn.gelu(h2 @ P["wg"], approximate=True) * (h2 @ P["wu"])) @ P["wd"]
    return x + m, kc, vc


f = jax.jit(block)
x = jnp.asarray(rng.standard_normal(H), dt)
kc = jnp.asarray(rng.standard_normal((NKV, T, HD)), dt)
vc = jnp.asarray(rng.standard_normal((NKV, T, HD)), dt)
pos = jnp.int32(100)
kernels = None  # the jax-metal plugin's compiled.as_text() is None: no HLO fusion list, time is still measured
try:
    txt = f.lower(x, kc, vc, pos, P).compile().as_text()
    if txt:
        kernels = len(re.findall(r"^\s*(?:ROOT )?%?\S+ = .*\b(?:fusion|custom-call)\(", txt, re.M))
except Exception as e:
    print("as_text unavailable:", e, file=sys.stderr)
for _ in range(20):
    jax.block_until_ready(f(x, kc, vc, pos, P))
ts = []
for _ in range(200):
    t = time.perf_counter()
    jax.block_until_ready(f(x, kc, vc, pos, P))
    ts.append((time.perf_counter() - t) * 1e6)
# Chained: 200 serially dependent calls (x, kc, vc = f(x, kc, vc, pos, P)), one sync at the end. This is
# Gibbs' execution model (each block consumes the previous block's output), so it measures per-block
# latency; independent back-to-back calls would measure overlapped throughput.
chained, finite = [], True
for _ in range(5):
    xs, kcs, vcs = x, kc, vc
    t = time.perf_counter()
    for _ in range(200):
        xs, kcs, vcs = f(xs, kcs, vcs, pos, P)
    jax.block_until_ready((xs, kcs, vcs))
    chained.append((time.perf_counter() - t) * 1e6 / 200)
    finite = finite and bool(jnp.all(jnp.isfinite(xs)))
print(json.dumps({"xla_us_per_block": round(statistics.median(ts), 1), "xla_us_chained": round(min(chained), 1),
                  "xla_chain_finite": finite, "xla_kernels": kernels,
                  "backend": jax.default_backend(), "note": "wall time incl. dispatch, block_until_ready"}))
