#!/usr/bin/env python3
"""Resumable bench runner over bench/matrix.json (Metal only).

Per item: download -> bench -> parity -> llama.cpp oracle -> record -> delete weights.
Usage: matrix_run.py [--once] [--limit N] [--summary] [--reset REGEX]
  --once      one pass over the queue, then exit (default: loop forever, re-measuring stale rows)
  --limit N   stop after N items
  --summary   print status counts, roofline percentiles, normalized error classes; exit
  --reset RE  forget state of items whose last error matches RE (they run again)
Rows go to autoresearch/matrix-results.jsonl, per-item state to autoresearch/matrix-state.json.
"""
import argparse, fcntl, glob, json, os, re, shutil, signal, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MATRIX = "bench/matrix.json"
RESULTS = "autoresearch/matrix-results.jsonl"
STATE = "autoresearch/matrix-state.json"
CACHE = "disk/matrix-cache"
REENC = os.path.expanduser("~/.cache/gibbs/reencoded")
LOCK = os.path.expanduser("~/.cache/gibbs/gpu.lock")
PROMPT = "The capital of France is"
PORT = 18091


def sh_out(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def binary():
    b = "target/loop-base/gibbs"
    return b if os.path.exists(b) else "target/release/gibbs"


def commit():
    try:
        return open("target/loop-base/COMMIT").read().strip()
    except OSError:
        return sh_out("git", "rev-parse", "HEAD")


def load(path, default):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def run(cmd, timeout, env_extra=None):
    env = dict(os.environ, HF_HUB_CACHE=os.path.abspath(CACHE), **(env_extra or {}))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), f"timeout after {timeout}s"


def snapshot_path(item):
    pat = os.path.join(CACHE, "models--" + item["repo"].replace("/", "--"), "snapshots", "*", item["file"])
    hits = glob.glob(pat)
    return hits[0] if hits else None


def oracle(item, b):
    """llama.cpp top-5 vs gibbs --probe-top5 on a raw prompt. Returns (verdict, detail)."""
    if not item["file"] or not shutil.which("llama-server"):
        return "skip", "not a GGUF item" if not item["file"] else "llama-server not on PATH"
    path = snapshot_path(item)
    if not path:
        return "skip", "snapshot not found"
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    ref = None
    with open(LOCK, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        srv = subprocess.Popen(["llama-server", "-m", path, "--port", str(PORT), "-c", "512", "-ngl", "999"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            base = f"http://127.0.0.1:{PORT}"
            for _ in range(300):
                if srv.poll() is not None:
                    return "skip", f"llama-server exited {srv.returncode} (unsupported model?)"
                try:
                    urllib.request.urlopen(base + "/health", timeout=2).read()
                    break
                except Exception:
                    time.sleep(1)
            else:
                return "skip", "llama-server not healthy after 300s"

            def post(ep, body):
                r = urllib.request.Request(base + ep, json.dumps(body).encode(), {"Content-Type": "application/json"})
                return json.loads(urllib.request.urlopen(r, timeout=120).read())
            toks = post("/tokenize", {"content": PROMPT, "add_special": True})["tokens"]
            c = post("/completion", {"prompt": PROMPT, "n_predict": 1, "temperature": 0, "n_probs": 5, "cache_prompt": False})
            cp = (c.get("completion_probabilities") or [{}])[0]
            top = cp.get("top_logprobs") or cp.get("probs")
            if not top or not all("id" in t for t in top):
                return "skip", f"no token ids in /completion response (keys: {sorted(cp)})"
            ref = ([t["id"] for t in top], toks)
        except Exception as e:
            return "skip", f"llama-server error: {e}"
        finally:
            srv.terminate()
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                srv.kill()
    # lock released: the probe takes it itself
    code, out, err = run([b, "--model", item["id"], "--probe-top5", PROMPT], 1800)
    line = next((l for l in out.splitlines() if l.startswith("{")), None)
    if code != 0 or not line:
        return "fail", f"probe exit {code}: {err.strip().splitlines()[-1] if err.strip() else 'no output'}"
    ours = json.loads(line)
    if ours["ids"] != ref[1]:
        return "tokenizer-mismatch", f"gibbs {ours['ids'][:8]} vs llama.cpp {ref[1][:8]}"
    ours5 = [t[0] for t in ours["top5"]]
    shared = len(set(ours5) & set(ref[0][:5]))
    ok = ours5[0] == ref[0][0] and shared >= 4
    return ("pass" if ok else "fail"), f"top1 gibbs {ours5[0]} llama {ref[0][0]}; shared {shared}/5"


def process(item, nxt):
    b, row = binary(), {"id": item["id"], "commit": commit(), "timestamp": sh_out("date", "-u", "+%FT%TZ"),
                        "loadavg": os.getloadavg()[0], "format": item["format"], "bytes": item["bytes"]}
    free = shutil.disk_usage(".").free
    if free < 2 * item["bytes"] + 20e9:
        row.update(status="no-space", error=f"free {free / 1e9:.0f} GB < 2 x {item['bytes'] / 1e9:.1f} GB + 20 GB")
        return row
    os.makedirs("target/matrix", exist_ok=True)
    json.dump({"defaults": {"decode_tokens": 128, "runs": 3}, "models": [{"id": item["id"], "reference": True}]},
              open("target/matrix/one.json", "w"))
    before = set(glob.glob(REENC + "/*"))
    code, out, err = run([b, "--bench", "target/matrix/one.json"], 7200)
    rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    res = next((r for r in rows if "decode_tok_s" in r), None)
    errrow = next((r for r in rows if "error" in r), None)
    row["exit"] = code
    if res:
        row.update(res)
        row["status"] = "ok"
        pcode, _, perr = run([b, "--bench-parity", "target/matrix/one.json"], 1800)
        cos = re.search(r"min cos ([\d.]+)", perr)
        row["parity"] = "ok" if pcode == 0 else "fail"
        row["parity_min_cos"] = float(cos.group(1)) if cos else None
        if pcode != 0:
            row["parity_error"] = (perr.strip().splitlines() or ["?"])[-1]
        v, d = oracle(item, b)
        row["oracle"], row["oracle_detail"] = v, d
    else:
        row["status"] = "timeout" if code == 124 else "error"
        row["error"] = (errrow or {}).get("error") or (err.strip().splitlines() or [f"exit {code}"])[-1]
    # Clean up: the repo's weights unless the next item shares them, and this run's re-encodes.
    if not (nxt and nxt["repo"] == item["repo"]):
        shutil.rmtree(os.path.join(CACHE, "models--" + item["repo"].replace("/", "--")), ignore_errors=True)
    for p in set(glob.glob(REENC + "/*")) - before:
        try:
            os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)
        except OSError:
            pass
    return row


def norm_error(msg):
    msg = re.sub(r"(/[\w.\-]+)+", "<path>", str(msg))
    return re.sub(r"\d+", "", msg)[:80]


def summary():
    state, rows = load(STATE, {}), []
    if os.path.exists(RESULTS):
        last = {}
        for l in open(RESULTS):
            r = json.loads(l)
            last[r["id"]] = r
        rows = list(last.values())
    n = len(load(MATRIX, {"items": []})["items"])
    cnt = {}
    for r in rows:
        cnt[r["status"]] = cnt.get(r["status"], 0) + 1
    print(f"items {n}, measured {len(rows)}; status: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    pcts = sorted(r["roofline_pct"] for r in rows if r.get("roofline_pct") is not None)
    if pcts:
        q = lambda f: pcts[min(len(pcts) - 1, int(f * len(pcts)))]
        print(f"roofline % p10 {q(.1):.1f} p50 {q(.5):.1f} p90 {q(.9):.1f} max {pcts[-1]:.1f}; >=90%: {sum(p >= 90 for p in pcts)}/{len(pcts)}")
    for key in ("parity", "oracle"):
        c = {}
        for r in rows:
            if key in r:
                c[r[key]] = c.get(r[key], 0) + 1
        print(f"{key}: {c}")
    classes = {}
    for r in rows:
        for e in (r.get("error"), r.get("parity_error") if r.get("parity") == "fail" else None):
            if e:
                classes.setdefault(norm_error(e), []).append(r["id"])
    print("error classes (items):")
    for k, ids in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(ids):4d}  {k}   e.g. {ids[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--reset")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if a.reset:
        state = load(STATE, {})
        gone = [k for k, v in state.items() if re.search(a.reset, v.get("error", ""))]
        for k in gone:
            del state[k]
        json.dump(state, open(STATE, "w"), indent=1)
        print(f"reset {len(gone)} items")
        return
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    done = 0
    while not stop:
        items = load(MATRIX, {"items": []})["items"]
        state, cur = load(STATE, {}), commit()
        fresh = [i for i in items if i["id"] not in state]
        stale = [i for i in items if i["id"] in state and state[i["id"]].get("last_commit") != cur]
        queue = fresh + stale
        if not queue:
            if a.once:
                break
            time.sleep(600)  # everything is current; matrix_gen.py or a new base commit adds work
            continue
        for k, item in enumerate(queue):
            if stop or (a.limit and done >= a.limit):
                break
            nxt = queue[k + 1] if k + 1 < len(queue) else None
            row = process(item, nxt)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(row) + "\n")
            state = load(STATE, {})
            old = state.get(item["id"], {})
            state[item["id"]] = {"last_commit": row["commit"], "last_status": row["status"],
                                 "roofline_pct": row.get("roofline_pct"), "attempts": old.get("attempts", 0) + 1,
                                 "error": row.get("error", "")}
            json.dump(state, open(STATE, "w"), indent=1)
            done += 1
            print(f"{row['timestamp']} {item['id']}: {row['status']} "
                  f"{row.get('decode_tok_s', 0):.1f} tok/s {row.get('roofline_pct', 0):.1f}% "
                  f"parity={row.get('parity')} oracle={row.get('oracle')} {row.get('error', '')}", flush=True)
        if a.once or (a.limit and done >= a.limit):
            break


if __name__ == "__main__":
    main()
