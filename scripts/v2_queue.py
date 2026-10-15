#!/usr/bin/env python3
"""Runs the autoresearch keep/revert protocol over a queue of candidate commits, unattended.

Usage: scripts/v2_queue.py QUEUE.json   (run from a clean main checkout on the research branch)

QUEUE.json: [{"id": "v2-34", "sha": "<commit to cherry-pick>", "luminal_ref": "...", "expected_pp": "0.5-0.9"}, ...]
Per candidate (frozen keep rule, autoresearch/roofline-loop-notes.md):
  (an item {"kind": "infra", "hyp": "..."} verifies HEAD itself against the old base and is kept when
  delta_pp >= -max(0.30, noise_pp); a failure stops the queue for bisection)
  cherry-pick -> scripts/verify_retry.sh (quiet gate + contention tripwires, retried) -> keep iff
  delta_pp >= max(0.30, noise_pp) and ratios_ok -> scripts/loop_guard.sh -> promote target/loop-base on pass;
  otherwise `git revert HEAD --no-edit`. Every outcome appends a row to autoresearch/roofline-loop.tsv and is
  committed. Progress is journaled in target/loop/queue.log; a done candidate is skipped on restart.
"""
import datetime, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LOG = "target/loop/queue.log"
DONE = "target/loop/queue.done"
TSV = "autoresearch/roofline-loop.tsv"


def log(msg):
    line = f"{datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def sh(*cmd, check=False, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and r.returncode:
        sys.exit(f"{' '.join(cmd)} failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r


def head():
    return sh("git", "rev-parse", "--short", "HEAD").stdout.strip()


def subject(sha):
    return sh("git", "log", "-1", "--format=%s", sha).stdout.strip()


def breakdown(v):
    m = next((m for m in v["models"] if "gemma-4-E2B-it-qat" in m["model"]), None)
    if not m:
        return "-"
    return f"E2B wall {m['wall_ms']:.2f} gpu {m['gpu_ms']:.2f} host {m['host_ms']:.2f} disp {m['dispatches']:.0f} ({m['cand_tok_s']:.1f} tok/s vs base {m['base_tok_s']:.1f})"


def row(c, commit, v, guard, status, hyp):
    f = lambda k: "-" if v is None else f"{v[k]:.2f}"
    cols = [c["id"], datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), commit,
            f("base_pct"), f("cand_pct"), f("delta_pp"), f("noise_pp"),
            "-" if v is None or v["e2b_tok_s"] is None else f"{v['e2b_tok_s']:.1f}",
            guard, status, c.get("luminal_ref", "original"), hyp, c.get("expected_pp", "-"),
            "-" if v is None else breakdown(v), "-"]
    with open(TSV, "a") as t:
        t.write("\t".join(cols) + "\n")
    sh("git", "add", TSV)
    sh("git", "commit", "-q", "-m", f"loop: {c['id']} {status}")


def promote():
    sh("cp", "target/release/gibbs", "target/loop-base/gibbs", check=True)
    open("target/loop-base/COMMIT", "w").write(head() + "\n")


def main():
    queue = json.load(open(sys.argv[1]))
    os.makedirs("target/loop", exist_ok=True)

    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    env = dict(os.environ, QUIET_MAX_PCT=os.environ.get("QUIET_MAX_PCT", "35"))
    for c in queue:
        if c["id"] in done:
            continue
        infra = c.get("kind") == "infra"  # HEAD itself, already committed: kept iff delta >= -max(0.30, noise)
        hyp = c["hyp"] if infra else subject(c["sha"])
        log(f"{c['id']}: {'verify HEAD (infra)' if infra else 'cherry-pick ' + c['sha']} ({hyp[:80]})")
        r = sh("true") if infra else sh("git", "cherry-pick", c["sha"])
        if r.returncode:
            sh("git", "cherry-pick", "--abort")
            log(f"{c['id']}: cherry-pick conflict, skipped")
            row(c, "-", None, "-", "discard", "skipped: cherry-pick conflict with the kept/discarded predecessors: " + hyp)
            open(DONE, "a").write(c["id"] + "\n")
            continue
        commit = head()
        v = None
        rc = sh("scripts/verify_retry.sh", env=env).returncode
        if rc == 0:
            v = json.load(open("target/loop/verify.json"))
        if v is None:
            if not infra:
                sh("git", "revert", "HEAD", "--no-edit", check=True)
            log(f"{c['id']}: no valid verify (contention); reverted, queue stopped (rerun to retry)")
            row(c, commit, None, "-", "unverified (contention)", hyp)
            sys.exit(2)
        bar = -max(0.30, v["noise_pp"]) if infra else max(0.30, v["noise_pp"])
        keep = v["delta_pp"] >= bar and v["ratios_ok"]
        guard = "-"
        if keep:
            guard = "pass" if sh("scripts/loop_guard.sh").returncode == 0 else "fail"
            keep = guard == "pass"
        if keep:
            promote()
            status = "keep"
        elif infra:
            log(f"{c['id']}: infra failed its bar; queue stopped for bisection")
            row(c, commit, v, guard, "infra-fail", hyp)
            sys.exit(3)
        else:
            sh("git", "revert", "HEAD", "--no-edit", check=True)
            status = "discard"
        log(f"{c['id']}: {status} delta={None if v is None else round(v['delta_pp'], 2)} noise={None if v is None else round(v['noise_pp'], 2)} guard={guard}")
        row(c, commit, v, guard, "infra" if infra else status, hyp)
        open(DONE, "a").write(c["id"] + "\n")


main()
