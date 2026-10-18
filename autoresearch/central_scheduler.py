#!/usr/bin/env python3
"""Centralized machine-wide FIFO resource scheduler for Gibbs builds, tests, screens, and benchmarks.

Ensures strictly ONE system-resource-intensive job (cargo build/test, shape_rates, ab_screen.py,
loop_guard.sh, verify_when_quiet.sh, gibbs --bench) runs at a time across the main checkout and any
worktrees. Holds a global lock at `~/.cache/gibbs/sched.lock` (without touching `~/.cache/gibbs/gpu.lock`,
which `gibbs --bench` and `with_gpu_lock.py` acquire internally), executes in the caller's `os.getcwd()`,
caps `QUIET_MAX_PCT` at `<= 40` (`min(floor, 34) + 5`, min 20) so foreign GPU load is never mistaken
for the compositor floor, and journals every job in `~/.cache/gibbs/sched.log`.
"""
import argparse
import datetime
import fcntl
import json
import os
import re
import subprocess
import sys
import time

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gibbs")
RIPSTER_HEAVY_LOCK = "/tmp/ripster-lane-heavy.lock"
SCHED_LOCK = os.path.join(CACHE_DIR, "sched.lock")
SCHED_LOG = os.path.join(CACHE_DIR, "sched.log")
COMPOSITOR_FLOOR_CAP = 34

def now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(event: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(SCHED_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


def sample_gpu_util() -> int:
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        return int(m.group(1)) if m else 100
    except Exception:
        return 100


def measure_idle_floor(samples: int = 3, delay_s: float = 0.5) -> int:
    vals = []
    for i in range(samples):
        vals.append(sample_gpu_util())
        if i + 1 < samples:
            time.sleep(delay_s)
    return min(vals) if vals else 100


def compute_quiet_max(raw_floor: int | None = None) -> int:
    if "QUIET_MAX_PCT" in os.environ:
        return int(os.environ["QUIET_MAX_PCT"])
    if raw_floor is None:
        return 20
    return max(20, min(raw_floor, COMPOSITOR_FLOOR_CAP) + 6)


def check_quiet_window(quiet_max: int, required_ok: int = 5, delay_s: float = 0.4) -> tuple[int, bool]:
    last = sample_gpu_util()
    for i in range(required_ok):
        if i > 0:
            time.sleep(delay_s)
            last = sample_gpu_util()
        if last > quiet_max:
            return last, False
    return last, True


def wait_until_quiet(quiet_max: int, required_ok: int = 5, timeout_s: int = 300) -> tuple[int, float, bool]:
    t0 = time.time()
    ok = 0
    last = sample_gpu_util()
    while ok < required_ok and (time.time() - t0) < timeout_s:
        last = sample_gpu_util()
        if last <= quiet_max:
            ok += 1
            if ok >= required_ok:
                break
            time.sleep(0.4)
        else:
            ok = 0
            time.sleep(2.0)
    return last, round(time.time() - t0, 3), (ok >= required_ok)


def acquire_locks() -> tuple[int, int]:
    ripster_fd = os.open(RIPSTER_HEAVY_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        sched_fd = os.open(SCHED_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    except BaseException:
        os.close(ripster_fd)
        raise
    try:
        fcntl.flock(ripster_fd, fcntl.LOCK_EX)
        fcntl.flock(sched_fd, fcntl.LOCK_EX)
    except BaseException:
        release_locks(ripster_fd, sched_fd)
        raise
    return ripster_fd, sched_fd


def release_locks(ripster_fd: int, sched_fd: int) -> None:
    for fd in (sched_fd, ripster_fd):
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

def run_job(args: argparse.Namespace) -> int:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.exit("central_scheduler: empty command")

    cwd = os.getcwd()
    wait_t0 = time.time()
    if getattr(args, "quiet", False):
        quiet_t0 = time.time()
        wait_s = 0.0
        raw_floor = 100
        quiet_max = 20
        is_quiet = False
        ripster_fd = sched_fd = -1
        while True:
            lock_t0 = time.time()
            ripster_fd, sched_fd = acquire_locks()
            wait_s = round(wait_s + (time.time() - lock_t0), 3)
            try:
                raw_floor = measure_idle_floor(samples=args.floor_samples, delay_s=0.4)
                default_max = args.quiet_max if args.quiet_max is not None else compute_quiet_max(raw_floor)
                quiet_max = int(os.environ.get("QUIET_MAX_PCT", str(default_max)))
                raw_floor, is_quiet = check_quiet_window(
                    quiet_max=quiet_max,
                    required_ok=args.quiet_samples,
                    delay_s=0.4,
                )
            except BaseException:
                release_locks(ripster_fd, sched_fd)
                raise
            if is_quiet:
                break
            release_locks(ripster_fd, sched_fd)
            if (time.time() - quiet_t0) >= args.quiet_timeout:
                break
            time.sleep(15.0)
            if (time.time() - quiet_t0) >= args.quiet_timeout:
                break
        quiet_wait_s = round(time.time() - quiet_t0, 3)
        if not is_quiet:
            fail_ev = {
                "ts": now_iso(),
                "event": "quiet_timeout",
                "label": args.label or " ".join(cmd[:3]),
                "cwd": cwd,
                "quiet_wait_s": quiet_wait_s,
                "gpu_raw_pct": raw_floor,
                "quiet_max_pct": quiet_max,
            }
            log_event(fail_ev)
            print(
                f"central_scheduler: GPU busy ({raw_floor}% > {quiet_max}%) after {quiet_wait_s}s; aborting",
                file=sys.stderr,
            )
            return 2
    else:
        ripster_fd, sched_fd = acquire_locks()
        wait_s = round(time.time() - wait_t0, 3)
        quiet_wait_s = 0.0

    try:
        if not getattr(args, "quiet", False):
            raw_floor = measure_idle_floor(samples=args.floor_samples, delay_s=0.4)
            default_max = args.quiet_max if args.quiet_max is not None else compute_quiet_max(raw_floor)
            quiet_max = int(os.environ.get("QUIET_MAX_PCT", str(default_max)))
        env = dict(os.environ)
        env["QUIET_MAX_PCT"] = str(quiet_max)
        env.setdefault("GIBBS_TUNE_BUSY_PCT", str(quiet_max))

        start_ev = {
            "ts": now_iso(),
            "event": "start",
            "label": args.label or " ".join(cmd[:3]),
            "cwd": cwd,
            "cmd": cmd,
            "wait_s": wait_s,
            "quiet_wait_s": quiet_wait_s,
            "gpu_raw_pct": raw_floor,
            "quiet_max_pct": int(env["QUIET_MAX_PCT"]),
        }
        log_event(start_ev)
        run_t0 = time.time()
        is_cargo_test = "cargo test" in " ".join(cmd)
        total_passed = 0
        saw_test_result = 0
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if is_cargo_test:
                m = re.search(r"test result: ok\. (\d+) passed;", line)
                if m:
                    total_passed += int(m.group(1))
                    saw_test_result += 1
        rc = proc.wait()
        if (
            is_cargo_test
            and saw_test_result > 0
            and total_passed == 0
            and not getattr(args, "allow_zero_tests", False)
        ):
            print(
                "central_scheduler: ERROR: cargo test ran 0 passed tests across all binaries (check --lib vs --test filter)",
                file=sys.stderr,
            )
            rc = 3
        dur_s = round(time.time() - run_t0, 3)
        end_ev = {
            "ts": now_iso(),
            "event": "done",
            "label": args.label or " ".join(cmd[:3]),
            "cwd": cwd,
            "rc": rc,
            "dur_s": dur_s,
            "gpu_after_pct": sample_gpu_util(),
        }
        log_event(end_ev)
        return rc
    finally:
        release_locks(ripster_fd, sched_fd)


def show_status(_args: argparse.Namespace) -> int:
    raw_floor = measure_idle_floor(samples=2, delay_s=0.4)
    print(
        f"gpu_raw_pct={raw_floor} recommended_QUIET_MAX_PCT={compute_quiet_max(raw_floor)} lock={SCHED_LOCK}"
    )
    if os.path.exists(SCHED_LOG):
        lines = open(SCHED_LOG).read().splitlines()[-10:]
        for l in lines:
            print(l)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Centralized single-job resource scheduler")
    sub = ap.add_subparsers(dest="sub", required=True)

    p_run = sub.add_parser("run", help="Run a resource-intensive command exclusively")
    p_run.add_argument("--label", default="", help="Job label for sched.log")
    p_run.add_argument("--floor-samples", type=int, default=2)
    p_run.add_argument("--quiet", action="store_true", help="Wait for quiet GPU before starting")
    p_run.add_argument("--quiet-samples", type=int, default=5)
    p_run.add_argument("--quiet-max", type=int, default=None, help="Max GPU utilization %% for --quiet (default: auto <=40 from idle floor)")
    p_run.add_argument("--quiet-timeout", type=int, default=3600)
    p_run.add_argument("--allow-zero-tests", action="store_true", help="Allow 'running 0 tests' in cargo test")
    p_run.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status", help="Show GPU idle floor and recent scheduled jobs")
    args = ap.parse_args()
    if args.sub == "run":
        return run_job(args)
    return show_status(args)


if __name__ == "__main__":
    sys.exit(main())
