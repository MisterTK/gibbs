#!/usr/bin/env python3
"""Run argv while holding the engine's GPU flock ($HOME/.cache/gibbs/gpu.lock); pass the exit code through.

Compatible with the Rust side (`hardware::gpu_lock`, std File::lock = flock(2))."""
import fcntl, os, subprocess, sys

d = os.path.join(os.environ["HOME"], ".cache", "gibbs")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "gpu.lock"), "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    sys.exit(subprocess.call(sys.argv[1:]))
