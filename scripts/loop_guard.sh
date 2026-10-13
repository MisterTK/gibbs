#!/usr/bin/env bash
# Frozen autoresearch guard: lib + GPU-only integration tests, bench parity, generic-code check.
set -u
cd "$(dirname "$0")/.."
python3 scripts/with_gpu_lock.py bash -c 'cargo test --release --lib -q && cargo test --release --test cpu_free_matrix --test op_coverage --test codec_parity --test gemma4_tests -q' || exit 1
target/release/gibbs --bench-parity bench/loop.json || exit 1
scripts/generic_guard.sh
