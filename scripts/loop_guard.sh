#!/usr/bin/env bash
# Frozen autoresearch guard: full lib tests, GPU-only integration tests, bench parity, generic-code check.
set -u
cargo test --lib -q || exit 1
if [ -f tests/cpu_free_matrix.rs ]; then
  cargo test --test cpu_free_matrix --test op_coverage -q || exit 1
fi
target/release/gibbs --bench-parity bench/loop.json || exit 1
scripts/generic_guard.sh
