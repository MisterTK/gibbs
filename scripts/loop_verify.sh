#!/usr/bin/env bash
set -u
cargo build --release -q || exit 1
target/release/gibbs --bench bench/loop.json > target/loop-bench.out
code=$?
case $code in 0|3) ;; *) exit "$code" ;; esac   # 3 = a floor missed: still a valid measurement
sed -n 's/^METRIC geomean_roofline_pct=//p' target/loop-bench.out
