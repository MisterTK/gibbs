#!/usr/bin/env bash
set -euo pipefail

# Canonical Autoresearch Benchmark Entrypoint for Gibbs Engine
# Measures real end-to-end decode throughput (tok/s) and memory roofline saturation.
OUTPUT=$(cargo test --release --test engine_tests -- --nocapture --test-threads=1 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "$OUTPUT" >&2
    echo "Benchmark failed with exit code $EXIT_CODE" >&2
    exit $EXIT_CODE
fi

# Extract Qwen 3.5 9B Real Metal Decode throughput and roofline percentage
QWEN_TPS=$(echo "$OUTPUT" | sed -n 's/.*Qwen 3.5 9B Real Disk Checkpoint on Metal: \([0-9.]*\) tok\/s.*/\1/p' | tail -n 1)
QWEN_ROOFLINE=$(echo "$OUTPUT" | sed -n 's/.*, \([0-9.]*\)% of [0-9.]* GB\/s roofline.*/\1/p' | tail -n 1)

# Extract Tiled GEMV Q4_0 Roofline Saturation
TILED_Q4=$(echo "$OUTPUT" | sed -n 's/.*Tiled GEMV Q4_0: [0-9.]* GB\/s (\([0-9.]*\)%.*/\1/p' | tail -n 1)

# Extract Gemma 4 E2B Decode throughput
GEMMA_TPS=$(echo "$OUTPUT" | sed -n 's/.*Gemma 4 E2B Real Disk Checkpoint: \([0-9.]*\) tok\/s.*/\1/p' | tail -n 1)

# Extract SmolLM2 Decode throughput
SMOLLM_TPS=$(echo "$OUTPUT" | sed -n 's/.*SmolLM2-135M Real Disk Checkpoint: \([0-9.]*\) tok\/s.*/\1/p' | tail -n 1)

if [ -z "$QWEN_TPS" ]; then
    echo "Failed to parse QWEN_TPS from output:" >&2
    echo "$OUTPUT" >&2
    exit 1
fi

echo "--- Gibbs Benchmark Results ---"
echo "Qwen 3.5 9B Metal: ${QWEN_TPS} tok/s (${QWEN_ROOFLINE}% roofline)"
echo "Tiled GEMV Q4_0: ${TILED_Q4}% roofline"
echo "Gemma 4 E2B: ${GEMMA_TPS} tok/s"
echo "SmolLM2 135M: ${SMOLLM_TPS} tok/s"
echo "-------------------------------"

# Primary Metric: decode throughput in tokens per second
echo "METRIC tok_per_sec=${QWEN_TPS}"

# Secondary Metrics
if [ -n "$QWEN_ROOFLINE" ]; then
    echo "METRIC roofline_pct=${QWEN_ROOFLINE}"
fi
if [ -n "$TILED_Q4" ]; then
    echo "METRIC tiled_gemv_pct=${TILED_Q4}"
fi
if [ -n "$GEMMA_TPS" ]; then
    echo "METRIC gemma_tok_per_sec=${GEMMA_TPS}"
fi
if [ -n "$SMOLLM_TPS" ]; then
    echo "METRIC smollm_tok_per_sec=${SMOLLM_TPS}"
fi

exit 0
