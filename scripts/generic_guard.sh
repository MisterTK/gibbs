#!/usr/bin/env bash
set -u
diff_lines=$(git diff HEAD~1 -U0 -- src/graph | grep '^+[^+]' | grep -iE 'gemma|qwen|llama|mistral|phi|smollm|\bt5\b|bert|flux|sana|ltx' || true)
if [ -n "$diff_lines" ]; then
    echo "GENERIC_GUARD FAILURE: model-specific identifier in src/graph diff:"
    echo "$diff_lines"
    exit 1
fi
exit 0
