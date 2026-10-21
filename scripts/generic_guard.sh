#!/usr/bin/env bash
# No model-specific identifiers in non-test lines added to the generic code (src/graph, src/hardware,
# src/scheduler, src/arch, src/runtime), ignoring everything from `#[cfg(test)]` to end of file.
#   scripts/generic_guard.sh [BASE]   lines added since BASE (default HEAD~1; use the merge base for a branch)
set -u
cd "$(dirname "$0")/.."
exec python3 - "${1:-HEAD~1}" <<'PY'
import re, subprocess, sys
base = sys.argv[1]
pat = re.compile(r"gemma|qwen|llama|mistral|smollm|granite|deepseek|gpt-oss|nemotron|\b(phi|bert|sana|ltx|flux|lfm|glm|t5)(\b|[_\d-])", re.I)
diff = subprocess.run(["git", "diff", base, "-U0", "--", "src/graph", "src/hardware", "src/scheduler", "src/arch", "src/runtime"],
                      capture_output=True, text=True, check=True).stdout
bad, path, lineno, test_from = [], None, 0, {}
def first_test_line(p):
    # Whole-file test modules (`#[cfg(test)] mod tests_x;` in the parent).
    if re.search(r"(^|/)(tests?(_[^/]*)?|testing)\.rs$", p):
        return 0
    try:
        for i, l in enumerate(open(p), 1):
            if l.strip() == "#[cfg(test)]":
                return i
    except OSError:
        pass
    return 10**9
for l in diff.splitlines():
    if l.startswith("+++ b/"):
        path = l[6:]; test_from[path] = first_test_line(path)
    elif l.startswith("@@"):
        lineno = int(re.search(r"\+(\d+)", l).group(1)) - 1
    elif l.startswith("+") and not l.startswith("+++"):
        lineno += 1
        if lineno < test_from.get(path, 10**9) and pat.search(l):
            bad.append(f"{path}:{lineno}: {l[1:].strip()}")
if bad:
    print("GENERIC_GUARD FAILURE: model-specific identifier in the diff:")
    print("\n".join(bad))
    sys.exit(1)
PY
