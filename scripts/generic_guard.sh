#!/usr/bin/env bash
# No model-specific identifiers in non-test lines of src/**, ignoring everything from
# `#[cfg(test)]` to end of file and whole-file test modules.
#   scripts/generic_guard.sh --all    scan all non-test lines in src/**
#   scripts/generic_guard.sh [BASE]   lines added since BASE (default HEAD~1)
set -u
cd "$(dirname "$0")/.."
exec python3 - "${1:-HEAD~1}" <<'PY'
import pathlib, re, subprocess, sys
arg = sys.argv[1]
pat = re.compile(r"gemma|qwen|llama|mistral|smollm|granite|deepseek|gpt-oss|nemotron|\b(phi|bert|sana|ltx|flux|lfm|glm|t5)(\b|[_\d-])", re.I)
skip_file = re.compile(r"(^|/)(tests?(_[^/]*)?|testing)\.rs$")
mod_decl = re.compile(r"^(pub(\([^)]*\))?\s+)?mod\s+([A-Za-z0-9_]+)\s*;\s*$")
mod_block = re.compile(r"^(pub(\([^)]*\))?\s+)?mod\s+[A-Za-z0-9_]+\s*\{")
allow = set()
allow_path = pathlib.Path("scripts/generic_guard.allow")
if allow_path.is_file():
    allow = {l.strip() for l in allow_path.read_text().splitlines() if l.strip() and not l.startswith("#")}
test_mod_files = set()
for p in sorted(pathlib.Path("src").rglob("*.rs")):
    lines = p.read_text().splitlines()
    for i, l in enumerate(lines):
        if l.strip() == "#[cfg(test)]" and i + 1 < len(lines):
            m = mod_decl.match(lines[i + 1].strip())
            if m:
                name = m.group(3)
                base = p.parent if p.name in ("mod.rs", "lib.rs") else p.with_suffix("")
                for cand in (base / f"{name}.rs", base / name / "mod.rs", p.parent / f"{name}.rs"):
                    if cand.is_file():
                        test_mod_files.add(cand.as_posix())
def non_test_lines(p):
    if skip_file.search(p) or p in test_mod_files:
        return {}
    try:
        lines = pathlib.Path(p).read_text().splitlines()
    except OSError:
        return {}
    out = {}
    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        if l.strip() == "#[cfg(test)]":
            j = i + 1
            while j < n and lines[j].strip().startswith("#["):
                j += 1
            if j >= n:
                break
            nxt = lines[j].strip()
            if mod_decl.match(nxt):
                i = j + 1
                continue
            if l.startswith("#[cfg(test)]") and mod_block.match(nxt):
                break
            depth = 0
            while j < n:
                code = lines[j].split("//")[0].strip()
                depth += code.count("{") - code.count("}")
                j += 1
                if depth <= 0 and code.endswith(("}", "};", ";")):
                    break
            i = j
            continue
        out[i + 1] = l
        i += 1
    return out
bad = []
if arg == "--all":
    for p in sorted(pathlib.Path("src").rglob("*.rs")):
        rel = p.as_posix()
        for lineno, l in non_test_lines(rel).items():
            s = l.strip()
            if pat.search(l) and f"{rel}:{s}" not in allow:
                bad.append(f"{rel}:{lineno}: {s}")
else:
    diff = subprocess.run(["git", "diff", arg, "-U0", "--", "src"],
                          capture_output=True, text=True, check=True).stdout
    path, lineno, active = None, 0, {}
    for l in diff.splitlines():
        if l.startswith("+++ b/"):
            path = l[6:]; active = non_test_lines(path)
        elif l.startswith("@@"):
            lineno = int(re.search(r"\+(\d+)", l).group(1)) - 1
        elif l.startswith("+") and not l.startswith("+++"):
            lineno += 1
            s = l[1:].strip()
            if lineno in active and pat.search(l) and f"{path}:{s}" not in allow:
                bad.append(f"{path}:{lineno}: {s}")
if bad:
    print("GENERIC_GUARD FAILURE: model-specific identifier found:")
    print("\n".join(bad))
    sys.exit(1)
PY
