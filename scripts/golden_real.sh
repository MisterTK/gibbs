#!/usr/bin/env bash
# Real-checkpoint goldens: raw-prompt top-5 logits of every cached checkpoint.
#   scripts/golden_real.sh capture BIN [SLUG...]   writes target/golden-real/<slug>.json
#   scripts/golden_real.sh check   BIN [SLUG...]   compares: prompt ids and top-5 ids exact, logits rel 1e-3
# Prompts: bench/prompts/golden.txt (one per line). Encoder-decoder rows record `--prompt` output text.
# Models missing from both caches (disk/, ~/.cache/huggingface/hub) are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
mode=$1 bin=$2; shift 2
out=target/golden-real
mkdir -p "$out"
# slug|model id|kind (probe = --probe-top5 per prompt, gen = --prompt translation)
models='
smollm2-st|HuggingFaceTB/SmolLM2-135M-Instruct|probe
smollm2-gguf|unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q2_K.gguf|probe
llama32-q4_0|unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf|probe
llama32-bnb|unsloth/Llama-3.2-1B-Instruct-bnb-4bit|probe
qwen35-2b|unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf|probe
qwen35-9b-iq2m|DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ2_M.gguf|probe
qwen3-30b-a3b|unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF|probe
gemma4-e2b-q4_0|ggml-org/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_0.gguf|probe
gemma4-e2b-q4km|unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf|probe
gemma4-e2b-qat|unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf|probe
gemma4-e4b-qat|unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q2_K_XL.gguf|probe
gemma4-12b|unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-UD-IQ2_M.gguf|probe
gemma4-26b-a4b|unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf|probe
kev-08b|ggml-org/Kev-0.8B-GGUF/Kev-0.8B-BF16.gguf|probe
tev-mlx4|gavinmroy/Tev1-0.8B-experimental-mlx-4bit|probe
qwen38-27b-nvfp4|Inferact/Qwen3.8-27B-NVFP4|probe
t5-small|google-t5/t5-small|gen
'
# The hub cache holding the repo's weights (a non-.tmp weight file), or nothing.
cache_of() {
  local repo file dir
  repo=$(echo "$1" | cut -d/ -f1-2) file=$(echo "$1" | cut -d/ -f3-)
  for c in "$PWD/disk" "$HOME/.cache/huggingface/hub"; do
    dir="$c/models--${repo//\//--}"
    if [ -n "$file" ]; then
      ls "$dir"/snapshots/*/"$file" >/dev/null 2>&1 && { echo "$c"; return; }
    else
      ls "$dir"/snapshots/*/*.safetensors >/dev/null 2>&1 && { echo "$c"; return; }
    fi
  done
}
rc=0
while IFS='|' read -r slug id kind; do
  [ -z "$slug" ] && continue
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$slug"; then continue; fi
  cache=$(cache_of "$id")
  if [ -z "$cache" ]; then echo "SKIP $slug (not cached)"; continue; fi
  tmp=$(mktemp)
  if [ "$kind" = gen ]; then
    HF_HUB_CACHE="$cache" python3 ~/.cache/gibbs/central_scheduler.py run --label "gr-$slug" -- \
      "$bin" -m "$id" --prompt "translate English to German: The house is wonderful." --max-tokens 16 \
      >"$tmp" 2>/dev/null
    python3 -c 'import json,sys; print(json.dumps({"text": open(sys.argv[1]).read().strip()}))' "$tmp" >"$tmp.json"
  else
    : >"$tmp.json"
    while IFS= read -r prompt; do
      HF_HUB_CACHE="$cache" python3 ~/.cache/gibbs/central_scheduler.py run --label "gr-$slug" -- \
        "$bin" -m "$id" --probe-top5 "$prompt" 2>/dev/null | grep '^{' >>"$tmp.json" \
        || echo '{"error": "probe failed"}' >>"$tmp.json"
    done <bench/prompts/golden.txt
  fi
  if [ "$mode" = capture ]; then
    mv "$tmp.json" "$out/$slug.json"; echo "captured $slug"
  else
    python3 - "$out/$slug.json" "$tmp.json" "$slug" <<'PY' || rc=1
import json, sys
want = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
got = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
slug, bad = sys.argv[3], []
if len(want) != len(got):
    bad.append(f"{len(got)} results vs {len(want)}")
for i, (w, g) in enumerate(zip(want, got)):
    if "text" in w:
        if w["text"] != g.get("text"): bad.append(f"text {g.get('text')!r} vs {w['text']!r}")
        continue
    if "error" in w or "error" in g:
        if w.get("error") != g.get("error"): bad.append(f"prompt {i}: error {g.get('error')} vs {w.get('error')}")
        continue
    if w["ids"] != g["ids"]: bad.append(f"prompt {i}: ids differ")
    wi, gi = [t[0] for t in w["top5"]], [t[0] for t in g["top5"]]
    if wi != gi: bad.append(f"prompt {i}: top5 ids {gi} vs {wi}")
    for (_, a), (_, b) in zip(g["top5"], w["top5"]):
        if abs(a - b) > 1e-3 * max(abs(b), 1.0): bad.append(f"prompt {i}: logit {a} vs {b}")
print(("FAIL " if bad else "ok   ") + slug + ("" if not bad else ": " + "; ".join(bad[:4])))
sys.exit(1 if bad else 0)
PY
    rm -f "$tmp.json"
  fi
  rm -f "$tmp"
done <<<"$models"
exit $rc
