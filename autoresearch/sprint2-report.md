# Research Sprint 2: Status Report (re-measured 2026-10-12)

Supersedes the earlier report, which claimed 496 passing tests and "100% Metal lowering"; both claims were false on the tree it described (2 scheduler parity tests failed; `tests/metal_coverage.rs` printed a hardcoded table).

## Tree state
- HEAD after reverting iteration 1 (`bb32ce2`, sliced-gate_up megakernel fusion): `cargo test --lib -q` → `496 passed; 0 failed; 3 ignored`.
- Iteration 1 broke `scheduler::tests::test_program_scheduler_metal_parity` and `…prefill_and_decode_parity`; its old guard (`cargo test --lib -q graph::`) never ran them. Guard is now `scripts/loop_guard.sh`.

## Loop (`autoresearch/sprint2.tsv`)
- Row 0 baseline `geomean_roofline_pct` 19.29; row 1 (bb32ce2) 19.58, later discarded; row 2 (revert 268dc28) 23.27.
- Target ≥ 90 is not met. 38 of 40 iterations unused.

## `target/release/gibbs --bench bench/loop.json` (this tree)
```
{"model":"unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf","file":"/Users/keas/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-GGUF/snapshots/main/config.json","codec_mix":"native","bytes_per_token":2186186784,"ceiling_gbps":258.2805895855427,"decode_tok_s":6.040531018736449,"roofline_tok_s":118.1420505675981,"roofline_pct":5.112939033744128,"gpu_ms":165.54835939062502,"encode_us":10.0,"prefill_tok_s":0.0,"rss_gb":7.67664128}
{"model":"unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf","file":"/Users/keas/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/main/config.json","codec_mix":"native","bytes_per_token":3106738272,"ceiling_gbps":258.2805895855427,"decode_tok_s":73.79879125790035,"roofline_tok_s":83.13561264987781,"roofline_pct":88.76916751513085,"gpu_ms":13.5503574375,"encode_us":10.0,"prefill_tok_s":0.0,"rss_gb":7.67664128}
{"model":"unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf","file":"unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf","codec_mix":"native","bytes_per_token":1339752704,"ceiling_gbps":258.2805895855427,"decode_tok_s":68.06220931447946,"roofline_tok_s":192.78228647302862,"roofline_pct":35.305219457495,"gpu_ms":14.692441078125,"encode_us":10.0,"prefill_tok_s":0.0,"rss_gb":8.909225984}
{"model":"DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ2_M.gguf","file":"DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ2_M.gguf","codec_mix":"native","bytes_per_token":4942441760,"ceiling_gbps":258.2805895855427,"decode_tok_s":35.38255451114231,"roofline_tok_s":52.25769005026024,"roofline_pct":67.70784257228398,"gpu_ms":28.262515632812498,"encode_us":10.0,"prefill_tok_s":0.0,"rss_gb":8.909225984}
{"model":"unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf","file":"/Users/keas/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/main/config.json","codec_mix":"native","bytes_per_token":773025824,"ceiling_gbps":258.2805895855427,"decode_tok_s":28.479560045762845,"roofline_tok_s":334.11637951378805,"roofline_pct":8.523844322510257,"gpu_ms":35.11290196875,"encode_us":10.0,"prefill_tok_s":0.0,"rss_gb":14.092353536}
{"model":"HuggingFaceTB/SmolLM2-135M-Instruct","file":"/Users/keas/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/config.json","codec_mix":"native","bytes_per_token":269060552,"ceiling_gbps":258.2805895855427,"decode_tok_s":164.83855981889602,"roofline_tok_s":959.9348089702228,"roofline_pct":17.171849408787228,"gpu_ms":6.06654171875,"encode_us":10.0,"prefill_tok_s":0.0,"rss_gb":14.092353536}

METRIC geomean_roofline_pct=23.27
METRIC min_roofline_pct=5.11
METRIC gemma-4-E2B-it-qat-UD-Q2_K_XL_tok_s=6.0
METRIC gemma-4-E2B-it-Q4_K_M_tok_s=73.8
METRIC Qwen3.5-2B-UD-Q4_K_XL_tok_s=68.1
METRIC Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ2_M_tok_s=35.4
METRIC Llama-3.2-1B-Instruct-Q4_0_tok_s=28.5
METRIC SmolLM2-135M-Instruct_tok_s=164.8
```

## Known gaps (verified by reading the code this session)
- Codecs for Q4_K, Q2_K, IQ4_XS, IQ2_M, IQ3_S, TQ2_0 in `src/graph/codec.rs` are stubs; `src/graph/codegen.rs` is not called by `GraphProgram::compile`.
- Many families run on the CPU executor by default; see `plans/research-sprint-2.md` "Audit addendum" for the full table.
