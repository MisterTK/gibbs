# Sprint 2 Engineering Notes

## E4B baseline
- **Command**: `/usr/bin/time -l target/release/gibbs --model unsloth/gemma-4-E4B-it-qat-GGUF --prompt "The capital of France is" --max-tokens 4`
- **Compiled GraphProgram**: 173 segments, primary device: Metal
- **CPU segment op list**: 86 interleaved CPU segments! Every layer falls back to CPU for `Rope`, `CachedAttention`, `GeGlu`, `Add`, and `Slice`.
- **Root cause of crawl**: Extreme dispatch bouncing (173 segments = 86 round trips between Metal and CPU per forward pass, copying tensors back and forth across PCIe/memory abstractions, plus dequantizing TQ2_0 into massive memory).
- **Peak RSS**: 23,194,389,440 bytes (~23.2 GB).
- **Cycles elapsed**: 555,135,306,239 cycles in 153.6 seconds.
