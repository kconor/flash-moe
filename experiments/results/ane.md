# Apple Neural Engine (ANE) for Flash-MoE Inference

## Context

The M4 Pro has a 16-core ANE (~38 TOPS) sitting completely idle during inference. The GPU achieves only ~22 GB/s effective bandwidth (8% of 273 GB/s bus). Could the ANE handle some of the matmul work in parallel with the GPU, especially during the 4ms SSD I/O window where the GPU is idle?

## ANE Capabilities and Constraints

**Access**: Only via Core ML (public) or private `_ANECompiler` APIs (reverse-engineered, unsupported). No custom Metal-like kernels.

**Supported ops**: matmul (as 1x1 conv2d), elementwise, ReLU/SiLU/sigmoid, layer norm, softmax, fused SDPA (iOS 18+). INT8 and FP16 only.

**NOT supported**: Custom dequant kernels, dynamic control flow (top-K routing), gather/scatter, variable-length operations.

**Data format**: FP16 in BC1S layout `[batch, channels, 1, seq]`. All `nn.Linear` must be expressed as `nn.Conv2d(kernel_size=1)`. 64-byte channel alignment.

**Throughput**: ~38 TOPS peak but only 5-9% utilization on real workloads. Memory-bandwidth-bound workloads (like our matvec) don't benefit from TOPS.

**Concurrent with GPU**: Yes, but shares the memory bus. Same contention issue as SSD DMA.

## What Could Run on ANE

The only viable candidates are operations that:
- Are standard ops (matmul, norm, activation) — no custom dequant
- Use FP16 weights — not 4-bit packed uint32
- Don't require dynamic routing

**Possible candidates:**
1. **Non-expert weight projections IF stored as FP16** — attention Q/K/V/O projections, shared expert. But our weights are 4-bit quantized, not FP16. Would need separate FP16 weight copies (2x memory).
2. **RMS norm + SiLU activation** — lightweight ops, not worth the dispatch overhead.
3. **Prefill acceleration** — Core ML model for prefill while GPU handles decode. But prefill is not the bottleneck.

## Why ANE Won't Help Flash-MoE

1. **4-bit dequant is the core kernel** — ANE can't do custom dequantization. Our weights are packed uint32 with bf16 scales/biases. ANE only handles standard FP16/INT8.

2. **Dynamic MoE routing** — top-K expert selection is a runtime decision per token. ANE requires static computation graphs compiled ahead of time. Can't dynamically select which experts to compute.

3. **Memory bus is the bottleneck** — GPU only uses 8% of bus bandwidth. The bottleneck is SSD I/O (63% of layer time). Adding ANE as a third memory consumer doesn't help — it would contend with the SSD reads.

4. **Expert I/O dominates** — 4.0ms of the 6.4ms per layer is waiting for SSD. Even if ANE could do the GPU's 2.3ms of work for free, you'd only save the overlap between GPU and SSD (which the per-expert CMD experiment already exploits).

5. **No proven path at scale** — the only project with direct ANE access (maderix/ANE) benchmarked models up to 600M params. No one has run a >1B model on ANE. MLX explicitly rejected ANE support.

## Alternative: Use ANE for a Different Purpose

The ANE could theoretically be useful for:
- **Expert routing prediction** — a small Core ML model that predicts which experts will be needed for the next token, running on ANE while the GPU does the current token. But temporal prediction (`--predict`) already achieves this with simple carryover, and ML-based prediction was tested in the paper and found unhelpful (31% accuracy).
- **Token embedding lookup** — trivial, not worth the overhead.

## Recommendation

**Don't pursue ANE integration.** The bottleneck is SSD I/O bandwidth, not compute. The ANE can't help with dequant (custom kernel), can't help with routing (dynamic), and would add memory bus contention. Engineering effort is better spent on:
- SSD I/O optimizations (expert compression, smarter caching)
- GPU kernel improvements (larger batch dispatch, reduced launch overhead)
- Model-level changes (fewer experts, smaller expert size)

## References

- https://github.com/maderix/ANE — only project with direct ANE access, benchmarks up to 600M params
- https://github.com/hollance/neural-engine — best public documentation of ANE limitations
- https://github.com/apple/ml-ane-transformers — Apple's official ANE transformer work (small models)
- MLX issue #18 — rejected ANE support: "closed source API", "limited layer support"
