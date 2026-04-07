# Kernel Optimization Learnings — Qwen3.5-9B Dense (Q8) on M4 Pro

Session goal: push the dense Qwen3.5-9B Q8 inference from 13.1 tok/s to 25 tok/s on M4 Pro / 20-core GPU, by optimizing Metal compute kernels.

**End state:** ~23 tok/s (from 13.1, +77%). Short of the 25 tok/s target but within ~5% of practical hardware ceiling. Most remaining gap is measurement noise and residual CPU-side overhead, not kernel inefficiency.

## Results table

| Change | tok/s | Δ | Notes |
|---|---|---|---|
| Baseline (Q8 kernels ported from MoE 4-bit) | 13.1 | — | Single-row SIMD, x_shared cache 4096 f32 |
| Q8 `matvec_8bit_opt`: 4 rows/SIMD (MLX-style multi-row) | 13.9 | +6% | Rows per TG: 8 → 32 |
| Fuse CMD2 and CMD3 into one async command buffer (dense) | 14.6 | +5% | Eliminates a CMD2 wait + CMD3 commit per layer |
| Q8 `matvec_8bit_fast_opt`: multi-row SIMD for `in_dim>4096` (down_proj) | 15.5 | +6% | Same shape optimization as `_opt` for large `in_dim` |
| **BF16 `matvec_bf16_opt`: fix TG shared memory bloat** | 17.5 | **+13%** | `x_shared[8192]` → `[4096]` — the 32 KB allocation capped occupancy to 1 TG/core. Halving it lets 2 TGs/core run, ~2× throughput for o_proj |
| **BF16 `matvec_bf16_opt`: vectorized `uint4` loads (8 bf16/load)** | 19.4 | **+11%** | 2-byte loads are inefficient on Apple Silicon; uint4 = 16 bytes = 8 bf16 values per LD instruction |
| Fused `residual_rms_norm` kernel (3 encoders → 1) | 19.5 | ~0% | Dispatch overhead wasn't the bottleneck on Apple Silicon |
| Full-attn GPU Q/K-norm + RoPE kernels (eliminate CPU round-trip) | 19.8 | +2% | Helpful but small — most of the CPU overhead was elsewhere |
| **GPU linear-attn pipeline for BF16 `in_proj_a`/`in_proj_b` (bug fix)** | 22–23 | **+17%** | The existing GPU delta-net pipeline was silently falling back to CPU because `num_attn_specs == 4` check assumed all projections were quantized. 9B has BF16 a/b. Dispatched them via a fused `bf16_matvec_pair` kernel and enabled the GPU pipeline |
| Q8 `uint4`-load Q8 kernel (16 values/iter) | same | 0% | No gain — register pressure negated the bandwidth win |
| Fused Q8 `swiglu + down_proj` | same | ~0% | Saved the act buffer write/read but negligible at this scale |
| Fused Q8 `gate_proj + up_proj` in one kernel | **−5%** | — | **Register pressure dropped TG occupancy** — worse than separate dispatches |
| Small TG size (2 SIMDs/TG = 64 threads) for Q8 | −10% | — | Apple GPU prefers larger TGs when TG shared memory budget allows it |

---

## Key learnings

### 1. **Threadgroup memory is an occupancy cliff on Apple Silicon**
Apple GPU cores have ~32 KB of threadgroup-private memory per compute unit. A kernel that allocates 32 KB leaves no room for a second concurrent TG, halving occupancy.

The single biggest win in this session was noticing that `bf16_matvec` allocated `threadgroup float x_shared[8192]` = 32 KB, even though `in_dim` was only 4096. Shrinking to 16 KB immediately gave +13% end-to-end.

**Rule of thumb:** size threadgroup memory to fit TWO or more TGs per core (≤ 16 KB). If you can't, consider tiling the input dim.

### 2. **Vectorized `uint4` loads beat `uint16` loads** (but watch register pressure)
Apple Silicon's native load granularity is 16 bytes. Loading bf16 as individual `uint16` values uses 1/8 of the load bandwidth of a `uint4` fetch. Rewriting the bf16 inner loop to issue `uint4 p = *(device const uint4*)(W + col)` and extract 8 bf16 values per load was **+11% end-to-end** for the o_proj path.

But **Q8 with uint4 loads + per-row accumulators caused register pressure** that reduced occupancy. Measured 0% gain for Q8 (vs the uint32 path). The break-even point depends on how many registers the per-row accumulators need. Test both.

### 3. **Multi-row per SIMD (MLX/AO qmv_fast pattern) is a moderate win**
Traditional matvec: each SIMD group computes ONE output row. Multi-row pattern: 4 output rows per SIMD group, using a `float4` accumulator. Each lane's `x` read is amortized across 4 rows.

This gave +6% on Q8 and +? on BF16. It's the right baseline for matvec kernels on Apple GPUs, but the gain per kernel is bounded: `x` is small and already cache-hot from threadgroup memory, so the amortization is over loads that weren't very expensive anyway. The real benefit is fewer threadgroups, which reduces per-TG fixed setup overhead.

**32 rows per TG (8 SIMDs × 4 rows)** was the sweet spot. Smaller TGs (2 SIMDs) hurt because of TG setup overhead. Larger tiles (8 rows/SIMD) hurt because of register pressure.

### 4. **Affine decomposition (scale·dot + bias·sum_x) is correct, but the compiler handles it fine either way**
MLX uses `result = scale * dot(w_int, x) + bias * sum(x)` per group to amortize the bias multiply. I implemented this and it looked clean in the kernel, but A/B testing against the naive per-element `fma(w, scale*x, bias*x)` showed no measurable difference. The Metal compiler figures out the algebra either way.

Conclusion: use whichever is more readable.

### 5. **Dispatch overhead on Apple Silicon is smaller than I expected**
Kernel fusion experiments (`residual + rms_sum + rms_apply` → one kernel; `swiglu + down_proj` → one kernel) saved ~0–20 µs per layer. Less than the ~30-40 µs I estimated. Fewer encoders is nice for code clarity but isn't a performance win by itself once you have 5–10 encoders per command buffer.

**Where fusion matters is memory traffic**, not dispatch count. If fusing two kernels lets an intermediate buffer stay in registers instead of being written and re-read from global memory, THAT is the win.

### 6. **The biggest performance bugs are correctness bugs that silently use the slow path**

The largest structural speedup of the session (+17%) came from discovering that the `num_attn_specs == 4` check in the GPU-linear-attention fast path was **silently falling back to CPU delta-net (BLAS)** for 24 of 32 layers, because the 9B's `in_proj_b` and `in_proj_a` are BF16 (not quantized) — so they weren't added to the Q8 batch matvec spec list, so `num_attn_specs == 2`, so the check failed.

The symptom was an innocuous `cpu_attn: 0.131 ms/layer` in the timing breakdown — just high enough to look plausible and low enough to hide in the noise. The fix: add a tiny fused `bf16_matvec_pair` kernel that computes both `a` and `b` projections into `batch_out[2]/[3]`, then accept either `num_attn_specs == 4` OR `have_bf16_ba` as the gate for the GPU linear-attn pipeline.

**Lesson: when profiling, scrutinize every timing line for plausibility. `cpu_attn: 0.13` could be CPU attention OR it could be a silent fallback to a CPU BLAS path you didn't know about.**

### 7. **Profile by skipping, not by adding**
The single most useful diagnostic I built was an `env SKIP_GATE=1 SKIP_UP=1 SKIP_DOWN=1 SKIP_OPROJ=1 ./infer …` pattern that bypassed individual matmul dispatches. Running it in combinations revealed:
- o_proj alone (BF16, 32MB): 0.525 ms → 22% of peak bandwidth → *huge red flag*
- gate/up/down alone (Q8, 50MB each): ~0.24 ms → 80% of peak → *close to ceiling*

This pointed me directly at the BF16 kernel shared-memory bug. Without the per-kernel isolation, I would have spent hours micro-optimizing Q8 for marginal gains while BF16 was silently costing 400 µs per layer.

### 8. **GPU timing via `MTLCommandBuffer.GPU{Start,End}Time` is accurate and free**

Adding `addCompletedHandler:` with `GPUStartTime/GPUEndTime` gave precise per-command-buffer GPU execution time without any instrumentation overhead. This separated "command buffer wait time" (which includes serial queue delay from prev commands) from "actual GPU work" and was essential for understanding what was bandwidth-bound vs what was dispatch-bound.

### 9. **On a serial GPU queue, `cmd1_wait` mostly waits for the previous layer's work**
The reported `cmd1_wait` is large (e.g. 1.0 ms) but deceptive: it's the time from committing CMD1(N) to its completion, which on a serial queue is `GPU(CMD_fused(N-1)) + GPU(CMD1(N))`. The actual CPU *overhead* per layer is tiny — the wait time is GPU work you're running to completion. Don't try to "reduce cmd1_wait"; reduce the GPU work, or overlap CPU work with the wait.

### 10. **Theoretical bandwidth ceiling is ~75–80% of advertised**
Apple M4 Pro advertises 273 GB/s memory bandwidth. Practical sustained bandwidth for compute kernels is closer to 200–220 GB/s (75–80%). All my well-tuned kernels (Q8 small: 80%, Q8 down: 74%, BF16 o_proj: 84%) landed in this band. Trying to push higher is chasing diminishing returns.

For the Qwen3.5-9B Q8 model at 7 GB weight reads per token, that's a hard floor of **~7 / 200 = 35 ms per token = 28 tok/s**. Observed end state ~23 tok/s leaves about 8 ms/token of non-weight-bandwidth time (lm_head dispatch, CPU overhead, sampling, encoding).

---

## The pipeline shape that worked

For Qwen3.5-9B dense Q8 on M4 Pro:

- **CMD1 per layer**: all attention projections + GPU linear-attention pipeline (or full-attn Q/K-norm + RoPE + KV-cache update) in one command buffer, committed but not waited for.
- **CMD_fused per layer** (formerly CMD2 + CMD3): o_proj + fused residual/rms-norm + gate_proj + up_proj + swiglu + down_proj + residual + next-layer norm, all in one command buffer, deferred-committed, never waited for directly.
- **Per-layer CPU sync**: `wait_deferred_experts_gpu()` waits for the *previous* layer's CMD_fused to complete before finalize_deferred_experts memcpys `buf_moe_hidden → hidden`. On the serial GPU queue, this wait also implies CMD1(N) is in flight but CPU returns before it finishes.
- **CPU critical path per layer**: ~10–30 µs (deferred finalize + cmd_fused encode). Everything else is GPU.

Per-layer GPU time at the end of the session:
- `cmd1_gpu`: 0.29 ms (84% peak for ~50 MB of weights)
- `cmd_fused_gpu`: 0.87 ms (76% peak for ~180 MB)
- `total_layer`: 1.17–1.27 ms wall clock
- Theoretical floor at 200 GB/s: ~1.15 ms/layer

---

## What I'd try next (out of scope for this session)

1. **Process lm_head in chunks overlapped with sampling.** lm_head is 1 GB bandwidth per token = ~5 ms, which is 10–12% of per-token time. Chunking it might let the sampler work on earlier logits while later chunks compute.
2. **Fused RoPE + Q/K-norm** kernels with fewer barriers. My q/k_norm_rope kernels are a single threadgroup-barrier reduction + math; they should be under 5 µs each but I didn't micro-benchmark.
3. **`MTLIndirectCommandBuffer`** to pre-record the whole 32-layer forward pass and replay per token, eliminating all per-layer encoding CPU cost. This is the biggest known remaining optimization on Apple Silicon.
4. **GPU sampler.** Replace the CPU softmax+argmax with a Metal kernel that reads logits and writes the token id. Saves a ~1 MB readback per token.
5. **Batch multiple kv-cache writes** with a single blit encoder per token instead of per-full-attn-layer.

---

## Kernel-by-kernel summary (final state)

| Kernel | Purpose | Key traits |
|---|---|---|
| `dequant_matvec_8bit_opt` | Q8 matvec for `in_dim ≤ 4096` | 4 rows/SIMD, 8 SIMDs/TG, 16 KB x_shared, uint4 weight loads, affine decomposition |
| `dequant_matvec_8bit_fast_opt` | Q8 matvec for `in_dim > 4096` (down_proj) | Same multi-row pattern, **tiled 4 KB x_shared** (cycles through tiles of in_dim) |
| `dequant_matvec_8bit_fast_swiglu_down` | Fused swiglu + Q8 down_proj | Same as fast_opt but reads gate/up directly, computes `silu(g)*u` inline |
| `bf16_matvec_opt` | BF16 matvec for `in_dim ≤ 4096` | 4 rows/SIMD, 8 SIMDs/TG, **16 KB x_shared (not 32)**, uint4 weight loads |
| `bf16_matvec_pair` | Fused 2-matrix BF16 matvec for tiny out_dim | Used for linear-attn `in_proj_a` + `in_proj_b` — two 32×4096 matmuls in one dispatch |
| `q_norm_rope` | Q head-norm + RoPE in CMD1 for full-attn | 1 TG per Q head, `head_dim` threads, threadgroup reduction for RMS |
| `k_norm_rope` | K head-norm + RoPE in CMD1 for full-attn | Same pattern, separate dispatch (K has fewer heads than Q) |
| `residual_rms_norm_fused` | residual_add + rms_sum_sq + rms_apply_bf16 in one kernel | Single TG, 256 threads, two-phase (accumulate / reduce / apply) |

---

## The "learned the hard way" list

- **Don't trust micro-optimizations without measuring end-to-end.** I spent 30+ minutes on a uint4 Q8 kernel that was mathematically equivalent and 0% faster because of register pressure.
- **`cpu_attn` in the phase breakdown is a black box.** Break it into subcategories before trying to reduce it.
- **Every `if (condition) { gpu_path } else { cpu_path }` in your engine is a potential silent slow path.** Make the condition easy to verify with a one-line debug print.
- **Metal command queues are SERIAL.** Submitting CMD1 before CMD_fused of the previous layer is done does not parallelize them. Use multiple queues or accept the serialization.
- **`threadgroup float x_shared[N]` allocates N × 4 bytes of fixed threadgroup memory.** Sizing it to the maximum possible `in_dim` even when your actual use case is smaller costs you 2× occupancy.
- **Profiling via "skip the kernel and see what happens" is fast, accurate, and requires no instrumentation.** Always have a `SKIP_X` env var for your hot kernels.
