Keep track of all experiment status in this file.
1. Add hypothesis to this doc
2. create a branch for the experiment
3. document implementation strategy in separate markdown doc in this directory
3. Attempt to implement
4. Add status of implementation to this doc
5. If successful:
 5.1 benchmark and add results to this document (see Benchmarking below)
 5.2 update implementation doc with what worked or didn't
 5.3 create a commit of changes in this directory and merge back into main
 5.4 create a commit of all changes, but leave in experiment specific branch



# Benchmarking

## Quick benchmark (10 queries × 100 tokens)
```bash
cd metal_infer
./benchmark.sh ../models/Qwen3.5-35B-A3B-4bit --k 8
./benchmark.sh ../models/Qwen3-Coder-Next-4bit --k 10
./benchmark.sh ../models/Qwen3.5-35B-A3B-8bit --k 8
```

Reports per-query tok/s and TTFT, plus min/avg/median/max summary.

## Per-layer timing breakdown
```bash
./infer --model ../models/Qwen3.5-35B-A3B-4bit --prompt "Hello" --tokens 100 --k 8 --timing
```

Shows avg ms per phase: expert_io (SSD), cmd1_wait/cmd2_wait (GPU), cpu_attn (CPU), etc.

## Baseline results (M4 Pro, 24GB)

| Model | tok/s (median) | TTFT (avg) | Expert size | K |
|-------|---------------|-----------|-------------|---|
| Qwen3.5-35B-A3B-4bit | 16.74 | 1247ms | 1.7MB | 8 |
| Qwen3.5-35B-A3B-8bit | 8.29 | 2957ms | 3.3MB | 8 |
| Qwen3-Coder-Next-4bit | 8.52 | 2613ms | 1.7MB | 10 |

# Experiments:

## Experiment 1: Per-Expert Command Buffers (overlap GPU with SSD I/O)
**Hypothesis:** Submitting a separate Metal command buffer per expert as soon as its pread completes (rather than waiting for all K) will reduce total layer time by overlapping GPU compute with remaining SSD reads. Expected improvement: ~10-20% on I/O-bound models where page cache hit rate is mixed (some cached, some cold).

**Status:** Implemented and benchmarked

**Results (Qwen3-Coder-Next-4bit, K=10, 100 tokens):**
| Metric | Baseline | Per-Expert CMD | Delta |
|--------|----------|---------------|-------|
| expert_io | 1.171 ms | 0.005 ms | -99.6% |
| total_layer | 2.340 ms | 2.140 ms | -8.5% |
| tok/s | 8.73 | 9.52 | **+9.0%** |

Expert I/O fully overlapped with GPU. sum_phases dropped 61% confirming heavy pipeline overlap.

## Experiment 2: Progressive Encoding (single CMD3, encode-as-available)
**Hypothesis:** Encoding expert forward passes into a single CMD3 as each pread completes (rather than waiting for all K then encoding all at once) will reduce the gap between last pread and CMD3 commit. Lower overhead than Experiment 1 (one commit vs K commits) but GPU can't start until all are encoded.

**Status:** Implemented and benchmarked — REJECTED

**Results (Qwen3-Coder-Next-4bit, K=10, 100 tokens):**
| Metric | Baseline | Progressive Encode | Delta |
|--------|----------|-------------------|-------|
| expert_io | 1.171 ms | 0.004 ms | -99.7% |
| cmd3_encode | 0.030 ms | 1.204 ms | +3913% (absorbed the wait) |
| total_layer | 2.340 ms | 2.800 ms | **+19.7%** |
| tok/s | 8.73 | 7.24 | **-17.1%** |

**Why it failed:** The single CMD3 can't be committed until all experts are encoded. The encoding loop spins waiting for slow preads, moving the I/O wait into cmd3_encode instead of eliminating it. Worse than baseline because the spin-poll adds CPU overhead and the GPU stays idle longer (no deferred overlap with next layer).

## Experiment 3: Expert Routing Predictor (pre-attention prefetch)
**Hypothesis:** A linear predictor trained on pre-attention hidden states can predict expert routing during CMD1 wait, enabling SSD prefetch before routing completes. If hit rate >70%, cold reads are avoided on the critical path.

**Status:** Implemented and evaluated — REJECTED

**Results (Qwen3-Coder-Next-4bit, K=10, predictor_coding dataset):**
| Metric | Predictor | Temporal baseline | Frequency baseline |
|--------|-----------|-------------------|-------------------|
| Hit rate | ~46% | ~53% | ~45% |

**Why it failed:** Pre-attention hidden states don't predict post-attention routing well enough. 46% hit rate is worse than simply reusing the previous token's experts (temporal carry, 53%). Additionally, working set analysis shows the expert cache (~16 GB on 24GB machine) can only hold ~50% of expert data (32.6 GB total across 36 MoE layers × 512 experts × 1.77 MB). At 200+ tokens, cache fills regardless of prediction. See [exp3-expert-predictor.md](exp3-expert-predictor.md) for full analysis.

**Branch:** `experiment/expert-predictor`

## Experiment 5: mlock Model Weights
**Hypothesis:** Wiring model_weights.bin into physical memory via mlock() prevents the kernel from evicting weight pages under memory pressure, eliminating GPU stalls during CMD1/CMD2.

**Status:** Implemented and benchmarked — NO EFFECT (without memory pressure)

**Results (Qwen3-Coder-Next-4bit, K=10, 10 queries × 100 tokens):**
| Config | median tok/s | TTFT (avg) |
|--------|-------------|-----------|
| Baseline | 10.29 | 2194 ms |
| mlock weights | 10.41 | 2132 ms |

No meaningful difference — mlock only helps when something else (e.g., a custom cache) is creating memory pressure. Without memory pressure, the OS keeps weight pages resident naturally. However, mlock proved critical when combined with CLOCK-Pro cache (4.76 → 8.08 tok/s, see experiment 4).

**Branch:** `experiment/mlock-weights`

## Experiment 6: MADV_RANDOM on Expert mmaps
**Hypothesis:** Telling the kernel expert access is random (MADV_RANDOM) will disable readahead on mmap'd expert layer files, saving SSD bandwidth. Previous testing on M3 Max/397B (7 MB experts) found it hurt — re-testing on M4 Pro with 1.7 MB experts.

**Status:** Implemented and benchmarked — NO EFFECT

**Results (Qwen3-Coder-Next-4bit, K=10, 10 queries × 100 tokens):**
| Config | median tok/s | TTFT (avg) |
|--------|-------------|-----------|
| Baseline | 10.29 | 2194 ms |
| MADV_RANDOM | 10.27 | 2185 ms |

No meaningful difference on M4 Pro either. The F_RDAHEAD=0 already set on the fds likely does the same thing. Original finding confirmed: kernel default madvise is fine for expert files.

**Branch:** `experiment/madv-random-experts`

## Experiment 7: Zero-Copy GPU Expert Access (mmap → Metal buffer → GPU)
**Hypothesis:** Eliminating the pread+memcpy step by wrapping mmap'd expert pages directly as Metal buffers saves ~6.5µs per expert (1.77MB memcpy). GPU reads directly from UBC pages via `newBufferWithBytesNoCopy`. Expected improvement: ~4% from eliminating memcpy overhead.

**Status:** Implemented and benchmarked — REJECTED (-50% tok/s)

**Implementation:** Per-expert individual mmaps (48 layers × 512 experts = 24,576 mmaps), each wrapped as a Metal buffer at init. I/O threads pre-fault pages via mlock before GPU dispatch. Tested three page-in strategies: madvise(WILLNEED)+touch loop, mlock, and main-thread touch. All produced the same result.

**Results (Qwen3-Coder-Next-4bit, K=10):**
| Config | tok/s | Notes |
|--------|-------|-------|
| Baseline (pread) | ~12 | 2MB-aligned anonymous Metal buffers |
| Zero-copy mmap | ~6 | GPU reads from scattered file-backed pages |

**Why it failed:** The baseline's memcpy isn't wasted work — it **compacts scattered UBC pages into contiguous, 2MB-aligned anonymous buffers**. The GPU dequant kernel is bandwidth-bound (~418 GiB/s). Reading from scattered 16KB file-backed pages causes:
1. **TLB pressure**: 108 scattered pages vs 1 contiguous 2MB region (possibly 1 TLB entry with huge pages)
2. **Memory controller coalescing**: Contiguous physical pages allow burst reads; scattered pages serialize
3. The comment in the baseline says "pread DMA controller transfers 3.6x faster with 2MB alignment vs 16KB" — the GPU has the same alignment sensitivity

**Key insight:** On unified memory architectures, the memcpy from UBC → aligned anonymous buffer is a necessary step for GPU bandwidth. Zero-copy only works when the source pages are already contiguous and well-aligned, which file-backed UBC pages are not.

**Branch:** `experiment/mmap-zero-copy`
