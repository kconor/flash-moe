Keep track of all experiment status in this file.
1. Add hypothesis to this doc
2. create a branch for the experiment
3. document implementation strategy in separate markdown doc in this directory
3. Attempt to implement
4. Add status of implementation to this doc
5. If successful:
 5.1 benchmark with --timing and add results to this document
 5.2 update implementation doc with what worked or didn't
 5.3 create a commit of changes in this directory and merge back into main
 5.4 create a commit of all changes, but leave in experiment specific branch



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
