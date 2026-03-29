# Experiment 3: Expert Routing Predictor (Pre-Attention Prefetch)

## Hypothesis
A lightweight linear predictor trained on pre-attention hidden states can predict which experts will be routed, enabling SSD prefetch during the CMD1 GPU wait (~1.2ms). If the predictor achieves >70% hit rate, prefetched experts avoid cold reads on the critical path, reducing per-layer time.

## Background

Each MoE layer routes K experts after running attention + routing MLP. The routing result isn't known until CMD2 completes. But the pre-attention hidden state is available at CMD1 submission — giving a ~1.6ms window (CMD1 wait + attention + CMD2) to prefetch predicted experts from SSD before routing finishes.

## Approach

### Data Collection
Added `--collect-predictor <path>` flag to infer.m. Per MoE layer, per token, writes:
- `int32 token_idx, layer_idx`
- `float32 pre_attn_hidden[hidden_dim]` (2048 floats)
- `int32 K` followed by `int32 expert_indices[K]`

Binary format, appended during generation. Converted to `.npz` via `read_predictor_data.py --to-npz`.

### Model
Per-layer single linear model: `P(expert_i | hidden) = sigmoid(W @ hidden + b)` where W is [num_experts, hidden_dim]. Trained with BCE loss, evaluated on top-K hit rate (what fraction of actual routed experts appear in the top-K predictions).

### Training
- Framework: MLX (Apple Silicon optimized)
- Data: ~10K tokens of coding prompts, K=10
- Train/test split: 80/20 by token order (temporal, not random)
- Optimizer: Adam, lr=0.01, 10 epochs, batch size 256
- Per-layer training: ~1-2s per layer on M4 Pro

### Runtime Integration
- Predictor weights exported to flat binary (`.bin`) via `export_weights.py`
- Loaded via mmap at startup (`--predictor <path>`)
- During CMD1 wait: `cblas_sgemv` computes scores, partial sort finds top-K
- Sequential prefetch thread reads predicted experts into buf_B
- After routing: match predictions against actual, pread misses into buf_A

## Results

### Predictor Accuracy (eval.py)
Hit rate = fraction of actual K=10 experts captured in top-10 predictions.

| Metric | Predictor | Temporal baseline | Frequency baseline |
|--------|-----------|-------------------|-------------------|
| Hit rate (avg) | ~46% | ~53% | ~45% |

The predictor **underperforms temporal carry** (reusing previous token's experts for the same layer). This was consistent across layers.

### Working Set Analysis (K=10 routing data)

Expert usage is broadly distributed — most layers touch 420-510 of 512 experts across 10K tokens.

| Tokens generated | Unique experts/layer (avg) | Working set (36 MoE layers) |
|---|---|---|
| 10 | 48 | 3.0 GB |
| 20 | 65 | 4.1 GB |
| 50 | 126 | 8.0 GB |
| 100 | 184 | 11.7 GB |
| 200 | 254 | 16.2 GB |

Cache needed for X% recall (fraction of activations covered by top-N experts per layer):

| Recall | Experts/layer | Cache needed |
|---|---|---|
| 90% | 220 | 14.0 GB |
| 95% | 279 | 17.7 GB |
| 99% | 371 | 23.6 GB |
| 100% | 475 | 30.3 GB |

Expert size is 1.77 MB (4-bit). Available page cache on 24GB machine is ~16 GB.

### Expert Frequency Distribution (per layer)
Top-50 experts cover only 39-68% of activations depending on layer. Top-200 cover 71-95%. Routing is not Zipfian enough for a small hot set to dominate.

## Conclusion: REJECTED

1. **Predictor accuracy too low.** 46% hit rate means 54% of prefetched reads are wasted, consuming SSD bandwidth that should serve actual experts. Worse than temporal carry (53%).

2. **Single linear model is underpowered.** Pre-attention hidden states don't contain enough information to predict post-attention routing. The attention computation significantly reshapes the representation before the router sees it.

3. **Working set exceeds cache.** On 24GB RAM, the page cache (~16 GB) can hold ~50% of expert data. At 200+ tokens, cache fills and eviction begins. No predictor can avoid cold reads when the working set fundamentally exceeds memory.

4. **Cancellation is hard.** Even with `dup()`/`close()` per-fd trick or `mincore()` for surgical prefetch, the fundamental accuracy problem (46%) means most prefetches are wasted.

## What Would Change This

- **More RAM (48+ GB):** The original M3 Max paper machine has 48GB. With ~35GB page cache, 95%+ of experts stay warm and prediction becomes less important.
- **K=4 instead of K=10:** Fewer active experts = smaller working set = higher cache hit rate naturally. The predictor data was collected at K=10; K=4 would have very different (likely better) cache behavior.
- **Deeper predictor model:** A 2-layer MLP or attention-aware predictor might do better, but adds latency to the CMD1 window.
- **mincore()-guided single prefetch:** Instead of prefetching all K, use `mincore()` on the mmap to find which predicted expert is cold and prefetch only that one. Minimal waste. Worth revisiting if predictor accuracy improves.

## Files

- `expert_predictor/train.py` — Per-layer predictor training
- `expert_predictor/eval.py` — Evaluation with temporal/frequency baselines
- `expert_predictor/export_weights.py` — NPZ to flat binary conversion
- `metal_infer/read_predictor_data.py` — Binary data reader + NPZ export
- Branch: `experiment/expert-predictor` — Runtime integration in infer.m
