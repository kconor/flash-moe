#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["mlx", "numpy"]
# ///
"""Evaluate a trained expert predictor on test data.

Loads predictor weights and evaluates on a dataset, reporting:
- Top-K hit rate per layer (what % of actual experts were predicted)
- How many preads would be saved vs baseline (temporal carry)
- Expected I/O time reduction

Usage:
    ./eval.py ../models/Qwen3-Coder-Next-4bit/predictor_weights.npz \
              ../models/Qwen3-Coder-Next-4bit/predictor_coding.npz
"""

import argparse
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ExpertPredictor(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_experts)

    def __call__(self, x):
        return self.linear(x)


def main():
    parser = argparse.ArgumentParser(description="Evaluate expert predictor")
    parser.add_argument("weights", help="Path to predictor_weights.npz")
    parser.add_argument("data", help="Path to predictor data .npz")
    parser.add_argument("--predict-k", type=int, default=None,
                        help="Predict this many experts (default: K from data)")
    parser.add_argument("--predict-extra", type=int, default=0,
                        help="Predict K + extra experts for higher recall")
    args = parser.parse_args()

    # Load data
    d = np.load(args.data)
    X_all, Y_all = d["X"], d["Y"]
    layers_all = d["layers"]
    hidden_dim = X_all.shape[1]
    num_experts = Y_all.shape[1]
    K = int(Y_all[0].sum())
    predict_k = (args.predict_k or K) + args.predict_extra

    # Load weights
    w = np.load(args.weights)
    num_layers = int(layers_all.max()) + 1

    print(f"Data: {len(X_all)} samples, hidden={hidden_dim}, experts={num_experts}, K={K}")
    print(f"Predicting top-{predict_k} experts per sample")
    print()

    total_hits = 0
    total_actual = 0
    total_samples = 0

    # Baselines
    total_temporal_hits = 0  # previous token, same layer
    total_freq_hits = 0      # most frequent from first 80%

    print(f"{'Layer':>5} {'Samples':>8} {'Predictor':>10} {'Temporal':>10} {'Frequency':>10}")
    print(f"{'':>5} {'':>8} {'hit rate':>10} {'hit rate':>10} {'hit rate':>10}")
    print("-" * 55)

    for layer_idx in range(num_layers):
        w_key = f"layer_{layer_idx}_W"
        b_key = f"layer_{layer_idx}_b"
        if w_key not in w:
            continue

        mask = layers_all == layer_idx
        X_layer = X_all[mask]
        Y_layer = Y_all[mask]

        if len(X_layer) < 2:
            continue

        # Use last 20% as test
        split = int(len(X_layer) * 0.8)
        X_test = X_layer[split:]
        Y_test = Y_layer[split:]
        Y_train = Y_layer[:split]

        # Model prediction
        model = ExpertPredictor(hidden_dim, num_experts)
        model.linear.weight = mx.array(w[w_key])
        model.linear.bias = mx.array(w[b_key])

        logits = np.array(model(mx.array(X_test)))
        pred_indices = np.argsort(logits, axis=-1)[:, -predict_k:]

        hits = 0
        actual = 0
        for i in range(len(Y_test)):
            predicted = set(pred_indices[i])
            actual_experts = set(np.where(Y_test[i] > 0.5)[0])
            hits += len(predicted & actual_experts)
            actual += len(actual_experts)

        # Temporal baseline (previous sample same layer)
        temporal_hits = 0
        temporal_total = 0
        for i in range(1, len(Y_test)):
            prev = set(np.where(Y_test[i - 1] > 0.5)[0])
            curr = set(np.where(Y_test[i] > 0.5)[0])
            temporal_hits += len(prev & curr)
            temporal_total += len(curr)

        # Frequency baseline
        freq = Y_train.sum(axis=0)
        top_freq = set(np.argsort(freq)[-predict_k:])
        freq_hits = 0
        freq_total = 0
        for i in range(len(Y_test)):
            actual_experts = set(np.where(Y_test[i] > 0.5)[0])
            freq_hits += len(top_freq & actual_experts)
            freq_total += len(actual_experts)

        pred_rate = hits / actual if actual > 0 else 0
        temp_rate = temporal_hits / temporal_total if temporal_total > 0 else 0
        freq_rate = freq_hits / freq_total if freq_total > 0 else 0

        print(f"{layer_idx:5d} {len(Y_test):8d} {pred_rate:10.1%} {temp_rate:10.1%} {freq_rate:10.1%}")

        total_hits += hits
        total_actual += actual
        total_samples += len(Y_test)
        total_temporal_hits += temporal_hits
        total_freq_hits += freq_hits

    print("-" * 55)
    pred_overall = total_hits / total_actual if total_actual > 0 else 0
    temp_overall = total_temporal_hits / total_actual if total_actual > 0 else 0
    freq_overall = total_freq_hits / total_actual if total_actual > 0 else 0

    print(f"{'ALL':>5} {total_samples:8d} {pred_overall:10.1%} {temp_overall:10.1%} {freq_overall:10.1%}")
    print()

    # Estimate I/O savings
    cold_per_layer_baseline = K * (1 - temp_overall)  # experts needing SSD read with temporal carry
    cold_per_layer_predictor = K * (1 - pred_overall)

    print(f"=== I/O Impact Estimate ===")
    print(f"Baseline (temporal carry): {cold_per_layer_baseline:.1f} cold reads/layer")
    print(f"Predictor (top-{predict_k}):     {cold_per_layer_predictor:.1f} cold reads/layer")
    print(f"Saved reads/layer: {cold_per_layer_baseline - cold_per_layer_predictor:.1f}")


if __name__ == "__main__":
    main()
