#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["mlx", "numpy"]
# ///
"""Train a per-layer expert predictor from pre-attention hidden states.

Trains a single-layer perceptron per layer:
    P(expert_i | hidden) = sigmoid(W @ hidden + b)

The model predicts which experts will be activated given the hidden state
available BEFORE attention runs, enabling prefetching during GPU compute.

Usage:
    ./train.py ../models/Qwen3-Coder-Next-4bit/predictor_coding.npz
    ./train.py data.npz --epochs 20 --lr 0.001 --layer 24
"""

import argparse
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class ExpertPredictor(nn.Module):
    """Single hidden layer predictor: hidden_dim -> num_experts."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_experts)

    def __call__(self, x):
        return self.linear(x)


def binary_cross_entropy(logits, targets):
    """BCE loss for multi-hot targets."""
    # Numerically stable: -t*log(sigmoid(x)) - (1-t)*log(1-sigmoid(x))
    # = max(x,0) - x*t + log(1 + exp(-|x|))
    return mx.mean(mx.maximum(logits, 0) - logits * targets + mx.log1p(mx.exp(-mx.abs(logits))))


def topk_hit_rate(logits, targets, k):
    """Fraction of actual experts captured by top-k predictions."""
    preds = mx.argsort(logits, axis=-1)[:, -k:]  # top-k indices
    hits = 0
    total = 0
    # MLX doesn't have gather_nd, do it with a loop over batch
    preds_np = np.array(preds)
    targets_np = np.array(targets)
    for i in range(preds_np.shape[0]):
        predicted = set(preds_np[i])
        actual = set(np.where(targets_np[i] > 0.5)[0])
        hits += len(predicted & actual)
        total += len(actual)
    return hits / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Train expert predictor")
    parser.add_argument("data", help="Path to .npz file from read_predictor_data.py")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--layer", type=int, default=None,
                        help="Train for specific layer only (default: all layers)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save trained weights (default: <data_dir>/predictor_weights.npz)")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--predict-k", type=int, default=None,
                        help="Number of experts to predict (default: K from data)")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.data}...")
    d = np.load(args.data)
    X_all, Y_all = d["X"], d["Y"]
    layers_all, tokens_all = d["layers"], d["tokens"]
    hidden_dim = X_all.shape[1]
    num_experts = Y_all.shape[1]
    K = int(Y_all[0].sum())
    predict_k = args.predict_k or K

    print(f"  Samples: {len(X_all)}, Hidden: {hidden_dim}, Experts: {num_experts}, K: {K}")

    num_layers = int(layers_all.max()) + 1
    layer_list = [args.layer] if args.layer is not None else list(range(num_layers))

    # Storage for all layer weights
    all_weights = {}

    for layer_idx in layer_list:
        mask = layers_all == layer_idx
        X_layer = X_all[mask]
        Y_layer = Y_all[mask]

        if len(X_layer) < 10:
            print(f"Layer {layer_idx}: skipping ({len(X_layer)} samples)")
            continue

        # Train/test split (by token order, not random — respects temporal structure)
        n = len(X_layer)
        split = int(n * (1 - args.test_split))
        X_train, X_test = X_layer[:split], X_layer[split:]
        Y_train, Y_test = Y_layer[:split], Y_layer[split:]

        # Convert to MLX
        X_train_mx = mx.array(X_train)
        Y_train_mx = mx.array(Y_train)
        X_test_mx = mx.array(X_test)
        Y_test_mx = mx.array(Y_test)

        # Model
        model = ExpertPredictor(hidden_dim, num_experts)
        mx.eval(model.parameters())

        optimizer = optim.Adam(learning_rate=args.lr)

        def loss_fn(model, x, y):
            logits = model(x)
            return binary_cross_entropy(logits, y)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        # Training loop
        t0 = time.time()
        n_train = len(X_train)
        bs = args.batch_size

        for epoch in range(args.epochs):
            # Shuffle training data
            perm = np.random.permutation(n_train)
            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_train, bs):
                idx = perm[i : i + bs]
                xb = X_train_mx[mx.array(idx)]
                yb = Y_train_mx[mx.array(idx)]

                loss, grads = loss_and_grad(model, xb, yb)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                epoch_loss += loss.item()
                n_batches += 1

        train_time = time.time() - t0

        # Evaluate
        test_logits = model(X_test_mx)
        test_loss = binary_cross_entropy(test_logits, Y_test_mx).item()
        hit_rate = topk_hit_rate(test_logits, Y_test_mx, predict_k)

        # Baseline: random prediction
        random_hit_rate = predict_k / num_experts

        # Baseline: always predict most frequent experts from training set
        freq = Y_train.sum(axis=0)
        top_freq = np.argsort(freq)[-predict_k:]
        freq_hits = sum(Y_test[i, top_freq].sum() for i in range(len(Y_test)))
        freq_hit_rate = freq_hits / (len(Y_test) * K)

        print(f"Layer {layer_idx:2d}: loss={test_loss:.4f} "
              f"top-{predict_k} hit={hit_rate:.1%} "
              f"(freq baseline={freq_hit_rate:.1%}, random={random_hit_rate:.1%}) "
              f"train={train_time:.1f}s samples={n_train}+{len(Y_test)}")

        # Save weights
        w = np.array(model.linear.weight)
        b = np.array(model.linear.bias)
        all_weights[f"layer_{layer_idx}_W"] = w
        all_weights[f"layer_{layer_idx}_b"] = b

    # Save all weights
    if all_weights:
        out_path = args.output or os.path.join(os.path.dirname(args.data), "predictor_weights.npz")
        np.savez(out_path, **all_weights, hidden_dim=hidden_dim, num_experts=num_experts)
        print(f"\nSaved weights to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
