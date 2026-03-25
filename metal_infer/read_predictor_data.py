#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""Read and summarize predictor training data from --collect-predictor.

Usage:
    ./read_predictor_data.py <predictor_data.bin> [--hidden-dim N]

Binary format per record:
    int32   token_idx
    int32   layer_idx
    float32 pre_attn_hidden[hidden_dim]
    int32   K
    int32   expert_indices[K]
"""

import argparse
import struct
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Read predictor training data")
    parser.add_argument("file", help="Path to predictor_data.bin")
    parser.add_argument("--hidden-dim", type=int, default=3072, help="Hidden dimension (default: 3072)")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics only")
    args = parser.parse_args()

    data = open(args.file, "rb").read()
    hidden_dim = args.hidden_dim

    records = []
    offset = 0
    while offset < len(data):
        if offset + 8 > len(data):
            break
        ti, li = struct.unpack_from("<ii", data, offset)
        offset += 8
        if offset + hidden_dim * 4 > len(data):
            break
        h = np.frombuffer(data, dtype=np.float32, count=hidden_dim, offset=offset).copy()
        offset += hidden_dim * 4
        if offset + 4 > len(data):
            break
        K = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        if offset + K * 4 > len(data):
            break
        experts = list(struct.unpack_from(f"<{K}i", data, offset))
        offset += K * 4
        records.append((ti, li, h, K, experts))

    num_tokens = max(r[0] for r in records) + 1 if records else 0
    num_layers = max(r[1] for r in records) + 1 if records else 0

    print(f"Records: {len(records)}")
    print(f"Tokens: {num_tokens}, Layers: {num_layers}")
    print(f"Hidden dim: {hidden_dim}, K: {records[0][3] if records else '?'}")
    print(f"File size: {len(data) / 1e6:.1f} MB")
    print()

    if not args.summary:
        # Print first few records
        for i, (ti, li, h, K, experts) in enumerate(records[:10]):
            rms = np.sqrt(np.mean(h ** 2))
            print(f"  token={ti:3d} layer={li:2d} hidden_rms={rms:.3f} K={K} experts={experts}")
        if len(records) > 10:
            print(f"  ... ({len(records) - 10} more records)")
        print()

    # Token-to-token expert overlap per layer
    print("=== Token-to-token expert overlap (same layer, consecutive tokens) ===")
    overlaps = []
    for layer in range(num_layers):
        layer_records = [(ti, experts) for ti, li, _, _, experts in records if li == layer]
        layer_records.sort()
        for i in range(1, len(layer_records)):
            prev = set(layer_records[i - 1][1])
            curr = set(layer_records[i][1])
            overlaps.append(len(prev & curr))
    K = records[0][3] if records else 8
    print(f"  Avg overlap: {np.mean(overlaps):.1f}/{K} ({np.mean(overlaps)/K*100:.0f}%)")
    print()

    # Hidden state similarity vs expert overlap
    print("=== Pre-attention hidden similarity vs expert overlap ===")
    for test_layer in [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]:
        layer_records = [(ti, h, experts) for ti, li, h, _, experts in records if li == test_layer]
        layer_records.sort()
        cos_sims, expert_ovls = [], []
        for i in range(1, len(layer_records)):
            h1, h2 = layer_records[i - 1][1], layer_records[i][1]
            cos = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)
            e1, e2 = set(layer_records[i - 1][2]), set(layer_records[i][2])
            cos_sims.append(cos)
            expert_ovls.append(len(e1 & e2))
        if cos_sims:
            corr = np.corrcoef(cos_sims, expert_ovls)[0, 1]
            print(
                f"  Layer {test_layer:2d}: cos_sim={np.mean(cos_sims):.3f} "
                f"overlap={np.mean(expert_ovls):.1f}/{K} corr={corr:.3f}"
            )
    print()

    # Per-layer expert concentration
    print("=== Expert concentration per layer ===")
    for layer in range(0, num_layers, max(1, num_layers // 8)):
        layer_experts = [e for ti, li, _, _, experts in records if li == layer for e in experts]
        unique = len(set(layer_experts))
        total = len(layer_experts)
        from collections import Counter

        counts = Counter(layer_experts)
        top10 = sum(c for _, c in counts.most_common(10))
        print(
            f"  Layer {layer:2d}: {unique} unique experts, "
            f"top-10 cover {top10/total*100:.0f}%"
        )


if __name__ == "__main__":
    main()
