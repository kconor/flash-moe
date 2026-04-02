#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# ///
"""Build expert_index.json from a model's safetensors files.

Scans the weight index for expert tensors (switch_mlp.*) and records their
byte locations in each shard file. Output is consumed by repack_experts.py.

Usage:
    ./build_expert_index.py --model ../models/Qwen3.5-35B-A3B-4bit
"""

import argparse
import json
import struct
import os
import sys
from pathlib import Path
from collections import defaultdict


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


DTYPE_SIZES = {"U32": 4, "BF16": 2, "F32": 4, "F16": 2, "I32": 4}


def main():
    parser = argparse.ArgumentParser(description="Build expert_index.json from safetensors")
    parser.add_argument('--model', required=True, help='Path to model directory')
    parser.add_argument('--output', default=None, help='Output path (default: model_dir/expert_index.json)')
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = args.output or str(model_path / 'expert_index.json')

    # Load weight index — support both sharded and single-file models
    index_path = model_path / 'model.safetensors.index.json'
    single_path = model_path / 'model.safetensors'
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx['weight_map']
    elif single_path.exists():
        header, _ = parse_safetensors_header(str(single_path))
        header.pop('__metadata__', None)
        weight_map = {name: 'model.safetensors' for name in header.keys()}
        print(f"Single safetensors file: {len(weight_map)} tensors")
    else:
        print(f"ERROR: no model.safetensors or index.json in {model_path}", file=sys.stderr)
        sys.exit(1)

    # Find expert tensors: pattern is *.layers.{L}.mlp.switch_mlp.{component}
    # component is one of: gate_proj.weight, gate_proj.scales, gate_proj.biases,
    #                       up_proj.weight, up_proj.scales, up_proj.biases,
    #                       down_proj.weight, down_proj.scales, down_proj.biases
    import re
    expert_pattern = re.compile(
        r'(?:language_model\.)?model\.layers\.(\d+)\.mlp\.switch_mlp\.'
        r'((?:gate|up|down)_proj\.(?:weight|scales|biases))$'
    )

    expert_tensors = defaultdict(dict)  # layer -> comp -> (name, filename)
    for name, filename in weight_map.items():
        m = expert_pattern.search(name)
        if m:
            layer = int(m.group(1))
            comp = m.group(2)
            expert_tensors[layer][comp] = (name, filename)

    if not expert_tensors:
        print("ERROR: No expert tensors found in weight index", file=sys.stderr)
        sys.exit(1)

    num_layers = max(expert_tensors.keys()) + 1
    components_per_layer = len(next(iter(expert_tensors.values())))
    print(f"Found expert tensors: {num_layers} layers, {components_per_layer} components each")

    # Parse safetensors headers to get exact byte offsets
    header_cache = {}
    needed_files = set()
    for layer_comps in expert_tensors.values():
        for _, filename in layer_comps.values():
            needed_files.add(filename)

    for filename in sorted(needed_files):
        filepath = model_path / filename
        header_cache[filename] = parse_safetensors_header(str(filepath))
    print(f"Parsed {len(header_cache)} safetensors headers")

    # Build expert_reads index
    expert_reads = {}
    for layer in sorted(expert_tensors.keys()):
        layer_info = {}
        for comp, (tensor_name, filename) in expert_tensors[layer].items():
            header, data_start = header_cache[filename]

            if tensor_name not in header:
                print(f"WARNING: {tensor_name} not in {filename}")
                continue

            meta = header[tensor_name]
            offsets = meta['data_offsets']
            shape = meta['shape']
            dtype = meta['dtype']

            tensor_bytes = offsets[1] - offsets[0]
            abs_offset = data_start + offsets[0]

            # Shape is [num_experts, ...rest]
            num_experts = shape[0]
            expert_size = tensor_bytes // num_experts
            expert_stride = expert_size  # contiguous in memory

            layer_info[comp] = {
                "file": filename,
                "abs_offset": abs_offset,
                "expert_stride": expert_stride,
                "expert_size": expert_size,
                "shape": shape,
                "dtype": dtype,
                "num_experts": num_experts,
            }

        expert_reads[str(layer)] = layer_info

    # Write index
    index = {
        "model_path": str(model_path.resolve()),
        "expert_reads": expert_reads,
    }

    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {output_path}")
    print(f"  Layers: {len(expert_reads)}")
    if expert_reads:
        first = next(iter(expert_reads.values()))
        for comp, info in sorted(first.items()):
            print(f"  {comp}: {info['expert_size']} bytes/expert, "
                  f"shape={info['shape']}, dtype={info['dtype']}")


if __name__ == '__main__':
    main()
