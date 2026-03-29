# Dequant Kernel Improvements from torchao/MLX Analysis

## Source Files

**torchao Metal kernels:**
- `../ao/torchao/experimental/kernels/mps/metal/qmv.metal` — MLX-derived matvec kernel (805 lines)
  - `load_vector` (lines 19-118): pre-divides x by bit-position powers
  - `qdot` (lines 226-366): inner dot product, mask-without-shift
  - `qmv_fast` (lines 511-579): main kernel — 2 SIMD groups, 4 rows each
- `../ao/torchao/experimental/kernels/mps/metal/int4mm_opt.metal` — 4-bit matmul (191 lines)
  - `act_div_scales` (line 110): the inverse scale constants `{1, 1/16, 1/256, 1/4096}`
  - zero-point comment (line 123): "Adding zero point results in 10% perf penalty"
  - inner loop (lines 131-156): pre-multiply x, mask weights without shifting, separate bias
- `../ao/torchao/experimental/kernels/mps/src/dispatch.h` — dispatch config for Metal threadgroups

**Current Flash-MoE kernel:**
- `metal_infer/shaders.metal` — `dequant_matvec_4bit_v3` (lines 251-340)

## Technique 1: Inverse Scale Pre-multiplication (eliminate shifts)

**What it does:** Instead of shifting each nibble right and converting to float, mask the nibble in place and pre-divide the x value to compensate.

**torchao int4mm_opt.metal lines 108-133:**
```metal
// MLX trick: avoid right-shifting weights by pre-dividing activations
float4 act_div_scales = {1.f, 1 / 16.f, 1 / 256.f, 1 / 4096.f};
float4 a_vec = a_val * act_div_scales;

// Weights masked but NOT shifted:
ushort b_val = ...;
b_mat[0] = scales[0] * float4(
    float(b_val & 0x000f),    // nibble 0: value 0-15
    float(b_val & 0x00f0),    // nibble 1: value 0-240 (still shifted left by 4)
    float(b_val & 0x0f00),    // nibble 2: still shifted left by 8
    float(b_val & 0xf000));   // nibble 3: still shifted left by 12

// The pre-divided a_vec compensates: a_vec[1] = x/16, so x/16 * (nibble<<4) = x * nibble
result += a_vec * b_mat;
```

**Current Flash-MoE shaders.metal lines 310-330:**
```metal
// Current: shift + convert + FMA per nibble
float sx0 = scale * x_shared[x_base + 0];
float bx0 = bias * x_shared[x_base + 0];
acc += fma(float((packed >> 0) & 0xF), sx0, bx0);   // shift by 0
acc += fma(float((packed >> 4) & 0xF), sx1, bx1);   // shift by 4
acc += fma(float((packed >> 8) & 0xF), sx2, bx2);   // shift by 8
// ...
```

**How to adapt:** For a uint32 with 8 nibbles (our format), pre-divide x by `{1, 1/16, 1/256, 1/4096, 1/65536, 1/1048576, 1/16777216, 1/268435456}`. Mask as `packed & 0xF`, `packed & 0xF0`, etc. without shifting. Eliminates 7 shifts and 8 uint-to-float conversions.

**Note:** torchao uses uint16 (2 nibbles per short) while Flash-MoE uses uint32 (8 nibbles). The principle extends to 8 pre-division factors.

## Technique 2: Separate Bias/Zero-point Accumulation

**torchao int4mm_opt.metal lines 134, 155:**
```metal
float a_val_sum = a_val[0] + a_val[1] + a_val[2] + a_val[3];  // sum of x values
// ... weight * pre-divided-x accumulation ...
result += a_val_sum * zeros_float;  // bias applied ONCE at end
```

**Current Flash-MoE:** `bx = bias * x` computed per nibble, added via FMA per element.

**How to adapt:** Compute `sum(x_shared[col..col+7])` once per packed uint32. At end of group: `acc += x_sum * bias`. Saves 7 multiplies per uint32 (bias * x for each of 8 nibbles → 1 multiply for sum * bias).

## Technique 3: Multiple Rows Per SIMD Group

**torchao qmv.metal lines 527-528, 557-564:**
```metal
constexpr int num_simdgroups = 2;
constexpr int results_per_simdgroup = 4;
// Each SIMD group processes 4 output rows:
for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] += qdot(ws + row * stride, x_thread, s, b, sum);
}
```

**Current Flash-MoE:** 1 row per SIMD group (`ROWS_PER_TG=8`, 8 SIMD groups).

**How to adapt:** Each SIMD group loads x once, processes 4 rows of weights. Amortizes x load cost 4x. Threadgroup becomes 2 SIMD groups × 4 rows = 8 rows (same total) but with 64 threads instead of 256. The x data lives in thread-local registers, not shared memory — eliminates the barrier.

## Technique 4: Thread-Local x Instead of Shared Memory

**torchao qmv.metal line 539:**
```metal
thread U x_thread[values_per_thread];  // private per-thread registers
```

**Current Flash-MoE shaders.metal line 274:**
```metal
threadgroup float x_shared[4096];  // shared across threadgroup, requires barrier
```

Thread-local avoids the `threadgroup_barrier` cost. Each thread loads its own portion of x from global memory. With 32 lanes per SIMD, each lane handles `in_dim/32` elements — for in_dim=2048 that's 64 floats per thread (256 bytes, fits in registers).

## Priority Order for Experiments

1. **Inverse scale pre-multiplication** — highest expected impact, eliminates 7 shifts + 8 int-to-float per uint32
2. **Separate bias accumulation** — easy to add alongside #1, saves 7 multiplies per uint32
3. **4 rows per SIMD** — larger change, potentially best bandwidth improvement but requires restructuring dispatch
4. **Thread-local x** — natural companion to #3, eliminates barrier

## Caveat

GPU time is 35% of total layer time. SSD I/O is 63%. Even a 2x kernel speedup saves only ~1.15ms/layer (18% total). Worth doing but the I/O bottleneck dominates.
