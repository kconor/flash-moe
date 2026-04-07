/*
 * shaders.metal — Optimized Metal compute shaders for 4-bit quantized MoE inference
 *
 * Core operations:
 *   1. dequant_matvec_4bit: Naive 4-bit affine dequant matvec (reference)
 *   2. dequant_matvec_4bit_fast: SIMD-optimized with simd_sum reduction
 *   3. dequant_matvec_4bit_v3: Fully optimized — tiled threadgroup, vector loads,
 *      coalesced access, shared input cache. Target: <0.1ms per matmul.
 *   4. swiglu_fused / swiglu_fused_vec4: SwiGLU activation
 *   5. weighted_sum: combine expert outputs with routing weights
 *   6. rms_norm: RMS normalization
 *
 * Quantization format (MLX affine N-bit, group_size=64):
 *   - Weights stored as uint32, packed sequentially in bit order
 *   - For N-bit: each group of 64 values → 64*N/32 uint32 words
 *   - Per-group scale and bias in bfloat16
 *   - Dequantized value = uintN_val * scale + bias
 *   - Supports 2, 4, 5, 6, 8-bit (5/6-bit values may straddle uint32 boundaries)
 *
 * Matrix layout for expert projections:
 *   gate_proj/up_proj: [1024, 512] uint32 = [1024, 4096] logical (out=1024, in=4096)
 *   down_proj: [4096, 128] uint32 = [4096, 1024] logical (out=4096, in=1024)
 *
 *   Scales/biases: [out_dim, in_dim/group_size]
 *   gate/up scales: [1024, 64]   (4096/64 = 64 groups)
 *   down scales:    [4096, 16]   (1024/64 = 16 groups)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

inline uint16_t f32_to_bf16(float f) {
    return uint16_t(as_type<uint>(f) >> 16);
}


// ============================================================================
// Kernel 1: 4-bit dequantized matrix-vector multiply (NAIVE — reference)
// ============================================================================

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    float acc = 0.0f;

    device const uint32_t* w_row = W_packed + tid * packed_cols;
    device const uint16_t* s_row = scales + tid * num_groups;
    device const uint16_t* b_row = biases + tid * num_groups;

    for (uint g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            for (uint n = 0; n < 8; n++) {
                uint nibble = (packed >> (n * 4)) & 0xF;
                float w_val = float(nibble) * scale + bias;
                acc += w_val * x[x_base + n];
            }
        }
    }

    out[tid] = acc;
}


// ============================================================================
// Kernel 1b: 4-bit dequant matvec — SIMD-optimized (legacy, kept for compat)
// ============================================================================

kernel void dequant_matvec_4bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            acc += (float((packed >>  0) & 0xF) * scale + bias) * x[x_base + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x[x_base + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x[x_base + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x[x_base + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x[x_base + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x[x_base + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x[x_base + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x[x_base + 7];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}

// ============================================================================
// Fused gate+up+SwiGLU: reads x ONCE, computes silu(gate(x)) * up(x)
// Saves one input read + one kernel dispatch per expert
// ============================================================================
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint16_t* gate_s    [[buffer(1)]],
    device const uint16_t* gate_b    [[buffer(2)]],
    device const uint32_t* up_W      [[buffer(3)]],
    device const uint16_t* up_s      [[buffer(4)]],
    device const uint16_t* up_b      [[buffer(5)]],
    device const float*    x         [[buffer(6)]],
    device float*          out       [[buffer(7)]],
    constant uint&         out_dim   [[buffer(8)]],
    constant uint&         in_dim    [[buffer(9)]],
    constant uint&         group_size [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint16_t* gs = gate_s + tgid * num_groups;
    device const uint16_t* gb = gate_b + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint16_t* us = up_s   + tgid * num_groups;
    device const uint16_t* ub = up_b   + tgid * num_groups;
    float ga = 0.0f, ua = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float gsc = bf16_to_f32(gs[g]), gbi = bf16_to_f32(gb[g]);
        float usc = bf16_to_f32(us[g]), ubi = bf16_to_f32(ub[g]);
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp+p], up = ur[bp+p];
            for (uint i = 0; i < 8; i++) {
                float xv = x[bx + p*8 + i];
                ga += (float((gp>>(i*4))&0xF)*gsc+gbi)*xv;
                ua += (float((up>>(i*4))&0xF)*usc+ubi)*xv;
            }
        }
    }
    threadgroup float sg[32], su[32];
    float rg = simd_sum(ga), ru = simd_sum(ua);
    uint sl = lid%32, si = lid/32, ns = (tg_size+31)/32;
    if (sl==0) { sg[si]=rg; su[si]=ru; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (si==0 && sl<ns) {
        float vg=simd_sum(sg[sl]), vu=simd_sum(su[sl]);
        if (sl==0) out[tgid] = (vg/(1.0f+exp(-vg))) * vu;
    }
}

// ============================================================================
// Kernel 1c: FULLY OPTIMIZED 4-bit dequant matvec
// ============================================================================
//
// Design for M3 Max (40-core GPU, SIMD width 32):
//
// Strategy: Each threadgroup handles ROWS_PER_TG output rows.
//   - Threadgroup size = 256 (8 SIMD groups of 32)
//   - Each SIMD group handles one output row
//   - Within a SIMD group, 32 threads split the input dimension
//   - Each thread processes in_dim/32 input elements using vector loads
//   - Reduction via simd_sum (single instruction)
//
// Memory optimizations:
//   - Input vector x cached in threadgroup shared memory (loaded once)
//   - uint4 vector loads for weights (128 bits = 32 nibbles per load)
//   - float4 vector loads for x (128 bits = 4 floats per load)
//   - Coalesced weight reads: adjacent threads read adjacent uint4 vectors
//
// For gate/up_proj [1024, 4096]: 1024/8 = 128 threadgroups, 256 threads each
//   - 128 * 256 = 32768 threads across 40 cores = good occupancy
//   - Each thread processes 4096/32 = 128 input elements = 16 uint32 packed words
//     = 4 uint4 loads per thread per row
//
// For down_proj [4096, 1024]: 4096/8 = 512 threadgroups
//   - Each thread processes 1024/32 = 32 input elements = 4 uint32 packed words
//     = 1 uint4 load per thread per row

// Number of output rows per threadgroup = number of SIMD groups (256/32 = 8)
#define ROWS_PER_TG 8

kernel void dequant_matvec_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],     // which tile of rows
    uint lid    [[thread_position_in_threadgroup]],    // 0..255
    uint simd_lane  [[thread_index_in_simdgroup]],    // 0..31
    uint simd_group [[simdgroup_index_in_threadgroup]] // 0..7
) {
    // Which output row this SIMD group handles
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;      // uint32 columns per row
    uint num_groups  = in_dim / group_size;

    // ---- Cache input vector in threadgroup shared memory ----
    // Max in_dim = 4096, so we need 4096 floats = 16KB shared memory
    // This is well within the 32KB threadgroup memory limit on M3
    threadgroup float x_shared[4096];

    // Cooperative load: 256 threads load 4096 floats (16 per thread)
    // ALL threads must participate in this load + barrier, even if their
    // row is out of bounds. Early return before the barrier causes only
    // partial loading of x_shared, corrupting results for valid rows.
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now safe to bail out for out-of-bounds rows
    if (row >= out_dim) return;

    // ---- Pointer setup for this row ----
    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    // ---- Each lane processes a strided slice of the packed columns ----
    // Lane k processes columns: k, k+32, k+64, ...
    // This gives coalesced reads: adjacent lanes read adjacent uint32 words.

    float acc = 0.0f;

    // Process packed columns in strides of 32 (one per SIMD lane)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // Determine which group this column belongs to
        // packed_per_group = group_size / 8 = 64 / 8 = 8
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        // Dequantize 8 nibbles and multiply with cached x
        // Rearranged: (nibble * scale + bias) * x = nibble * (scale*x) + bias*x
        // Pre-compute scale*x and bias*x, then use FMA for dequant+multiply in one op.
        // This reduces per-nibble from (convert + mul + add + mul + add) to (convert + FMA + add).
        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    // ---- SIMD reduction: sum across 32 lanes ----
    float sum = simd_sum(acc);

    // Lane 0 writes the result
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1f: 4-bit dequant matvec with LUT (eliminates uint→float conversions)
// ============================================================================
// Instead of converting each nibble to float (expensive conversion instruction),
// pre-compute a 16-entry LUT per group: lut[v] = float(v) * scale + bias.
// Then inner loop is just: acc += lut[nibble] * x_shared[i] — pure math, no conversions.
// The LUT is recomputed every group_size/8 iterations (amortized).

#define ROWS_PER_TG_V5 8

kernel void dequant_matvec_4bit_v5(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_V5 + simd_group;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;
    uint prev_g = 0xFFFFFFFF;
    float lut[16];

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_group;

        // Rebuild LUT when group changes
        if (g != prev_g) {
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);
            for (uint v = 0; v < 16; v++) {
                lut[v] = float(v) * scale + bias;
            }
            prev_g = g;
        }

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += lut[(packed >>  0) & 0xF] * x_shared[x_base + 0];
        acc += lut[(packed >>  4) & 0xF] * x_shared[x_base + 1];
        acc += lut[(packed >>  8) & 0xF] * x_shared[x_base + 2];
        acc += lut[(packed >> 12) & 0xF] * x_shared[x_base + 3];
        acc += lut[(packed >> 16) & 0xF] * x_shared[x_base + 4];
        acc += lut[(packed >> 20) & 0xF] * x_shared[x_base + 5];
        acc += lut[(packed >> 24) & 0xF] * x_shared[x_base + 6];
        acc += lut[(packed >> 28) & 0xF] * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 1e: 2-bit affine dequant matvec (same structure as v3)
// ============================================================================
// Packs 16 x 2-bit values per uint32. Each value is 0-3, dequantized as:
//   val = uint2 * scale + bias (same affine quantization, just 2-bit range)
// Same group structure: group_size elements share one (scale, bias) pair.
// packed_cols = in_dim / 16 (16 values per uint32, vs 8 for 4-bit)

kernel void dequant_matvec_2bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/16]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 16;  // 16 values per uint32 for 2-bit
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    // Each lane processes strided columns (16 values per uint32)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // group_size/16 packed words per group
        uint g = col / (group_size / 16);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 16;

        // Unroll 16 x 2-bit extractions
        acc += (float((packed >>  0) & 0x3) * scale + bias) * x_shared[x_base +  0];
        acc += (float((packed >>  2) & 0x3) * scale + bias) * x_shared[x_base +  1];
        acc += (float((packed >>  4) & 0x3) * scale + bias) * x_shared[x_base +  2];
        acc += (float((packed >>  6) & 0x3) * scale + bias) * x_shared[x_base +  3];
        acc += (float((packed >>  8) & 0x3) * scale + bias) * x_shared[x_base +  4];
        acc += (float((packed >> 10) & 0x3) * scale + bias) * x_shared[x_base +  5];
        acc += (float((packed >> 12) & 0x3) * scale + bias) * x_shared[x_base +  6];
        acc += (float((packed >> 14) & 0x3) * scale + bias) * x_shared[x_base +  7];
        acc += (float((packed >> 16) & 0x3) * scale + bias) * x_shared[x_base +  8];
        acc += (float((packed >> 18) & 0x3) * scale + bias) * x_shared[x_base +  9];
        acc += (float((packed >> 20) & 0x3) * scale + bias) * x_shared[x_base + 10];
        acc += (float((packed >> 22) & 0x3) * scale + bias) * x_shared[x_base + 11];
        acc += (float((packed >> 24) & 0x3) * scale + bias) * x_shared[x_base + 12];
        acc += (float((packed >> 26) & 0x3) * scale + bias) * x_shared[x_base + 13];
        acc += (float((packed >> 28) & 0x3) * scale + bias) * x_shared[x_base + 14];
        acc += (float((packed >> 30) & 0x3) * scale + bias) * x_shared[x_base + 15];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1g: 8-bit affine dequant matvec (same structure as v3)
// ============================================================================
// Packs 4 x 8-bit values per uint32. Each value is 0-255, dequantized as:
//   val = uint8 * scale + bias (same affine quantization, 8-bit range)
// Same group structure: group_size elements share one (scale, bias) pair.

kernel void dequant_matvec_8bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/4]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 4;  // 4 values per uint32 for 8-bit
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // group_size/4 packed words per group
        uint g = col / (group_size / 4);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 4;

        // FMA optimization: (uint8 * scale + bias) * x = uint8 * (scale*x) + bias*x
        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];

        acc += fma(float((packed >>  0) & 0xFF), sx0, bx0);
        acc += fma(float((packed >>  8) & 0xFF), sx1, bx1);
        acc += fma(float((packed >> 16) & 0xFF), sx2, bx2);
        acc += fma(float((packed >> 24) & 0xFF), sx3, bx3);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 1g_opt: 8-bit affine dequant matvec — multi-row SIMD (MLX-style)
// ============================================================================
// Each SIMD group (32 lanes) computes 4 output rows in parallel, sharing one
// load of the x cache across all 4 rows. 8 SIMD groups/TG × 4 rows = 32 rows/TG.
// Uses the affine decomposition:
//     out[r] = sum_g ( scale[r,g] * dot(w_int[r,g], x[g]) + bias[r,g] * sum(x[g]) )
// The scale*x / bias*x precompute is replaced with one mul per row per col
// (for the int*x dot) + one fma per col per row (for scale*dot + bias*sum_x).
// For gate/up/qkv matmuls (in_dim=4096, out_dim=4096..12288), this cuts the
// number of threadgroups by 4× (and therefore the number of x_shared loads
// from global memory by 4×), which is the dominant win.

#define ROWS_PER_SIMD_8BIT 4
#define SIMDS_PER_TG_8BIT 8
#define ROWS_PER_TG_8BIT_OPT (ROWS_PER_SIMD_8BIT * SIMDS_PER_TG_8BIT)  // 32

kernel void dequant_matvec_8bit_opt(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/4]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Cache input vector in threadgroup shared memory (once per TG).
    threadgroup float x_shared[4096];
    uint tg_threads = SIMDS_PER_TG_8BIT * 32;  // 64 or 256
    for (uint i = lid; i < in_dim; i += tg_threads) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_base = tgid * ROWS_PER_TG_8BIT_OPT + simd_group * ROWS_PER_SIMD_8BIT;
    if (row_base >= out_dim) return;

    uint packed_cols    = in_dim / 4;              // uint32 columns per row
    uint num_groups     = in_dim / group_size;
    uint packs_per_group = group_size / 4;         // 16 for group_size=64

    // Clamp tail rows to keep reads in-bounds if out_dim is not a multiple of 4.
    uint r0 = row_base + 0;
    uint r1 = min(row_base + 1, out_dim - 1);
    uint r2 = min(row_base + 2, out_dim - 1);
    uint r3 = min(row_base + 3, out_dim - 1);

    // Row pointers for the (up to) 4 output rows handled by this SIMD group.
    device const uint32_t* w0 = W_packed + r0 * packed_cols;
    device const uint32_t* w1 = W_packed + r1 * packed_cols;
    device const uint32_t* w2 = W_packed + r2 * packed_cols;
    device const uint32_t* w3 = W_packed + r3 * packed_cols;
    device const uint16_t* s0 = scales + r0 * num_groups;
    device const uint16_t* s1 = scales + r1 * num_groups;
    device const uint16_t* s2 = scales + r2 * num_groups;
    device const uint16_t* s3 = scales + r3 * num_groups;
    device const uint16_t* b0 = biases + r0 * num_groups;
    device const uint16_t* b1 = biases + r1 * num_groups;
    device const uint16_t* b2 = biases + r2 * num_groups;
    device const uint16_t* b3 = biases + r3 * num_groups;

    float4 acc = float4(0.0);

    // Stride-32 uint32 loads (lower register pressure than uint4). 4 values
    // per iteration per lane, amortizing scale/bias loads across 16 packed
    // columns per group. For the 9B's HIDDEN_DIM=4096 input, each lane does
    // packed_cols/32 = 32 iterations.
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packs_per_group;

        threadgroup const float4* xp = (threadgroup const float4*)(x_shared + col * 4);
        float4 xv = *xp;
        float x_sum = xv.x + xv.y + xv.z + xv.w;

        float4 scale = float4(
            bf16_to_f32(s0[g]), bf16_to_f32(s1[g]),
            bf16_to_f32(s2[g]), bf16_to_f32(s3[g]));
        float4 bias = float4(
            bf16_to_f32(b0[g]), bf16_to_f32(b1[g]),
            bf16_to_f32(b2[g]), bf16_to_f32(b3[g]));

        uint p0 = w0[col];
        uint p1 = w1[col];
        uint p2 = w2[col];
        uint p3 = w3[col];

        float d0 = float((p0      ) & 0xFF) * xv.x
                 + float((p0 >>  8) & 0xFF) * xv.y
                 + float((p0 >> 16) & 0xFF) * xv.z
                 + float((p0 >> 24) & 0xFF) * xv.w;
        float d1 = float((p1      ) & 0xFF) * xv.x
                 + float((p1 >>  8) & 0xFF) * xv.y
                 + float((p1 >> 16) & 0xFF) * xv.z
                 + float((p1 >> 24) & 0xFF) * xv.w;
        float d2 = float((p2      ) & 0xFF) * xv.x
                 + float((p2 >>  8) & 0xFF) * xv.y
                 + float((p2 >> 16) & 0xFF) * xv.z
                 + float((p2 >> 24) & 0xFF) * xv.w;
        float d3 = float((p3      ) & 0xFF) * xv.x
                 + float((p3 >>  8) & 0xFF) * xv.y
                 + float((p3 >> 16) & 0xFF) * xv.z
                 + float((p3 >> 24) & 0xFF) * xv.w;

        acc.x = fma(scale.x, d0, fma(bias.x, x_sum, acc.x));
        acc.y = fma(scale.y, d1, fma(bias.y, x_sum, acc.y));
        acc.z = fma(scale.z, d2, fma(bias.z, x_sum, acc.z));
        acc.w = fma(scale.w, d3, fma(bias.w, x_sum, acc.w));
    }

    // Per-row SIMD reduction.
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);

    if (simd_lane == 0) {
        // Bounds-check the writes in case out_dim is not a multiple of 4.
        if (row_base + 0 < out_dim) out[row_base + 0] = acc.x;
        if (row_base + 1 < out_dim) out[row_base + 1] = acc.y;
        if (row_base + 2 < out_dim) out[row_base + 2] = acc.z;
        if (row_base + 3 < out_dim) out[row_base + 3] = acc.w;
    }
}


// ============================================================================
// 8-bit matvec for large in_dim (> 4096) — multi-row SIMD, no shared x cache.
// ============================================================================
// Same multi-row pattern as dequant_matvec_8bit_opt but without the threadgroup
// x cache (the input is too big to fit in shared memory). Each SIMD group
// computes 4 output rows in parallel, using thread registers for x; x is still
// re-read per SIMD group within the TG (no way to avoid that without shared).
//
// Primarily used for down_proj (4096×12288) in dense FFNs at large intermediate
// sizes.

// ============================================================================
// Fused gate_proj + up_proj for dense MLPs. Both matmuls share the same
// input x, so we share the threadgroup x cache and cut the number of
// threadgroups in half (each TG produces 32 gate rows AND 32 up rows).
// Saves one dispatch and ~half the x-cache fill bandwidth.
kernel void dequant_matvec_8bit_gate_up(
    device const uint32_t* W_gate     [[buffer(0)]],
    device const uint16_t* S_gate     [[buffer(1)]],
    device const uint16_t* B_gate     [[buffer(2)]],
    device const uint32_t* W_up       [[buffer(3)]],
    device const uint16_t* S_up       [[buffer(4)]],
    device const uint16_t* B_up       [[buffer(5)]],
    device const float*    x          [[buffer(6)]],
    device float*          out_gate   [[buffer(7)]],
    device float*          out_up     [[buffer(8)]],
    constant uint&         out_dim    [[buffer(9)]],
    constant uint&         in_dim     [[buffer(10)]],
    constant uint&         group_size [[buffer(11)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) x_shared[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_base = tgid * 32 + simd_group * 4;
    if (row_base >= out_dim) return;

    uint packed_cols     = in_dim / 4;
    uint num_groups      = in_dim / group_size;
    uint packs_per_group = group_size / 4;

    uint r0 = row_base + 0;
    uint r1 = min(row_base + 1, out_dim - 1);
    uint r2 = min(row_base + 2, out_dim - 1);
    uint r3 = min(row_base + 3, out_dim - 1);

    device const uint32_t* wg0 = W_gate + r0 * packed_cols;
    device const uint32_t* wg1 = W_gate + r1 * packed_cols;
    device const uint32_t* wg2 = W_gate + r2 * packed_cols;
    device const uint32_t* wg3 = W_gate + r3 * packed_cols;
    device const uint32_t* wu0 = W_up + r0 * packed_cols;
    device const uint32_t* wu1 = W_up + r1 * packed_cols;
    device const uint32_t* wu2 = W_up + r2 * packed_cols;
    device const uint32_t* wu3 = W_up + r3 * packed_cols;

    device const uint16_t* sg0 = S_gate + r0 * num_groups;
    device const uint16_t* sg1 = S_gate + r1 * num_groups;
    device const uint16_t* sg2 = S_gate + r2 * num_groups;
    device const uint16_t* sg3 = S_gate + r3 * num_groups;
    device const uint16_t* bg0 = B_gate + r0 * num_groups;
    device const uint16_t* bg1 = B_gate + r1 * num_groups;
    device const uint16_t* bg2 = B_gate + r2 * num_groups;
    device const uint16_t* bg3 = B_gate + r3 * num_groups;

    device const uint16_t* su0 = S_up + r0 * num_groups;
    device const uint16_t* su1 = S_up + r1 * num_groups;
    device const uint16_t* su2 = S_up + r2 * num_groups;
    device const uint16_t* su3 = S_up + r3 * num_groups;
    device const uint16_t* bu0 = B_up + r0 * num_groups;
    device const uint16_t* bu1 = B_up + r1 * num_groups;
    device const uint16_t* bu2 = B_up + r2 * num_groups;
    device const uint16_t* bu3 = B_up + r3 * num_groups;

    float4 acc_g = float4(0.0);
    float4 acc_u = float4(0.0);

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packs_per_group;

        threadgroup const float4* xp = (threadgroup const float4*)(x_shared + col * 4);
        float4 xv = *xp;
        float x_sum = xv.x + xv.y + xv.z + xv.w;

        float4 scale_g = float4(
            bf16_to_f32(sg0[g]), bf16_to_f32(sg1[g]),
            bf16_to_f32(sg2[g]), bf16_to_f32(sg3[g]));
        float4 bias_g = float4(
            bf16_to_f32(bg0[g]), bf16_to_f32(bg1[g]),
            bf16_to_f32(bg2[g]), bf16_to_f32(bg3[g]));
        float4 scale_u = float4(
            bf16_to_f32(su0[g]), bf16_to_f32(su1[g]),
            bf16_to_f32(su2[g]), bf16_to_f32(su3[g]));
        float4 bias_u = float4(
            bf16_to_f32(bu0[g]), bf16_to_f32(bu1[g]),
            bf16_to_f32(bu2[g]), bf16_to_f32(bu3[g]));

        uint pg0 = wg0[col], pg1 = wg1[col], pg2 = wg2[col], pg3 = wg3[col];
        uint pu0 = wu0[col], pu1 = wu1[col], pu2 = wu2[col], pu3 = wu3[col];

        #define Q8DOT(p, xv) \
            (float((p) & 0xFF) * xv.x + \
             float(((p) >> 8) & 0xFF) * xv.y + \
             float(((p) >> 16) & 0xFF) * xv.z + \
             float(((p) >> 24) & 0xFF) * xv.w)

        float d0g = Q8DOT(pg0, xv);
        float d1g = Q8DOT(pg1, xv);
        float d2g = Q8DOT(pg2, xv);
        float d3g = Q8DOT(pg3, xv);
        float d0u = Q8DOT(pu0, xv);
        float d1u = Q8DOT(pu1, xv);
        float d2u = Q8DOT(pu2, xv);
        float d3u = Q8DOT(pu3, xv);
        #undef Q8DOT

        acc_g.x = fma(scale_g.x, d0g, fma(bias_g.x, x_sum, acc_g.x));
        acc_g.y = fma(scale_g.y, d1g, fma(bias_g.y, x_sum, acc_g.y));
        acc_g.z = fma(scale_g.z, d2g, fma(bias_g.z, x_sum, acc_g.z));
        acc_g.w = fma(scale_g.w, d3g, fma(bias_g.w, x_sum, acc_g.w));
        acc_u.x = fma(scale_u.x, d0u, fma(bias_u.x, x_sum, acc_u.x));
        acc_u.y = fma(scale_u.y, d1u, fma(bias_u.y, x_sum, acc_u.y));
        acc_u.z = fma(scale_u.z, d2u, fma(bias_u.z, x_sum, acc_u.z));
        acc_u.w = fma(scale_u.w, d3u, fma(bias_u.w, x_sum, acc_u.w));
    }

    acc_g.x = simd_sum(acc_g.x);
    acc_g.y = simd_sum(acc_g.y);
    acc_g.z = simd_sum(acc_g.z);
    acc_g.w = simd_sum(acc_g.w);
    acc_u.x = simd_sum(acc_u.x);
    acc_u.y = simd_sum(acc_u.y);
    acc_u.z = simd_sum(acc_u.z);
    acc_u.w = simd_sum(acc_u.w);

    if (simd_lane == 0) {
        if (row_base + 0 < out_dim) { out_gate[row_base + 0] = acc_g.x; out_up[row_base + 0] = acc_u.x; }
        if (row_base + 1 < out_dim) { out_gate[row_base + 1] = acc_g.y; out_up[row_base + 1] = acc_u.y; }
        if (row_base + 2 < out_dim) { out_gate[row_base + 2] = acc_g.z; out_up[row_base + 2] = acc_u.z; }
        if (row_base + 3 < out_dim) { out_gate[row_base + 3] = acc_g.w; out_up[row_base + 3] = acc_u.w; }
    }
}


// ============================================================================
// Fused SwiGLU + dequant matvec (Q8) for the down_proj slot in dense MLPs.
// Reads gate[] and up[] directly instead of a pre-computed swiglu buffer.
// Each input position: act_i = silu(gate[i]) * up[i] computed inline, then
// dotted with the Q8 down_proj weights. Saves:
//   - one swiglu dispatch
//   - one intermediate buffer (buf_shared_act) write + read
// Same tiled x-cache structure as dequant_matvec_8bit_fast_opt.
kernel void dequant_matvec_8bit_fast_swiglu_down(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    gate_in    [[buffer(3)]],  // buf_shared_gate
    device const float*    up_in      [[buffer(4)]],  // buf_shared_up
    device float*          out        [[buffer(5)]],  // buf_shared_out
    constant uint&         out_dim    [[buffer(6)]],
    constant uint&         in_dim     [[buffer(7)]],
    constant uint&         group_size [[buffer(8)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float x_tile[4096];  // holds swiglu(gate, up) for current tile

    uint row_base = tgid * ROWS_PER_TG_8BIT_OPT + simd_group * ROWS_PER_SIMD_8BIT;
    uint packed_cols    = in_dim / 4;
    uint num_groups     = in_dim / group_size;
    uint packs_per_group = group_size / 4;

    uint r0 = row_base + 0;
    uint r1 = min(row_base + 1, out_dim - 1);
    uint r2 = min(row_base + 2, out_dim - 1);
    uint r3 = min(row_base + 3, out_dim - 1);

    bool row_active = (row_base < out_dim);

    device const uint32_t* w0 = W_packed + r0 * packed_cols;
    device const uint32_t* w1 = W_packed + r1 * packed_cols;
    device const uint32_t* w2 = W_packed + r2 * packed_cols;
    device const uint32_t* w3 = W_packed + r3 * packed_cols;
    device const uint16_t* s0 = scales + r0 * num_groups;
    device const uint16_t* s1 = scales + r1 * num_groups;
    device const uint16_t* s2 = scales + r2 * num_groups;
    device const uint16_t* s3 = scales + r3 * num_groups;
    device const uint16_t* b0 = biases + r0 * num_groups;
    device const uint16_t* b1 = biases + r1 * num_groups;
    device const uint16_t* b2 = biases + r2 * num_groups;
    device const uint16_t* b3 = biases + r3 * num_groups;

    float4 acc = float4(0.0);

    const uint TILE = 4096;

    for (uint tile_base = 0; tile_base < in_dim; tile_base += TILE) {
        uint tile_len = min(TILE, in_dim - tile_base);

        // Cooperative load of swiglu(gate, up)[tile_base .. tile_base+tile_len).
        for (uint i = lid; i < tile_len; i += 256) {
            float g = gate_in[tile_base + i];
            float u = up_in[tile_base + i];
            // silu(g) * u = (g / (1 + exp(-g))) * u
            float sig = 1.0f / (1.0f + exp(-g));
            x_tile[i] = g * sig * u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_active) {
            uint tile_packed = tile_len / 4;
            uint col_start = tile_base / 4;
            for (uint col = simd_lane; col < tile_packed; col += 32) {
                uint abs_col = col_start + col;
                uint g = abs_col / packs_per_group;

                threadgroup const float4* xp = (threadgroup const float4*)(x_tile + col * 4);
                float4 xvals = *xp;
                float x_sum = xvals.x + xvals.y + xvals.z + xvals.w;

                float4 scale = float4(
                    bf16_to_f32(s0[g]), bf16_to_f32(s1[g]),
                    bf16_to_f32(s2[g]), bf16_to_f32(s3[g]));
                float4 bias = float4(
                    bf16_to_f32(b0[g]), bf16_to_f32(b1[g]),
                    bf16_to_f32(b2[g]), bf16_to_f32(b3[g]));

                uint p0 = w0[abs_col];
                uint p1 = w1[abs_col];
                uint p2 = w2[abs_col];
                uint p3 = w3[abs_col];

                float d0 = float((p0      ) & 0xFF) * xvals.x
                         + float((p0 >>  8) & 0xFF) * xvals.y
                         + float((p0 >> 16) & 0xFF) * xvals.z
                         + float((p0 >> 24) & 0xFF) * xvals.w;
                float d1 = float((p1      ) & 0xFF) * xvals.x
                         + float((p1 >>  8) & 0xFF) * xvals.y
                         + float((p1 >> 16) & 0xFF) * xvals.z
                         + float((p1 >> 24) & 0xFF) * xvals.w;
                float d2 = float((p2      ) & 0xFF) * xvals.x
                         + float((p2 >>  8) & 0xFF) * xvals.y
                         + float((p2 >> 16) & 0xFF) * xvals.z
                         + float((p2 >> 24) & 0xFF) * xvals.w;
                float d3 = float((p3      ) & 0xFF) * xvals.x
                         + float((p3 >>  8) & 0xFF) * xvals.y
                         + float((p3 >> 16) & 0xFF) * xvals.z
                         + float((p3 >> 24) & 0xFF) * xvals.w;

                acc.x = fma(scale.x, d0, fma(bias.x, x_sum, acc.x));
                acc.y = fma(scale.y, d1, fma(bias.y, x_sum, acc.y));
                acc.z = fma(scale.z, d2, fma(bias.z, x_sum, acc.z));
                acc.w = fma(scale.w, d3, fma(bias.w, x_sum, acc.w));
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!row_active) return;

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);

    if (simd_lane == 0) {
        if (row_base + 0 < out_dim) out[row_base + 0] = acc.x;
        if (row_base + 1 < out_dim) out[row_base + 1] = acc.y;
        if (row_base + 2 < out_dim) out[row_base + 2] = acc.z;
        if (row_base + 3 < out_dim) out[row_base + 3] = acc.w;
    }
}

// Tiled variant: x is too large for shared cache in one chunk (in_dim > 4096),
// but we can process it in 4096-value tiles. Each tile caches its 16 KB of x
// into threadgroup memory once, every SIMD group in the TG reads from that
// cache, then advance to the next tile. For 32 rows/TG and 3 tiles (in_dim
// 12288), x bandwidth from global is reduced by ~32x vs untiled.
kernel void dequant_matvec_8bit_fast_opt(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float x_tile[4096];  // one tile of x at a time

    uint row_base = tgid * ROWS_PER_TG_8BIT_OPT + simd_group * ROWS_PER_SIMD_8BIT;
    uint packed_cols    = in_dim / 4;
    uint num_groups     = in_dim / group_size;
    uint packs_per_group = group_size / 4;

    uint r0 = row_base + 0;
    uint r1 = min(row_base + 1, out_dim - 1);
    uint r2 = min(row_base + 2, out_dim - 1);
    uint r3 = min(row_base + 3, out_dim - 1);

    // Bail out early? No — threads need to participate in the x_tile loads.
    bool row_active = (row_base < out_dim);

    device const uint32_t* w0 = W_packed + r0 * packed_cols;
    device const uint32_t* w1 = W_packed + r1 * packed_cols;
    device const uint32_t* w2 = W_packed + r2 * packed_cols;
    device const uint32_t* w3 = W_packed + r3 * packed_cols;
    device const uint16_t* s0 = scales + r0 * num_groups;
    device const uint16_t* s1 = scales + r1 * num_groups;
    device const uint16_t* s2 = scales + r2 * num_groups;
    device const uint16_t* s3 = scales + r3 * num_groups;
    device const uint16_t* b0 = biases + r0 * num_groups;
    device const uint16_t* b1 = biases + r1 * num_groups;
    device const uint16_t* b2 = biases + r2 * num_groups;
    device const uint16_t* b3 = biases + r3 * num_groups;

    float4 acc = float4(0.0);

    const uint TILE = 4096;
    const uint PACKED_PER_TILE = TILE / 4;  // 1024 uint32 per tile

    for (uint tile_base = 0; tile_base < in_dim; tile_base += TILE) {
        uint tile_len = min(TILE, in_dim - tile_base);

        // Cooperative load of x_tile[0..tile_len).
        for (uint i = lid; i < tile_len; i += 256) {
            x_tile[i] = x[tile_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_active) {
            uint tile_packed = tile_len / 4;
            uint col_start = tile_base / 4;
            for (uint col = simd_lane; col < tile_packed; col += 32) {
                uint abs_col = col_start + col;
                uint g = abs_col / packs_per_group;

                threadgroup const float4* xp = (threadgroup const float4*)(x_tile + col * 4);
                float4 xvals = *xp;
                float x_sum = xvals.x + xvals.y + xvals.z + xvals.w;

                float4 scale = float4(
                    bf16_to_f32(s0[g]), bf16_to_f32(s1[g]),
                    bf16_to_f32(s2[g]), bf16_to_f32(s3[g]));
                float4 bias = float4(
                    bf16_to_f32(b0[g]), bf16_to_f32(b1[g]),
                    bf16_to_f32(b2[g]), bf16_to_f32(b3[g]));

                uint p0 = w0[abs_col];
                uint p1 = w1[abs_col];
                uint p2 = w2[abs_col];
                uint p3 = w3[abs_col];

                float d0 = float((p0      ) & 0xFF) * xvals.x
                         + float((p0 >>  8) & 0xFF) * xvals.y
                         + float((p0 >> 16) & 0xFF) * xvals.z
                         + float((p0 >> 24) & 0xFF) * xvals.w;
                float d1 = float((p1      ) & 0xFF) * xvals.x
                         + float((p1 >>  8) & 0xFF) * xvals.y
                         + float((p1 >> 16) & 0xFF) * xvals.z
                         + float((p1 >> 24) & 0xFF) * xvals.w;
                float d2 = float((p2      ) & 0xFF) * xvals.x
                         + float((p2 >>  8) & 0xFF) * xvals.y
                         + float((p2 >> 16) & 0xFF) * xvals.z
                         + float((p2 >> 24) & 0xFF) * xvals.w;
                float d3 = float((p3      ) & 0xFF) * xvals.x
                         + float((p3 >>  8) & 0xFF) * xvals.y
                         + float((p3 >> 16) & 0xFF) * xvals.z
                         + float((p3 >> 24) & 0xFF) * xvals.w;

                acc.x = fma(scale.x, d0, fma(bias.x, x_sum, acc.x));
                acc.y = fma(scale.y, d1, fma(bias.y, x_sum, acc.y));
                acc.z = fma(scale.z, d2, fma(bias.z, x_sum, acc.z));
                acc.w = fma(scale.w, d3, fma(bias.w, x_sum, acc.w));
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!row_active) return;

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);

    if (simd_lane == 0) {
        if (row_base + 0 < out_dim) out[row_base + 0] = acc.x;
        if (row_base + 1 < out_dim) out[row_base + 1] = acc.y;
        if (row_base + 2 < out_dim) out[row_base + 2] = acc.z;
        if (row_base + 3 < out_dim) out[row_base + 3] = acc.w;
    }
}

// 8-bit variant of matvec_fast (for in_dim > 4096, no shared memory cache)
kernel void dequant_matvec_8bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 4;
    uint packed_cols = in_dim / 4;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 4;

            acc += fma(float((packed >>  0) & 0xFF), scale, bias) * x[x_base + 0];
            acc += fma(float((packed >>  8) & 0xFF), scale, bias) * x[x_base + 1];
            acc += fma(float((packed >> 16) & 0xFF), scale, bias) * x[x_base + 2];
            acc += fma(float((packed >> 24) & 0xFF), scale, bias) * x[x_base + 3];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}


// ============================================================================
// Kernel 1h: 6-bit affine dequant matvec — FMA-optimized, chunk-based
// ============================================================================
// MLX affine 6-bit, group_size=64: 64 values × 6 bits = 384 bits = 12 uint32s per group.
// Processes 16 values (3 uint32 words = 96 bits) per iteration with fully unrolled
// FMA extraction. Pre-computes scale*x and bias*x for all 16 positions, then uses
// hardware FMA for dequant+multiply in one instruction per value.
// Straddle values (5 and 10) use hardcoded cross-word extraction — no branches.

kernel void dequant_matvec_6bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size * 6 / 32;  // 12 for gs=64
    uint packed_cols = num_groups * packed_per_group;

    // 16 values per chunk, 4 chunks per group (64/16=4)
    uint num_chunks = in_dim / 16;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    // Each lane processes every 32nd chunk of 16 values (3 uint32 words)
    for (uint chunk = simd_lane; chunk < num_chunks; chunk += 32) {
        uint val_base = chunk * 16;
        uint g = val_base / group_size;
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        // Word offset: chunk_within_group * 3 words per chunk
        uint chunk_in_group = (val_base - g * group_size) / 16;
        device const uint32_t* gw = w_row + g * packed_per_group + chunk_in_group * 3;
        uint32_t w0 = gw[0], w1 = gw[1], w2 = gw[2];

        // Pre-compute scale*x and bias*x for all 16 positions
        float sx00 = scale * x_shared[val_base +  0]; float bx00 = bias * x_shared[val_base +  0];
        float sx01 = scale * x_shared[val_base +  1]; float bx01 = bias * x_shared[val_base +  1];
        float sx02 = scale * x_shared[val_base +  2]; float bx02 = bias * x_shared[val_base +  2];
        float sx03 = scale * x_shared[val_base +  3]; float bx03 = bias * x_shared[val_base +  3];
        float sx04 = scale * x_shared[val_base +  4]; float bx04 = bias * x_shared[val_base +  4];
        float sx05 = scale * x_shared[val_base +  5]; float bx05 = bias * x_shared[val_base +  5];
        float sx06 = scale * x_shared[val_base +  6]; float bx06 = bias * x_shared[val_base +  6];
        float sx07 = scale * x_shared[val_base +  7]; float bx07 = bias * x_shared[val_base +  7];
        float sx08 = scale * x_shared[val_base +  8]; float bx08 = bias * x_shared[val_base +  8];
        float sx09 = scale * x_shared[val_base +  9]; float bx09 = bias * x_shared[val_base +  9];
        float sx10 = scale * x_shared[val_base + 10]; float bx10 = bias * x_shared[val_base + 10];
        float sx11 = scale * x_shared[val_base + 11]; float bx11 = bias * x_shared[val_base + 11];
        float sx12 = scale * x_shared[val_base + 12]; float bx12 = bias * x_shared[val_base + 12];
        float sx13 = scale * x_shared[val_base + 13]; float bx13 = bias * x_shared[val_base + 13];
        float sx14 = scale * x_shared[val_base + 14]; float bx14 = bias * x_shared[val_base + 14];
        float sx15 = scale * x_shared[val_base + 15]; float bx15 = bias * x_shared[val_base + 15];

        // 16 FMA extractions — hardcoded shifts, no branches
        // Values 0-4: word 0 (clean)
        acc += fma(float((w0 >>  0) & 0x3F), sx00, bx00);
        acc += fma(float((w0 >>  6) & 0x3F), sx01, bx01);
        acc += fma(float((w0 >> 12) & 0x3F), sx02, bx02);
        acc += fma(float((w0 >> 18) & 0x3F), sx03, bx03);
        acc += fma(float((w0 >> 24) & 0x3F), sx04, bx04);
        // Value 5: straddle word 0→1
        acc += fma(float(((w0 >> 30) | (w1 << 2)) & 0x3F), sx05, bx05);
        // Values 6-9: word 1 (clean)
        acc += fma(float((w1 >>  4) & 0x3F), sx06, bx06);
        acc += fma(float((w1 >> 10) & 0x3F), sx07, bx07);
        acc += fma(float((w1 >> 16) & 0x3F), sx08, bx08);
        acc += fma(float((w1 >> 22) & 0x3F), sx09, bx09);
        // Value 10: straddle word 1→2
        acc += fma(float(((w1 >> 28) | (w2 << 4)) & 0x3F), sx10, bx10);
        // Values 11-15: word 2 (clean)
        acc += fma(float((w2 >>  2) & 0x3F), sx11, bx11);
        acc += fma(float((w2 >>  8) & 0x3F), sx12, bx12);
        acc += fma(float((w2 >> 14) & 0x3F), sx13, bx13);
        acc += fma(float((w2 >> 20) & 0x3F), sx14, bx14);
        acc += fma(float((w2 >> 26) & 0x3F), sx15, bx15);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// 6-bit variant for in_dim > 4096 (no shared memory cache, group-based iteration)
kernel void dequant_matvec_6bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size * 6 / 32;
    uint packed_cols = num_groups * packed_per_group;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint i = 0; i < group_size; i++) {
            uint bit_pos = i * 6;
            uint word = bit_pos >> 5;
            uint bit  = bit_pos & 31;

            uint val = (w_row[base_packed + word] >> bit);
            if (bit + 6 > 32) {
                val |= (w_row[base_packed + word + 1] << (32 - bit));
            }
            val &= 0x3F;
            acc += (float(val) * scale + bias) * x[base_x + i];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}


// ============================================================================
// Kernel 1i: 5-bit affine dequant matvec — FMA-optimized, chunk-based
// ============================================================================
// MLX affine 5-bit, group_size=64: 64 values × 5 bits = 320 bits = 10 uint32s per group.
// Processes 32 values (5 uint32 words = 160 bits) per iteration with fully unrolled
// FMA extraction. Pre-computes scale*x and bias*x for all 32 positions.
// Straddle values (6, 12, 19, 25) use hardcoded cross-word extraction — no branches.

kernel void dequant_matvec_5bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size * 5 / 32;  // 10 for gs=64
    uint packed_cols = num_groups * packed_per_group;

    // 32 values per chunk, 2 chunks per group (64/32=2)
    uint num_chunks = in_dim / 32;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    // Each lane processes every 32nd chunk of 32 values (5 uint32 words)
    for (uint chunk = simd_lane; chunk < num_chunks; chunk += 32) {
        uint val_base = chunk * 32;
        uint g = val_base / group_size;
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        // Word offset: chunk_within_group * 5 words per chunk
        uint chunk_in_group = (val_base - g * group_size) / 32;
        device const uint32_t* gw = w_row + g * packed_per_group + chunk_in_group * 5;
        uint32_t w0 = gw[0], w1 = gw[1], w2 = gw[2], w3 = gw[3], w4 = gw[4];

        // Pre-compute scale*x and bias*x for all 32 positions
        float sx00 = scale * x_shared[val_base +  0]; float bx00 = bias * x_shared[val_base +  0];
        float sx01 = scale * x_shared[val_base +  1]; float bx01 = bias * x_shared[val_base +  1];
        float sx02 = scale * x_shared[val_base +  2]; float bx02 = bias * x_shared[val_base +  2];
        float sx03 = scale * x_shared[val_base +  3]; float bx03 = bias * x_shared[val_base +  3];
        float sx04 = scale * x_shared[val_base +  4]; float bx04 = bias * x_shared[val_base +  4];
        float sx05 = scale * x_shared[val_base +  5]; float bx05 = bias * x_shared[val_base +  5];
        float sx06 = scale * x_shared[val_base +  6]; float bx06 = bias * x_shared[val_base +  6];
        float sx07 = scale * x_shared[val_base +  7]; float bx07 = bias * x_shared[val_base +  7];
        float sx08 = scale * x_shared[val_base +  8]; float bx08 = bias * x_shared[val_base +  8];
        float sx09 = scale * x_shared[val_base +  9]; float bx09 = bias * x_shared[val_base +  9];
        float sx10 = scale * x_shared[val_base + 10]; float bx10 = bias * x_shared[val_base + 10];
        float sx11 = scale * x_shared[val_base + 11]; float bx11 = bias * x_shared[val_base + 11];
        float sx12 = scale * x_shared[val_base + 12]; float bx12 = bias * x_shared[val_base + 12];
        float sx13 = scale * x_shared[val_base + 13]; float bx13 = bias * x_shared[val_base + 13];
        float sx14 = scale * x_shared[val_base + 14]; float bx14 = bias * x_shared[val_base + 14];
        float sx15 = scale * x_shared[val_base + 15]; float bx15 = bias * x_shared[val_base + 15];
        float sx16 = scale * x_shared[val_base + 16]; float bx16 = bias * x_shared[val_base + 16];
        float sx17 = scale * x_shared[val_base + 17]; float bx17 = bias * x_shared[val_base + 17];
        float sx18 = scale * x_shared[val_base + 18]; float bx18 = bias * x_shared[val_base + 18];
        float sx19 = scale * x_shared[val_base + 19]; float bx19 = bias * x_shared[val_base + 19];
        float sx20 = scale * x_shared[val_base + 20]; float bx20 = bias * x_shared[val_base + 20];
        float sx21 = scale * x_shared[val_base + 21]; float bx21 = bias * x_shared[val_base + 21];
        float sx22 = scale * x_shared[val_base + 22]; float bx22 = bias * x_shared[val_base + 22];
        float sx23 = scale * x_shared[val_base + 23]; float bx23 = bias * x_shared[val_base + 23];
        float sx24 = scale * x_shared[val_base + 24]; float bx24 = bias * x_shared[val_base + 24];
        float sx25 = scale * x_shared[val_base + 25]; float bx25 = bias * x_shared[val_base + 25];
        float sx26 = scale * x_shared[val_base + 26]; float bx26 = bias * x_shared[val_base + 26];
        float sx27 = scale * x_shared[val_base + 27]; float bx27 = bias * x_shared[val_base + 27];
        float sx28 = scale * x_shared[val_base + 28]; float bx28 = bias * x_shared[val_base + 28];
        float sx29 = scale * x_shared[val_base + 29]; float bx29 = bias * x_shared[val_base + 29];
        float sx30 = scale * x_shared[val_base + 30]; float bx30 = bias * x_shared[val_base + 30];
        float sx31 = scale * x_shared[val_base + 31]; float bx31 = bias * x_shared[val_base + 31];

        // 32 FMA extractions — hardcoded shifts, no branches
        // Values 0-5: word 0 (clean)
        acc += fma(float((w0 >>  0) & 0x1F), sx00, bx00);
        acc += fma(float((w0 >>  5) & 0x1F), sx01, bx01);
        acc += fma(float((w0 >> 10) & 0x1F), sx02, bx02);
        acc += fma(float((w0 >> 15) & 0x1F), sx03, bx03);
        acc += fma(float((w0 >> 20) & 0x1F), sx04, bx04);
        acc += fma(float((w0 >> 25) & 0x1F), sx05, bx05);
        // Value 6: straddle word 0→1
        acc += fma(float(((w0 >> 30) | (w1 << 2)) & 0x1F), sx06, bx06);
        // Values 7-11: word 1 (clean)
        acc += fma(float((w1 >>  3) & 0x1F), sx07, bx07);
        acc += fma(float((w1 >>  8) & 0x1F), sx08, bx08);
        acc += fma(float((w1 >> 13) & 0x1F), sx09, bx09);
        acc += fma(float((w1 >> 18) & 0x1F), sx10, bx10);
        acc += fma(float((w1 >> 23) & 0x1F), sx11, bx11);
        // Value 12: straddle word 1→2
        acc += fma(float(((w1 >> 28) | (w2 << 4)) & 0x1F), sx12, bx12);
        // Values 13-18: word 2 (clean)
        acc += fma(float((w2 >>  1) & 0x1F), sx13, bx13);
        acc += fma(float((w2 >>  6) & 0x1F), sx14, bx14);
        acc += fma(float((w2 >> 11) & 0x1F), sx15, bx15);
        acc += fma(float((w2 >> 16) & 0x1F), sx16, bx16);
        acc += fma(float((w2 >> 21) & 0x1F), sx17, bx17);
        acc += fma(float((w2 >> 26) & 0x1F), sx18, bx18);
        // Value 19: straddle word 2→3
        acc += fma(float(((w2 >> 31) | (w3 << 1)) & 0x1F), sx19, bx19);
        // Values 20-24: word 3 (clean)
        acc += fma(float((w3 >>  4) & 0x1F), sx20, bx20);
        acc += fma(float((w3 >>  9) & 0x1F), sx21, bx21);
        acc += fma(float((w3 >> 14) & 0x1F), sx22, bx22);
        acc += fma(float((w3 >> 19) & 0x1F), sx23, bx23);
        acc += fma(float((w3 >> 24) & 0x1F), sx24, bx24);
        // Value 25: straddle word 3→4
        acc += fma(float(((w3 >> 29) | (w4 << 3)) & 0x1F), sx25, bx25);
        // Values 26-31: word 4 (clean)
        acc += fma(float((w4 >>  2) & 0x1F), sx26, bx26);
        acc += fma(float((w4 >>  7) & 0x1F), sx27, bx27);
        acc += fma(float((w4 >> 12) & 0x1F), sx28, bx28);
        acc += fma(float((w4 >> 17) & 0x1F), sx29, bx29);
        acc += fma(float((w4 >> 22) & 0x1F), sx30, bx30);
        acc += fma(float((w4 >> 27) & 0x1F), sx31, bx31);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// 5-bit variant for in_dim > 4096 (no shared memory cache)
kernel void dequant_matvec_5bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size * 5 / 32;
    uint packed_cols = num_groups * packed_per_group;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint i = 0; i < group_size; i++) {
            uint bit_pos = i * 5;
            uint word = bit_pos >> 5;
            uint bit  = bit_pos & 31;

            uint val = (w_row[base_packed + word] >> bit);
            if (bit + 5 > 32) {
                val |= (w_row[base_packed + word + 1] << (32 - bit));
            }
            val &= 0x1F;
            acc += (float(val) * scale + bias) * x[base_x + i];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}


// ============================================================================
// Kernel 1d: FULLY OPTIMIZED with uint4 vector loads
// ============================================================================
//
// Same structure as v3 but uses uint4 loads (128-bit / 16 bytes) to maximize
// memory bandwidth per thread. Each uint4 = 4 uint32 = 32 nibbles.
//
// For gate/up (packed_cols=512): each thread processes 512/32 = 16 uint32
//   = 4 uint4 loads per thread
// For down (packed_cols=128): each thread processes 128/32 = 4 uint32
//   = 1 uint4 load per thread

kernel void dequant_matvec_4bit_v4(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache input vector — ALL threads must participate before the barrier
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    // Pointers — cast to uint4 for vector loads
    device const uint4* w_row_v = (device const uint4*)(W_packed + row * packed_cols);
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    uint vec4_cols = packed_cols / 4;  // number of uint4 vectors per row

    float acc = 0.0f;

    // Each lane processes vec4_cols / 32 vectors (coalesced: adjacent lanes read adjacent uint4)
    for (uint vi = simd_lane; vi < vec4_cols; vi += 32) {
        uint4 packed4 = w_row_v[vi];

        // Each uint4 covers 4 * 8 = 32 input elements
        // Starting packed column index = vi * 4
        uint base_col = vi * 4;
        uint x_base = base_col * 8;  // starting input element

        // Process each of the 4 uint32 words in the uint4
        // Unroll all 4 words x 8 nibbles = 32 multiply-adds
        #pragma unroll
        for (uint w = 0; w < 4; w++) {
            uint32_t packed = packed4[w];
            uint col = base_col + w;
            uint g = col / (group_size / 8);
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);

            uint xb = x_base + w * 8;
            acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[xb + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[xb + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[xb + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[xb + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[xb + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[xb + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[xb + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[xb + 7];
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1e: Multi-expert batched matvec
// ============================================================================
//
// Dispatch multiple experts simultaneously. The grid's Y dimension indexes
// the expert, so K experts' matmuls run as parallel threadgroups.
//
// Buffer layout: W_packed, scales, biases are arrays of K experts concatenated.
// x_inputs:  K input vectors concatenated [K * in_dim]
// out:       K output vectors concatenated [K * out_dim]
// expert_offsets: byte offset into W_packed buffer for each expert's weights
//                 (allows non-contiguous expert data in a shared buffer)

kernel void dequant_matvec_4bit_batched(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x_inputs   [[buffer(3)]],  // [K, in_dim]
    device float*          out        [[buffer(4)]],  // [K, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    // Per-expert offsets into the weight/scale/bias buffers (in elements)
    device const uint*     w_offsets  [[buffer(8)]],  // [K] offset in uint32 elements
    device const uint*     s_offsets  [[buffer(9)]],  // [K] offset in uint16 elements
    device const uint*     b_offsets  [[buffer(10)]], // [K] offset in uint16 elements
    constant uint&         num_row_tiles [[buffer(11)]], // ceil(out_dim / ROWS_PER_TG)
    uint tgid_flat [[threadgroup_position_in_grid]],  // linearized (row_tile + expert * num_row_tiles)
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // De-linearize: tgid_flat = row_tile + expert_k * num_row_tiles
    uint expert_k = tgid_flat / num_row_tiles;
    uint row_tile = tgid_flat % num_row_tiles;
    uint row = row_tile * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache this expert's input vector
    threadgroup float x_shared[4096];
    device const float* x_k = x_inputs + expert_k * in_dim;
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x_k[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Point to this expert's weights
    device const uint32_t* w_row = W_packed + w_offsets[expert_k] + row * packed_cols;
    device const uint16_t* s_row = scales   + s_offsets[expert_k] + row * num_groups;
    device const uint16_t* b_row = biases   + b_offsets[expert_k] + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[x_base + 0];
        acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[x_base + 1];
        acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[x_base + 2];
        acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[x_base + 3];
        acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[x_base + 4];
        acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[x_base + 5];
        acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[x_base + 6];
        acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert_k * out_dim + row] = sum;
    }
}


// ============================================================================
// Kernel 1j: BF16 (unquantized) matrix-vector multiply
// ============================================================================
// For tensors stored in raw BF16 (e.g. o_proj, out_proj in mixed-precision models).
// No dequantization needed — just bf16→f32 conversion and dot product.
// W: [out_dim, in_dim] stored as uint16 (bf16 bit pattern)

kernel void bf16_matvec(
    device const uint16_t* W        [[buffer(0)]],
    device const float*    x        [[buffer(1)]],
    device float*          out      [[buffer(2)]],
    constant uint&         out_dim  [[buffer(3)]],
    constant uint&         in_dim   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    threadgroup float x_shared[8192];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint16_t* w_row = W + row * in_dim;

    float acc = 0.0f;
    for (uint col = simd_lane; col < in_dim; col += 32) {
        acc += bf16_to_f32(w_row[col]) * x_shared[col];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// BF16 matvec — multi-row SIMD variant (4 rows/SIMD, 32 rows/TG)
// ============================================================================
// Same optimization as dequant_matvec_8bit_opt: each SIMD group computes 4
// output rows in parallel, reading x from threadgroup cache once. This cuts
// the number of threadgroups by 4× and amortizes the x-cache load, which is
// the only place where BF16 matvec output rows share memory traffic.

kernel void bf16_matvec_opt(
    device const uint16_t* W        [[buffer(0)]],
    device const float*    x        [[buffer(1)]],
    device float*          out      [[buffer(2)]],
    constant uint&         out_dim  [[buffer(3)]],
    constant uint&         in_dim   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Cache x in shared memory. 16 KB keeps 2 TGs resident per core.
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) x_shared[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4 rows/SIMD × 8 SIMDs/TG = 32 rows/TG
    uint row_base = tgid * 32 + simd_group * 4;
    if (row_base >= out_dim) return;

    uint r0 = row_base + 0;
    uint r1 = min(row_base + 1, out_dim - 1);
    uint r2 = min(row_base + 2, out_dim - 1);
    uint r3 = min(row_base + 3, out_dim - 1);

    // Use uint4 loads (16 bytes = 8 bf16 values per load) for better memory
    // throughput. 2-byte loads are inefficient on Apple Silicon; 16-byte
    // loads are the native transaction size.
    device const uint4* w0v = (device const uint4*)(W + r0 * in_dim);
    device const uint4* w1v = (device const uint4*)(W + r1 * in_dim);
    device const uint4* w2v = (device const uint4*)(W + r2 * in_dim);
    device const uint4* w3v = (device const uint4*)(W + r3 * in_dim);

    float4 acc = float4(0.0);

    // Each uint4 holds 8 bf16 values (4 uint32 × 2 bf16 each).
    // in_dim / 8 = number of uint4 chunks per row.
    uint in_vec = in_dim / 8;

    for (uint v = simd_lane; v < in_vec; v += 32) {
        // Load 8 x values corresponding to this uint4 of weights.
        threadgroup const float4* xp = (threadgroup const float4*)(x_shared + v * 8);
        float4 xv0 = xp[0];
        float4 xv1 = xp[1];

        uint4 p0 = w0v[v];
        uint4 p1 = w1v[v];
        uint4 p2 = w2v[v];
        uint4 p3 = w3v[v];

        // Each uint32 has 2 bf16 values: low 16 bits and high 16 bits.
        // Row 0
        acc.x = fma(bf16_to_f32(uint16_t(p0.x & 0xFFFF)), xv0.x, acc.x);
        acc.x = fma(bf16_to_f32(uint16_t(p0.x >> 16)),    xv0.y, acc.x);
        acc.x = fma(bf16_to_f32(uint16_t(p0.y & 0xFFFF)), xv0.z, acc.x);
        acc.x = fma(bf16_to_f32(uint16_t(p0.y >> 16)),    xv0.w, acc.x);
        acc.x = fma(bf16_to_f32(uint16_t(p0.z & 0xFFFF)), xv1.x, acc.x);
        acc.x = fma(bf16_to_f32(uint16_t(p0.z >> 16)),    xv1.y, acc.x);
        acc.x = fma(bf16_to_f32(uint16_t(p0.w & 0xFFFF)), xv1.z, acc.x);
        acc.x = fma(bf16_to_f32(uint16_t(p0.w >> 16)),    xv1.w, acc.x);

        // Row 1
        acc.y = fma(bf16_to_f32(uint16_t(p1.x & 0xFFFF)), xv0.x, acc.y);
        acc.y = fma(bf16_to_f32(uint16_t(p1.x >> 16)),    xv0.y, acc.y);
        acc.y = fma(bf16_to_f32(uint16_t(p1.y & 0xFFFF)), xv0.z, acc.y);
        acc.y = fma(bf16_to_f32(uint16_t(p1.y >> 16)),    xv0.w, acc.y);
        acc.y = fma(bf16_to_f32(uint16_t(p1.z & 0xFFFF)), xv1.x, acc.y);
        acc.y = fma(bf16_to_f32(uint16_t(p1.z >> 16)),    xv1.y, acc.y);
        acc.y = fma(bf16_to_f32(uint16_t(p1.w & 0xFFFF)), xv1.z, acc.y);
        acc.y = fma(bf16_to_f32(uint16_t(p1.w >> 16)),    xv1.w, acc.y);

        // Row 2
        acc.z = fma(bf16_to_f32(uint16_t(p2.x & 0xFFFF)), xv0.x, acc.z);
        acc.z = fma(bf16_to_f32(uint16_t(p2.x >> 16)),    xv0.y, acc.z);
        acc.z = fma(bf16_to_f32(uint16_t(p2.y & 0xFFFF)), xv0.z, acc.z);
        acc.z = fma(bf16_to_f32(uint16_t(p2.y >> 16)),    xv0.w, acc.z);
        acc.z = fma(bf16_to_f32(uint16_t(p2.z & 0xFFFF)), xv1.x, acc.z);
        acc.z = fma(bf16_to_f32(uint16_t(p2.z >> 16)),    xv1.y, acc.z);
        acc.z = fma(bf16_to_f32(uint16_t(p2.w & 0xFFFF)), xv1.z, acc.z);
        acc.z = fma(bf16_to_f32(uint16_t(p2.w >> 16)),    xv1.w, acc.z);

        // Row 3
        acc.w = fma(bf16_to_f32(uint16_t(p3.x & 0xFFFF)), xv0.x, acc.w);
        acc.w = fma(bf16_to_f32(uint16_t(p3.x >> 16)),    xv0.y, acc.w);
        acc.w = fma(bf16_to_f32(uint16_t(p3.y & 0xFFFF)), xv0.z, acc.w);
        acc.w = fma(bf16_to_f32(uint16_t(p3.y >> 16)),    xv0.w, acc.w);
        acc.w = fma(bf16_to_f32(uint16_t(p3.z & 0xFFFF)), xv1.x, acc.w);
        acc.w = fma(bf16_to_f32(uint16_t(p3.z >> 16)),    xv1.y, acc.w);
        acc.w = fma(bf16_to_f32(uint16_t(p3.w & 0xFFFF)), xv1.z, acc.w);
        acc.w = fma(bf16_to_f32(uint16_t(p3.w >> 16)),    xv1.w, acc.w);
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);

    if (simd_lane == 0) {
        if (row_base + 0 < out_dim) out[row_base + 0] = acc.x;
        if (row_base + 1 < out_dim) out[row_base + 1] = acc.y;
        if (row_base + 2 < out_dim) out[row_base + 2] = acc.z;
        if (row_base + 3 < out_dim) out[row_base + 3] = acc.w;
    }
}


// Fused BF16 matvec for two small-out matrices sharing the same input.
// Used for the linear-attention b/a projections in Qwen3.5-9B where each is
// only [32, 4096] bf16 — too small to dispatch independently. Computes both
// outputs in a single kernel launch, saving ~one dispatch worth of overhead.
//
// Dispatch: (out_dim) TGs, 64 threads each. Each TG computes ONE output row
// for BOTH matrices. Lanes do a strided dot product and reduce via simd_sum.
kernel void bf16_matvec_pair(
    device const uint16_t* W_a      [[buffer(0)]],
    device const uint16_t* W_b      [[buffer(1)]],
    device const float*    x        [[buffer(2)]],
    device float*          out_a    [[buffer(3)]],
    device float*          out_b    [[buffer(4)]],
    constant uint&         out_dim  [[buffer(5)]],
    constant uint&         in_dim   [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    device const uint16_t* wa = W_a + tgid * in_dim;
    device const uint16_t* wb = W_b + tgid * in_dim;

    float acc_a = 0.0f, acc_b = 0.0f;
    for (uint col = lid; col < in_dim; col += tg_size) {
        float xv = x[col];
        acc_a += bf16_to_f32(wa[col]) * xv;
        acc_b += bf16_to_f32(wb[col]) * xv;
    }

    acc_a = simd_sum(acc_a);
    acc_b = simd_sum(acc_b);

    threadgroup float sh_a[2], sh_b[2];
    uint simd_group = lid / 32;
    if (lid % 32 == 0) { sh_a[simd_group] = acc_a; sh_b[simd_group] = acc_b; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        float a_val = sh_a[0] + sh_a[1];
        float b_val = sh_b[0] + sh_b[1];
        out_a[tgid] = a_val;
        out_b[tgid] = b_val;
    }
}


// BF16 matvec for in_dim > 8192 (no shared memory)
kernel void bf16_matvec_fast(
    device const uint16_t* W        [[buffer(0)]],
    device const float*    x        [[buffer(1)]],
    device float*          out      [[buffer(2)]],
    constant uint&         out_dim  [[buffer(3)]],
    constant uint&         in_dim   [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    device const uint16_t* w_row = W + tgid * in_dim;

    float acc = 0.0f;
    for (uint col = lid; col < in_dim; col += tg_size) {
        acc += bf16_to_f32(w_row[col]) * x[col];
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}


// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================

kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// Vectorized SwiGLU: process 4 elements per thread
kernel void swiglu_fused_vec4(
    device const float4* gate [[buffer(0)]],
    device const float4* up   [[buffer(1)]],
    device float4*       out  [[buffer(2)]],
    constant uint&       dim  [[buffer(3)]],  // original dim (must be multiple of 4)
    uint tid [[thread_position_in_grid]]
) {
    uint vec_dim = dim / 4;
    if (tid >= vec_dim) return;

    float4 g = gate[tid];
    float4 silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 2b: Batched SwiGLU for K experts
// ============================================================================

kernel void swiglu_fused_batched(
    device const float* gate [[buffer(0)]],  // [K * dim]
    device const float* up   [[buffer(1)]],  // [K * dim]
    device float*       out  [[buffer(2)]],  // [K * dim]
    constant uint&      dim  [[buffer(3)]],
    constant uint&      K    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = K * dim;
    if (tid >= total) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 3: Weighted sum of expert outputs
// ============================================================================

kernel void weighted_sum(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      K           [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * dim + tid];
    }
    out[tid] = acc;
}


// ============================================================================
// Kernel 4: RMS Normalization
// ============================================================================

kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];

    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            sum_sq[0] = val;
        }
    }
}

kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}


// ============================================================================
// Kernel 4b: RMS Normalization with bf16 weights
// ============================================================================
// Same as rms_norm_apply but reads weights as bfloat16 (uint16_t) and
// converts to float32 inline. Used in the fused o_proj+norm+routing path
// where norm weights come directly from the mmap'd weight file (bf16).

kernel void rms_norm_apply_bf16(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],  // bf16 weights
    device const float*    sum_sq  [[buffer(2)]],
    device float*          out     [[buffer(3)]],
    constant uint&         dim     [[buffer(4)]],
    constant float&        eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    float w = bf16_to_f32(weight[tid]);
    out[tid] = x[tid] * rms * w;
}


// ============================================================================
// Kernel 5: Residual add
// ============================================================================
// out[i] = a[i] + b[i]
// Used to fuse the residual connection into a GPU command buffer,
// eliminating a CPU round-trip between o_proj and routing.

// ============================================================================
// Full-attention pre-processing on GPU: split q_proj into (q, gate), apply
// per-head RMS norm, apply partial RoPE (MLX-style split-half pairing), and
// write results into the GPU attention buffers. This eliminates the CPU
// round-trip that was the biggest remaining per-layer CPU cost (~0.5 ms per
// full-attn layer × 8 layers ≈ 4 ms/token).
//
// Implementation is two kernels — one for Q (which also carries the gate)
// and one for K — so each kernel can cleanly do a threadgroup-level
// reduction without branch-dependent barriers.
//
// Layout (Qwen3.5-9B):
//   q_proj_out:      [num_q_heads, 2, head_dim] — for each head, q then gate
//   k_out:           [num_kv_heads, head_dim]
//   Partial RoPE:    rotary_dim = head_dim * 0.25; MLX pairs (i, i+rotary/2)

kernel void q_norm_rope(
    device const float*    q_proj_out [[buffer(0)]],  // [num_q_heads*2*head_dim]
    device const uint16_t* q_norm_w   [[buffer(1)]],  // [head_dim] bf16
    device float*          out_q      [[buffer(2)]],  // [num_q_heads*head_dim]
    device float*          out_q_gate [[buffer(3)]],  // [num_q_heads*head_dim]
    constant uint&         head_dim   [[buffer(4)]],
    constant uint&         rotary_dim [[buffer(5)]],
    constant int&          pos        [[buffer(6)]],
    constant float&        rope_theta [[buffer(7)]],
    constant float&        rms_eps    [[buffer(8)]],
    uint head_id     [[threadgroup_position_in_grid]],
    uint tid         [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_sum[16];  // max 512 threads / 32 = 16 simds

    device const float* q_src = q_proj_out + head_id * 2 * head_dim;
    float qv = q_src[tid];
    float gv = q_src[head_dim + tid];

    // Sum of squares for RMS norm.
    float sqv = qv * qv;
    float simd_v = simd_sum(sqv);
    uint simd_lane  = tid % 32;
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_sum[simd_group] = simd_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    if (simd_group == 0) {
        uint num_simds = (head_dim + 31) / 32;
        tot = (simd_lane < num_simds) ? shared_sum[simd_lane] : 0.0f;
        tot = simd_sum(tot);
        if (simd_lane == 0) shared_sum[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot = shared_sum[0];

    float inv_rms = rsqrt(tot / float(head_dim) + rms_eps);
    float qw = bf16_to_f32(q_norm_w[tid]);
    float q_normed = qv * inv_rms * qw;

    // Partial RoPE. For tid < rotary_dim, rotate with paired element.
    uint half_rot = rotary_dim / 2;
    float q_out_val = q_normed;
    if (tid < half_rot) {
        uint j = tid + half_rot;
        float qh = q_src[j];
        float qh_w = bf16_to_f32(q_norm_w[j]);
        float qh_n = qh * inv_rms * qh_w;
        float freq = 1.0f / pow(rope_theta, float(2 * tid) / float(rotary_dim));
        float angle = float(pos) * freq;
        float c = cos(angle);
        float s = sin(angle);
        q_out_val = q_normed * c - qh_n * s;
    } else if (tid < rotary_dim) {
        uint i = tid - half_rot;
        float qlo = q_src[i];
        float qlo_w = bf16_to_f32(q_norm_w[i]);
        float qlo_n = qlo * inv_rms * qlo_w;
        float freq = 1.0f / pow(rope_theta, float(2 * i) / float(rotary_dim));
        float angle = float(pos) * freq;
        float c = cos(angle);
        float s = sin(angle);
        q_out_val = qlo_n * s + q_normed * c;
    }

    out_q[head_id * head_dim + tid] = q_out_val;
    out_q_gate[head_id * head_dim + tid] = gv;
}

kernel void k_norm_rope(
    device const float*    k_out      [[buffer(0)]],  // [num_kv_heads*head_dim]
    device const uint16_t* k_norm_w   [[buffer(1)]],  // [head_dim] bf16
    device float*          out_k      [[buffer(2)]],  // [num_kv_heads*head_dim]
    constant uint&         head_dim   [[buffer(3)]],
    constant uint&         rotary_dim [[buffer(4)]],
    constant int&          pos        [[buffer(5)]],
    constant float&        rope_theta [[buffer(6)]],
    constant float&        rms_eps    [[buffer(7)]],
    uint head_id     [[threadgroup_position_in_grid]],
    uint tid         [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_sum[16];

    device const float* k_src = k_out + head_id * head_dim;
    float kv = k_src[tid];

    float skv = kv * kv;
    float simd_v = simd_sum(skv);
    uint simd_lane  = tid % 32;
    uint simd_group = tid / 32;
    if (simd_lane == 0) shared_sum[simd_group] = simd_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    if (simd_group == 0) {
        uint num_simds = (head_dim + 31) / 32;
        tot = (simd_lane < num_simds) ? shared_sum[simd_lane] : 0.0f;
        tot = simd_sum(tot);
        if (simd_lane == 0) shared_sum[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot = shared_sum[0];

    float inv_rms = rsqrt(tot / float(head_dim) + rms_eps);
    float kw = bf16_to_f32(k_norm_w[tid]);
    float k_normed = kv * inv_rms * kw;

    uint half_rot = rotary_dim / 2;
    float k_out_val = k_normed;
    if (tid < half_rot) {
        uint j = tid + half_rot;
        float kh = k_src[j];
        float kh_w = bf16_to_f32(k_norm_w[j]);
        float kh_n = kh * inv_rms * kh_w;
        float freq = 1.0f / pow(rope_theta, float(2 * tid) / float(rotary_dim));
        float angle = float(pos) * freq;
        float c = cos(angle);
        float s = sin(angle);
        k_out_val = k_normed * c - kh_n * s;
    } else if (tid < rotary_dim) {
        uint i = tid - half_rot;
        float klo = k_src[i];
        float klo_w = bf16_to_f32(k_norm_w[i]);
        float klo_n = klo * inv_rms * klo_w;
        float freq = 1.0f / pow(rope_theta, float(2 * i) / float(rotary_dim));
        float angle = float(pos) * freq;
        float c = cos(angle);
        float s = sin(angle);
        k_out_val = klo_n * s + k_normed * c;
    }

    out_k[head_id * head_dim + tid] = k_out_val;
}


kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}


// ============================================================================
// Kernel 5b: Fused residual_add + rms_norm_apply (bf16 weight)
// ============================================================================
// Single-dispatch fusion of three previously-separate encoders:
//   1. residual_add:        h_mid   = a + b
//   2. rms_norm_sum_sq:     sum_sq  = sum(h_mid^2)
//   3. rms_norm_apply_bf16: out     = h_mid / sqrt(mean_sq + eps) * w
//
// Saves two dispatch launches per site (~50-70 µs per layer). Also writes
// h_mid out so downstream kernels can still read the unnormalized residual.
//
// Dispatch: 1 threadgroup with 256 threads. dim must be ≤ 4096 (fits the
// thread loop; hidden_dim is always 4096 for Qwen3.5 variants).

kernel void residual_rms_norm_fused(
    device const float*    a         [[buffer(0)]],  // residual input (stays; dim)
    device const float*    b         [[buffer(1)]],  // other input    (stays; dim)
    device float*          h_mid     [[buffer(2)]],  // output: h_mid = a + b   (dim)
    device const uint16_t* weight    [[buffer(3)]],  // bf16 rms weight         (dim)
    device float*          out       [[buffer(4)]],  // output: normed          (dim)
    constant uint&         dim       [[buffer(5)]],
    constant float&        eps       [[buffer(6)]],
    uint lid         [[thread_position_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared[32];

    // Phase 1: compute h_mid = a + b, accumulate sum of squares.
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += 256) {
        float v = a[i] + b[i];
        h_mid[i] = v;
        acc = fma(v, v, acc);
    }

    // Reduce sum of squares across the threadgroup.
    float simd_val = simd_sum(acc);
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot = 0.0f;
    if (simd_group == 0) {
        tot = (simd_lane < 8) ? shared[simd_lane] : 0.0f;
        tot = simd_sum(tot);
        if (simd_lane == 0) shared[0] = tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot = shared[0];

    float rms = rsqrt(tot / float(dim) + eps);

    // Phase 2: write out[i] = h_mid[i] * rms * weight[i].
    for (uint i = lid; i < dim; i += 256) {
        float w = bf16_to_f32(weight[i]);
        out[i] = h_mid[i] * rms * w;
    }
}


// ============================================================================
// Kernel 6: Batched GPU attention scores (Q @ K^T, scaled) — all heads at once
// ============================================================================
//
// Computes scores[h, p] = sum_d(Q[h, d] * K[p, kv_h*head_dim + d]) * scale
// for all heads h in [0, num_heads) and positions p in [0, seq_len).
//
// Grid: linearized (pos + h * num_seq_tgs) — one threadgroup per (position, head).
// Each threadgroup of 256 threads reduces over head_dim=256.
//
// GQA mapping: kv_head = h / heads_per_kv (e.g. 16 query heads share 1 KV head)
//
// Output layout: scores[h * seq_stride + p] where seq_stride = MAX_SEQ_LEN

kernel void attn_scores_batched(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       scores     [[buffer(2)]],  // [num_heads, seq_stride]
    constant uint&      head_dim   [[buffer(3)]],  // 256
    constant uint&      kv_dim     [[buffer(4)]],  // 512
    constant uint&      seq_len    [[buffer(5)]],  // current seq length
    constant uint&      seq_stride [[buffer(6)]],  // MAX_SEQ_LEN
    constant float&     scale      [[buffer(7)]],  // 1/sqrt(head_dim)
    constant uint&      heads_per_kv [[buffer(8)]], // 16 (GQA ratio)
    constant uint&      num_seq_tgs  [[buffer(9)]],  // = seq_len
    uint tgid  [[threadgroup_position_in_grid]],    // linearized: pos + h * num_seq_tgs
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    device const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

    float acc = 0.0f;
    for (uint d = lid; d < head_dim; d += tg_size) {
        acc += qh[d] * kp[d];
    }

    // SIMD reduction
    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}


// ============================================================================
// Kernel 7: Batched softmax — one threadgroup per head
// ============================================================================

kernel void attn_softmax_batched(
    device float*    scores     [[buffer(0)]],  // [num_heads, seq_stride]
    constant uint&   seq_len    [[buffer(1)]],
    constant uint&   seq_stride [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],     // head index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    device float* s = scores + tgid * seq_stride;

    // Pass 1: find max
    threadgroup float shared_max[32];
    float local_max = -1e30f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        local_max = max(local_max, s[i]);
    }
    float sm = simd_max(local_max);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared_max[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_max = simd_max(shared_max[simd_lane]);
    }
    threadgroup float broadcast_max;
    if (lid == 0) broadcast_max = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = broadcast_max;

    // Pass 2: exp and sum
    threadgroup float shared_sum[32];
    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        float val = exp(s[i] - global_max);
        s[i] = val;
        local_sum += val;
    }
    float simd_s = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_group] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_sum = simd_sum(shared_sum[simd_lane]);
    }
    threadgroup float broadcast_sum;
    if (lid == 0) broadcast_sum = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = broadcast_sum;

    // Pass 3: normalize
    float inv_sum = 1.0f / global_sum;
    for (uint i = lid; i < seq_len; i += tg_size) {
        s[i] *= inv_sum;
    }
}


// ============================================================================
// Kernel 8: Batched attention value aggregation (scores @ V) — all heads
// ============================================================================
//
// For each head h: output[h*head_dim + d] = sum_p(scores[h*seq_stride+p] * V[p*kv_dim + kv_h*head_dim + d])
//
// Grid: linearized over (head_dim * num_heads) — one thread per (dimension, head).

kernel void attn_values_batched(
    device const float* scores   [[buffer(0)]],  // [num_heads, seq_stride]
    device const float* V_cache  [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       out      [[buffer(2)]],  // [num_heads, head_dim]
    constant uint&      head_dim  [[buffer(3)]],  // 256
    constant uint&      kv_dim    [[buffer(4)]],  // 512
    constant uint&      seq_len   [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant uint&      heads_per_kv [[buffer(7)]],
    uint tid [[thread_position_in_grid]]          // linearized: d + h * head_dim
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint p = 0; p < seq_len; p++) {
        acc += s[p] * V_cache[p * kv_dim + kv_h * head_dim + d];
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Kernel 9: Sigmoid element-wise gate
// ============================================================================
// out[i] = x[i] * sigmoid(gate[i])

kernel void sigmoid_gate(
    device float*       x_out  [[buffer(0)]],  // [dim] in/out
    device const float* gate   [[buffer(1)]],  // [dim] gate values
    constant uint&      dim    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = 1.0f / (1.0f + exp(-gate[tid]));
    x_out[tid] = x_out[tid] * g;
}


// ============================================================================
// Kernel 10: GatedDeltaNet linear attention step (single token, all heads)
// ============================================================================
//
// Implements the GatedDeltaNet recurrence for autoregressive generation:
//   1. State decay:  S[vi][ki] *= g_decay
//   2. Memory read:  kv_mem[vi] = sum_ki(S[vi][ki] * k[ki])
//   3. Delta:        delta[vi] = (v[vi] - kv_mem[vi]) * beta_gate
//   4. State update: S[vi][ki] += k[ki] * delta[vi]
//   5. Output:       out[vi] = sum_ki(S[vi][ki] * q[ki])
//
// Dispatch: 64 threadgroups (one per v-head), 128 threads each (one per vi).
// Each thread owns one row S[head_id][vi][:] of the 128x128 state matrix.
//
// State layout: [64 * 128 * 128] float = 4MB total, persisted across tokens.
// k-head sharing: 4 v-heads share 1 k-head (64 v-heads / 16 k-heads).

kernel void gated_delta_net_step(
    device float *state,             // [64 * 128 * 128] persistent state
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &k_heads_per_v,    // = 4
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Step 3+4: Delta update — S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < 128; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output — out[vi] = dot(S[vi][:], q[:])
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        out_val += state[state_base + ki] * q[k_base + ki];
    }
    output[v_base + vi] = out_val;
}


// ============================================================================
// Kernel 11: Conv1d depthwise step (single token, incremental inference)
// ============================================================================
//
// Depthwise 1D convolution for one new input token:
//   output[c] = sum_k(history[k][c] * weight[c][k]) + input[c] * weight[c][3]
//   then SiLU activation: output[c] = output[c] / (1 + exp(-output[c]))
//
// After computing, shifts the history buffer left and appends the new input.
//
// Weight layout: [channels * kernel_size] bf16, weight[c * kernel_size + k]
// Conv state layout: [(kernel_size-1) * channels] row-major, state[k * channels + c]
// kernel_size = 4 (hardcoded), so 3 history slots + 1 new input.
//
// Dispatch: conv_dim threads (12288), one per channel.

kernel void conv1d_step(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    device const float *input,        // [conv_dim] current input
    device const uint16_t *weights,   // [conv_dim * 4] bf16 as uint16
    device float *output,             // [conv_dim] convolution output
    constant uint &conv_dim,          // = 12288
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    // Convolution: dot product of history + new input with weights
    // weight layout: weight[c * 4 + k] for channel c, position k
    uint w_base = idx * 4;
    float acc = 0.0f;

    // 3 history slots (k=0,1,2)
    acc += conv_state[0 * conv_dim + idx] * bf16_to_f32(weights[w_base + 0]);
    acc += conv_state[1 * conv_dim + idx] * bf16_to_f32(weights[w_base + 1]);
    acc += conv_state[2 * conv_dim + idx] * bf16_to_f32(weights[w_base + 2]);

    // New input (k=3)
    float inp = input[idx];
    acc += inp * bf16_to_f32(weights[w_base + 3]);

    // SiLU activation
    output[idx] = acc / (1.0f + exp(-acc));

    // Shift history: move slots 1,2 -> 0,1, append input at slot 2
    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = inp;
}


// ============================================================================
// Kernel 12: Per-head RMS normalize for q and k vectors
// ============================================================================
// q: [num_k_heads * key_dim], k: [num_k_heads * key_dim]
// Normalize each head independently, then scale by 1/sqrt(key_dim)^2 for q, 1/sqrt(key_dim) for k
// Dispatch: num_k_heads threadgroups, key_dim threads each

kernel void rms_norm_qk(
    device float *q,              // [num_k_heads * key_dim] in/out
    device float *k,              // [num_k_heads * key_dim] in/out
    constant uint &key_dim,       // = 128
    constant float &inv_scale,    // = 1/sqrt(key_dim)
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * key_dim;

    // RMS norm for q
    threadgroup float q_sum_sq;
    if (tid == 0) q_sum_sq = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qval = (tid < key_dim) ? q[base + tid] : 0;
    // Use threadgroup atomic add for sum of squares
    float q_sq_local = qval * qval;
    // Simple reduction: thread 0 accumulates (key_dim=128, fits in one pass)
    threadgroup float q_partial[128];
    q_partial[tid] = q_sq_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += q_partial[i];
        q_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_inv_rms = rsqrt(q_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;  // q gets extra scale
    }

    // RMS norm for k
    threadgroup float k_sum_sq;
    float kval = (tid < key_dim) ? k[base + tid] : 0;
    threadgroup float k_partial[128];
    k_partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += k_partial[i];
        k_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_inv_rms = rsqrt(k_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}


// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
// Per v-head: g_decay = exp(-A * softplus(alpha + dt_bias)), beta_gate = sigmoid(beta)
// Dispatch: num_v_heads threads (64)

kernel void compute_decay_beta(
    device const float *alpha_out,   // [num_v_heads] from projection
    device const float *beta_out,    // [num_v_heads] from projection
    device const float *A_log,       // [num_v_heads] log of decay base (persistent)
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay,           // [num_v_heads] output
    device float *beta_gate,         // [num_v_heads] output
    uint idx [[thread_position_in_grid]]
) {
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = exp(A_log[idx]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}


// ============================================================================
// Kernel 14: Gated RMS norm (z-gated output normalization)
// ============================================================================
// output[i] = rms_norm(values[i]) * SiLU(z[i]) * weight[i]
// Per v-head: normalize values, gate with z, scale with weight
// Dispatch: num_v_heads threadgroups, value_dim threads each

kernel void gated_rms_norm(
    device const float *values,       // [num_v_heads * value_dim] delta-net output
    device const float *z,            // [num_v_heads * value_dim] gate values
    device const uint16_t *weight,    // [value_dim] bf16 norm weights (shared across heads)
    device float *output,             // [num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant float &eps,              // = 1e-6
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * value_dim;

    float val = (tid < value_dim) ? values[base + tid] : 0;

    // RMS norm reduction
    threadgroup float partial[128];
    partial[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < value_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(value_dim) + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));  // SiLU
        float w = bf16_to_f32(weight[tid]);
        output[base + tid] = normed * gate * w;
    }
}


// ============================================================================
// Kernel 12: MoE combine + residual + shared expert gate (fused)
// ============================================================================
// Fused operation for CMD3 GPU-side combine:
//   hidden[i] = h_mid[i] + sum_k(expert_weight[k] * expert_out[k][i])
//               + sigmoid(shared_gate_score) * shared_out[i]
//
// All 8 expert output buffers are always bound (unused ones have weight=0).
// This avoids variable buffer bindings and keeps the dispatch simple.
//
// Dispatch: (dim + 255) / 256 threadgroups, 256 threads each.

kernel void moe_combine_residual(
    device const float* h_mid       [[buffer(0)]],   // [dim]
    device const float* shared_out  [[buffer(1)]],   // [dim]
    device float*       hidden_out  [[buffer(2)]],   // [dim] output
    device const float* expert_out0 [[buffer(3)]],   // [dim] expert 0
    device const float* expert_out1 [[buffer(4)]],   // [dim] expert 1
    device const float* expert_out2 [[buffer(5)]],   // [dim] expert 2
    device const float* expert_out3 [[buffer(6)]],   // [dim] expert 3
    device const float* expert_out4 [[buffer(7)]],   // [dim] expert 4
    device const float* expert_out5 [[buffer(8)]],   // [dim] expert 5
    device const float* expert_out6 [[buffer(9)]],   // [dim] expert 6
    device const float* expert_out7 [[buffer(10)]],  // [dim] expert 7
    device const float* params      [[buffer(11)]],  // [10]: weights[0..7], shared_gate_score, (unused)
    constant uint&      dim         [[buffer(12)]],
    constant uint&      K           [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    // Read expert weights and shared gate from params buffer
    float shared_gate = 1.0f / (1.0f + exp(-params[8]));  // sigmoid(shared_gate_score)

    // Weighted sum of expert outputs
    float moe = 0.0f;
    // Unrolled for MAX_K=8 with branch on K to avoid reading invalid buffers
    if (K > 0) moe += params[0] * expert_out0[tid];
    if (K > 1) moe += params[1] * expert_out1[tid];
    if (K > 2) moe += params[2] * expert_out2[tid];
    if (K > 3) moe += params[3] * expert_out3[tid];
    if (K > 4) moe += params[4] * expert_out4[tid];
    if (K > 5) moe += params[5] * expert_out5[tid];
    if (K > 6) moe += params[6] * expert_out6[tid];
    if (K > 7) moe += params[7] * expert_out7[tid];

    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}
