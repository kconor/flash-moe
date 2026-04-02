/*
 * config.h — Runtime model configuration for Flash-MoE inference engine
 *
 * Replaces hardcoded #define constants with a runtime config loaded from
 * model_weights.json. Supports any Qwen3.5 MoE variant (397B, 35B, etc.).
 *
 * Usage:
 *   1. Include this header
 *   2. Define: static ModelConfig *g_cfg = NULL;
 *   3. Load config: g_cfg = load_model_config("model_weights.json");
 *   4. All redirect macros (HIDDEN_DIM, NUM_LAYERS, etc.) work transparently
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Model configuration struct
// ============================================================================

typedef struct {
    // Core dimensions
    int hidden_dim;
    int num_layers;
    int num_attn_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    float rms_norm_eps;

    // MoE
    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate;
    int shared_intermediate;

    // Attention pattern
    int full_attn_interval;
    int num_full_attn_layers;   // computed: num_layers / full_attn_interval
    int num_linear_layers;      // computed: num_layers - num_full_attn_layers

    // Linear attention (GatedDeltaNet)
    int linear_num_v_heads;
    int linear_num_k_heads;
    int linear_key_dim;
    int linear_value_dim;
    int linear_total_key;       // computed: linear_num_k_heads * linear_key_dim
    int linear_total_value;     // computed: linear_num_v_heads * linear_value_dim
    int linear_conv_dim;        // computed: linear_total_key * 2 + linear_total_value
    int conv_kernel_size;

    // RoPE
    float rope_theta;
    float partial_rotary;
    int rotary_dim;             // computed: (int)(head_dim * partial_rotary)

    // Quantization
    int group_size;
    int bits;                   // default quantization (4, 8, etc.)
    int down_bits;              // expert down_proj bits (0 = same as bits)

    // Special tokens (model-dependent)
    int eos_token_1;
    int eos_token_2;
    int think_start_token;
    int think_end_token;

    // Expert binary layout — primary quantization (cfg->bits)
    size_t expert_size;
    size_t gate_w_off, gate_w_size;
    size_t gate_s_off, gate_s_size;
    size_t gate_b_off, gate_b_size;
    size_t up_w_off, up_w_size;
    size_t up_s_off, up_s_size;
    size_t up_b_off, up_b_size;
    size_t down_w_off, down_w_size;
    size_t down_s_off, down_s_size;
    size_t down_b_off, down_b_size;

    // Expert binary layout — 2-bit (for --2bit runtime option)
    size_t expert_size_2bit;
    size_t gate_w_off_2, gate_s_off_2, gate_b_off_2;
    size_t up_w_off_2, up_s_off_2, up_b_off_2;
    size_t down_w_off_2, down_s_off_2, down_b_off_2;
} ModelConfig;

// ============================================================================
// Expert binary layout computation
// ============================================================================

// Compute expert layout for given bit widths.
// Layout: gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b
// gate_up_bits: bit width for gate/up projections
// down_bits: bit width for down projection (may differ in mixed-precision models)
static inline void compute_expert_layout_bits(
    int hidden_dim, int moe_intermediate, int group_size, int gate_up_bits, int down_bits,
    size_t *out_expert_size,
    size_t *gw_off, size_t *gw_sz, size_t *gs_off, size_t *gs_sz, size_t *gb_off, size_t *gb_sz,
    size_t *uw_off, size_t *uw_sz, size_t *us_off, size_t *us_sz, size_t *ub_off, size_t *ub_sz,
    size_t *dw_off, size_t *dw_sz, size_t *ds_off, size_t *ds_sz, size_t *db_off, size_t *db_sz
) {
    int in_dim = hidden_dim;
    int mid = moe_intermediate;
    int gs = group_size;

    // gate/up: [mid, in_dim] -> packed [mid, in_dim * gate_up_bits / 32] uint32
    // Uses bits*dim/32 to handle non-power-of-2 bit widths (5, 6, etc.)
    size_t gate_up_packed_cols = (size_t)in_dim * gate_up_bits / 32;
    size_t w_size = (size_t)mid * gate_up_packed_cols * sizeof(uint32_t);
    // scales/biases: [mid, in_dim/gs] bf16 (independent of bit width)
    size_t sb_size = (size_t)mid * (in_dim / gs) * sizeof(uint16_t);

    // down: [in_dim, mid] -> packed [in_dim, mid * down_bits / 32] uint32
    size_t down_packed_cols = (size_t)mid * down_bits / 32;
    size_t dw_size_ = (size_t)in_dim * down_packed_cols * sizeof(uint32_t);
    // scales/biases: [in_dim, mid/gs] bf16
    size_t dsb_size = (size_t)in_dim * (mid / gs) * sizeof(uint16_t);

    size_t off = 0;
    *gw_off = off; *gw_sz = w_size;    off += w_size;
    *gs_off = off; *gs_sz = sb_size;   off += sb_size;
    *gb_off = off; *gb_sz = sb_size;   off += sb_size;
    *uw_off = off; *uw_sz = w_size;    off += w_size;
    *us_off = off; *us_sz = sb_size;   off += sb_size;
    *ub_off = off; *ub_sz = sb_size;   off += sb_size;
    *dw_off = off; *dw_sz = dw_size_;  off += dw_size_;
    *ds_off = off; *ds_sz = dsb_size;  off += dsb_size;
    *db_off = off; *db_sz = dsb_size;  off += dsb_size;
    *out_expert_size = off;
}

// Compute all derived fields and expert layouts
static inline void model_config_compute(ModelConfig *cfg) {
    // Derived attention layer counts
    cfg->num_full_attn_layers = cfg->num_layers / cfg->full_attn_interval;
    cfg->num_linear_layers = cfg->num_layers - cfg->num_full_attn_layers;

    // Derived linear attention dims
    cfg->linear_total_key = cfg->linear_num_k_heads * cfg->linear_key_dim;
    cfg->linear_total_value = cfg->linear_num_v_heads * cfg->linear_value_dim;
    cfg->linear_conv_dim = cfg->linear_total_key * 2 + cfg->linear_total_value;

    // Derived RoPE
    cfg->rotary_dim = (int)(cfg->head_dim * cfg->partial_rotary);

    // Resolve down_bits: 0 means same as bits
    int eff_down_bits = cfg->down_bits > 0 ? cfg->down_bits : cfg->bits;

    // Primary expert layout (cfg->bits for gate/up, eff_down_bits for down)
    compute_expert_layout_bits(
        cfg->hidden_dim, cfg->moe_intermediate, cfg->group_size, cfg->bits, eff_down_bits,
        &cfg->expert_size,
        &cfg->gate_w_off, &cfg->gate_w_size, &cfg->gate_s_off, &cfg->gate_s_size,
        &cfg->gate_b_off, &cfg->gate_b_size,
        &cfg->up_w_off, &cfg->up_w_size, &cfg->up_s_off, &cfg->up_s_size,
        &cfg->up_b_off, &cfg->up_b_size,
        &cfg->down_w_off, &cfg->down_w_size, &cfg->down_s_off, &cfg->down_s_size,
        &cfg->down_b_off, &cfg->down_b_size
    );

    // 2-bit expert layout (for --2bit option)
    size_t dummy_sz;  // we only store offsets for 2-bit, sizes same structure
    compute_expert_layout_bits(
        cfg->hidden_dim, cfg->moe_intermediate, cfg->group_size, 2, 2,
        &cfg->expert_size_2bit,
        &cfg->gate_w_off_2, &dummy_sz, &cfg->gate_s_off_2, &dummy_sz,
        &cfg->gate_b_off_2, &dummy_sz,
        &cfg->up_w_off_2, &dummy_sz, &cfg->up_s_off_2, &dummy_sz,
        &cfg->up_b_off_2, &dummy_sz,
        &cfg->down_w_off_2, &dummy_sz, &cfg->down_s_off_2, &dummy_sz,
        &cfg->down_b_off_2, &dummy_sz
    );

    printf("[config] %s: %d layers (%d linear + %d full-attn), "
           "hidden=%d, experts=%d, intermediate=%d, bits=%d%s\n",
           cfg->num_layers == 60 ? "Qwen3.5-397B" :
           cfg->num_layers == 40 ? "Qwen3.5-35B" : "Qwen3.5",
           cfg->num_layers, cfg->num_linear_layers, cfg->num_full_attn_layers,
           cfg->hidden_dim, cfg->num_experts, cfg->moe_intermediate, cfg->bits,
           eff_down_bits != cfg->bits ? " (mixed-precision)" : "");
    printf("[config] Expert size: %zu bytes (%zu 2-bit)\n",
           cfg->expert_size, cfg->expert_size_2bit);
}

// Return a default config (Qwen3.5-397B-A17B) for backward compatibility
static inline ModelConfig *model_config_default(void) {
    ModelConfig *cfg = calloc(1, sizeof(ModelConfig));
    cfg->hidden_dim = 4096;
    cfg->num_layers = 60;
    cfg->num_attn_heads = 32;
    cfg->num_kv_heads = 2;
    cfg->head_dim = 256;
    cfg->vocab_size = 248320;
    cfg->rms_norm_eps = 1e-6f;
    cfg->num_experts = 512;
    cfg->num_experts_per_tok = 10;
    cfg->moe_intermediate = 1024;
    cfg->shared_intermediate = 1024;
    cfg->full_attn_interval = 4;
    cfg->linear_num_v_heads = 64;
    cfg->linear_num_k_heads = 16;
    cfg->linear_key_dim = 128;
    cfg->linear_value_dim = 128;
    cfg->conv_kernel_size = 4;
    cfg->rope_theta = 10000000.0f;
    cfg->partial_rotary = 0.25f;
    cfg->group_size = 64;
    cfg->bits = 4;
    cfg->eos_token_1 = 248046;
    cfg->eos_token_2 = 248044;
    cfg->think_start_token = 248068;
    cfg->think_end_token = 248069;
    model_config_compute(cfg);
    return cfg;
}

// ============================================================================
// Redirect macros — backward compatibility with existing code
// Requires: static ModelConfig *g_cfg declared in the including file
// ============================================================================

// Core dimensions
#define HIDDEN_DIM          (g_cfg->hidden_dim)
#define NUM_LAYERS          (g_cfg->num_layers)
#define NUM_ATTN_HEADS      (g_cfg->num_attn_heads)
#define NUM_KV_HEADS        (g_cfg->num_kv_heads)
#define HEAD_DIM            (g_cfg->head_dim)
#define VOCAB_SIZE          (g_cfg->vocab_size)
#define RMS_NORM_EPS        (g_cfg->rms_norm_eps)
#define NUM_EXPERTS         (g_cfg->num_experts)
#define NUM_EXPERTS_PER_TOK (g_cfg->num_experts_per_tok)
#define MOE_INTERMEDIATE    (g_cfg->moe_intermediate)
#define SHARED_INTERMEDIATE (g_cfg->shared_intermediate)
#define FULL_ATTN_INTERVAL  (g_cfg->full_attn_interval)
#define GROUP_SIZE          (g_cfg->group_size)
#define BITS                (g_cfg->bits)

// Special tokens
#define EOS_TOKEN_1         (g_cfg->eos_token_1)
#define EOS_TOKEN_2         (g_cfg->eos_token_2)
#define THINK_START_TOKEN   (g_cfg->think_start_token)
#define THINK_END_TOKEN     (g_cfg->think_end_token)

// Linear attention
#define LINEAR_NUM_V_HEADS  (g_cfg->linear_num_v_heads)
#define LINEAR_NUM_K_HEADS  (g_cfg->linear_num_k_heads)
#define LINEAR_KEY_DIM      (g_cfg->linear_key_dim)
#define LINEAR_VALUE_DIM    (g_cfg->linear_value_dim)
#define LINEAR_TOTAL_KEY    (g_cfg->linear_total_key)
#define LINEAR_TOTAL_VALUE  (g_cfg->linear_total_value)
#define LINEAR_CONV_DIM     (g_cfg->linear_conv_dim)
#define CONV_KERNEL_SIZE    (g_cfg->conv_kernel_size)

// RoPE
#define ROPE_THETA          (g_cfg->rope_theta)
#define PARTIAL_ROTARY      (g_cfg->partial_rotary)
#define ROTARY_DIM          (g_cfg->rotary_dim)

// Expert layout — primary quantization
#define EXPERT_SIZE         (g_cfg->expert_size)
#define GATE_W_OFF          (g_cfg->gate_w_off)
#define GATE_S_OFF          (g_cfg->gate_s_off)
#define GATE_B_OFF          (g_cfg->gate_b_off)
#define UP_W_OFF            (g_cfg->up_w_off)
#define UP_S_OFF            (g_cfg->up_s_off)
#define UP_B_OFF            (g_cfg->up_b_off)
#define DOWN_W_OFF          (g_cfg->down_w_off)
#define DOWN_S_OFF          (g_cfg->down_s_off)
#define DOWN_B_OFF          (g_cfg->down_b_off)

// Expert layout — 2-bit
#define EXPERT_SIZE_2BIT    (g_cfg->expert_size_2bit)
#define GATE_W_OFF_2        (g_cfg->gate_w_off_2)
#define GATE_S_OFF_2        (g_cfg->gate_s_off_2)
#define GATE_B_OFF_2        (g_cfg->gate_b_off_2)
#define UP_W_OFF_2          (g_cfg->up_w_off_2)
#define UP_S_OFF_2          (g_cfg->up_s_off_2)
#define UP_B_OFF_2          (g_cfg->up_b_off_2)
#define DOWN_W_OFF_2        (g_cfg->down_w_off_2)
#define DOWN_S_OFF_2        (g_cfg->down_s_off_2)
#define DOWN_B_OFF_2        (g_cfg->down_b_off_2)

// Derived attention layer counts
#define NUM_FULL_ATTN_LAYERS (g_cfg->num_full_attn_layers)
#define NUM_LINEAR_LAYERS    (g_cfg->num_linear_layers)

// Maximum for struct array sizing (must be compile-time constant)
#define MAX_MODEL_LAYERS 128

#endif // CONFIG_H
