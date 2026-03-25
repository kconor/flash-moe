#!/bin/bash
# benchmark_coding.sh — Coding performance benchmark
#
# Measures tok/s and TTFT on code-heavy prompts that include real code
# context from this repository. Tests long-context prefill (large prompts)
# and code generation throughput.
#
# Usage: ./benchmark_coding.sh <model_path> [--k N] [--tokens N]

set -euo pipefail

MODEL="${1:?Usage: ./benchmark_coding.sh <model_path> [--k N] [--tokens N]}"
shift
K=8
TOKENS=200
PORT=8097
while [[ $# -gt 0 ]]; do
    case "$1" in
        --k) K="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODEL_NAME=$(basename "$MODEL")
STDERR_LOG=$(mktemp)
RESULTS_FILE=$(mktemp)
SERVER_PID=""
trap 'rm -f "$STDERR_LOG" "$RESULTS_FILE"; [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null' EXIT

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Flash-MoE Coding Benchmark ==="
echo "Model:  $MODEL_NAME"
echo "K:      $K"
echo "Tokens: $TOKENS"
echo ""

make -s -C "$SCRIPT_DIR" 2>/dev/null || true

PREDICTOR_FILE="${SCRIPT_DIR}/../models/$(basename "$MODEL")/predictor_coding.bin"
"$SCRIPT_DIR/infer" --model "$MODEL" --serve "$PORT" --k "$K" --collect-predictor "$PREDICTOR_FILE" 2>"$STDERR_LOG" &
SERVER_PID=$!

echo -n "Starting server..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo " FAILED"
        cat "$STDERR_LOG"
        exit 1
    fi
    sleep 0.5
done

# Read real code snippets from the repo for context
SHADER_SNIPPET=$(head -100 "$SCRIPT_DIR/shaders.metal" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
CONFIG_SNIPPET=$(head -80 "$SCRIPT_DIR/config.h" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
INFER_SNIPPET=$(sed -n '900,990p' "$SCRIPT_DIR/infer.m" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')

send_prompt() {
    local session="$1"
    local prompt_json="$2"
    local max_tokens="$3"
    local mark_before=$(wc -l < "$STDERR_LOG")

    curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"messages\": [{\"role\": \"user\", \"content\": $prompt_json}],
            \"max_tokens\": $max_tokens,
            \"stream\": true,
            \"session_id\": \"$session\"
        }" >/dev/null 2>&1

    local stderr_new=$(tail -n +$((mark_before + 1)) "$STDERR_LOG")
    local prefill_ms=$(echo "$stderr_new" | grep -o "prefill=[0-9]* tokens in [0-9]*ms" | head -1 | grep -o "[0-9]*ms" | grep -o "[0-9]*")
    local gen_toks=$(echo "$stderr_new" | grep -o "generated=[0-9]*" | head -1 | grep -o "[0-9]*")
    local tok_s=$(echo "$stderr_new" | grep -o "[0-9]*\.[0-9]* tok/s" | head -1 | grep -o "[0-9]*\.[0-9]*")
    local prefill_toks=$(echo "$stderr_new" | grep -o "prefill=[0-9]*" | head -1 | grep -o "[0-9]*")

    echo "${tok_s:-0} ${prefill_ms:-0} ${gen_toks:-0} ${prefill_toks:-0}"
}

TOTAL_START=$(date +%s)

# ── Prompt 1: Short code generation (small prefill, pure generation speed)
echo ""
P1=$(python3 -c 'import json; print(json.dumps("Write a Python implementation of a binary search tree with insert, search, and delete operations."))')
echo -n "Query  1: short codegen .............. "
R1=$(send_prompt "code1" "$P1" "$TOKENS")
echo "$R1" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R1"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 2: Code with repo context (medium prefill)
P2=$(python3 -c "import json; print(json.dumps('Here is a Metal compute shader file:\n\n' + ${SHADER_SNIPPET} + '\n\nAdd a new kernel called dequant_matvec_3bit that handles 3-bit quantized weights. Follow the same pattern as the 4-bit and 8-bit kernels. Write the full kernel implementation.'))")
echo -n "Query  2: shader + context ........... "
R2=$(send_prompt "code2" "$P2" "$TOKENS")
echo "$R2" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R2"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 3: Code review with large context
P3=$(python3 -c "import json; print(json.dumps('Review this C struct and its initialization code for bugs and improvements:\n\n' + ${INFER_SNIPPET} + '\n\nList each issue found and provide corrected code.'))")
echo -n "Query  3: code review ................ "
R3=$(send_prompt "code3" "$P3" "$TOKENS")
echo "$R3" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R3"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 4: Architecture refactor (config.h context)
P4=$(python3 -c "import json; print(json.dumps('Here is a C header for model configuration:\n\n' + ${CONFIG_SNIPPET} + '\n\nRefactor this to support loading configuration from a YAML file at runtime instead of compile-time defines. Write the new struct definition and a load_config_yaml() function.'))")
echo -n "Query  4: refactor with context ...... "
R4=$(send_prompt "code4" "$P4" "$TOKENS")
echo "$R4" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R4"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 5: Algorithm implementation
P5=$(python3 -c 'import json; print(json.dumps("Implement an LRU cache in C that supports O(1) insert, lookup, and eviction. Use a hash table with chaining and a doubly-linked list. Include the full implementation with struct definitions, init, lookup, insert, and free functions."))')
echo -n "Query  5: algorithm from scratch ..... "
R5=$(send_prompt "code5" "$P5" "$TOKENS")
echo "$R5" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R5"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 6: Multi-file understanding (both snippets)
P6=$(python3 -c "import json; print(json.dumps('I have a Metal shader file and a C configuration header:\n\nShader:\n' + ${SHADER_SNIPPET} + '\n\nConfig:\n' + ${CONFIG_SNIPPET} + '\n\nWrite a Python script that reads the config, generates Metal shader source code with the correct dimensions compiled in as constants, and writes it to a file.'))")
echo -n "Query  6: multi-file synthesis ....... "
R6=$(send_prompt "code6" "$P6" "$TOKENS")
echo "$R6" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R6"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 7: Test generation
P7=$(python3 -c 'import json; print(json.dumps("Write a comprehensive test suite in Python for a function called compute_expert_layout(hidden_dim, moe_intermediate, group_size, bits) that computes byte offsets for quantized expert weight components. The function returns (components_list, expert_size). Test 4-bit, 8-bit, and edge cases. Use pytest."))')
echo -n "Query  7: test generation ............ "
R7=$(send_prompt "code7" "$P7" "$TOKENS")
echo "$R7" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R7"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 8: Debug analysis
P8=$(python3 -c 'import json; print(json.dumps("A Metal inference engine produces garbage output for one model but works for another. Both use GatedDeltaNet linear attention. The working model uses separate projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) while the broken model uses fused projections (in_proj_qkvz, in_proj_ba). The fused output is interleaved per k_head group: [q0(128),k0(128),v0v1(256),z0z1(256)] x 16 heads = 12288 total. The code splits it as flat [first 8192 = qkv, last 4096 = z] without de-interleaving. Write the de-interleaving function in C."))')
echo -n "Query  8: debug + fix ................ "
R8=$(send_prompt "code8" "$P8" "$TOKENS")
echo "$R8" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R8"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 9: Optimization task
P9=$(python3 -c 'import json; print(json.dumps("Write an optimized SIMD implementation of bf16_to_f32 batch conversion for ARM NEON. The function takes a uint16_t array of N bf16 values and writes N float32 values. Use NEON intrinsics for 4-wide processing. Include a scalar fallback for the tail elements."))')
echo -n "Query  9: SIMD optimization .......... "
R9=$(send_prompt "code9" "$P9" "$TOKENS")
echo "$R9" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R9"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

# ── Prompt 10: Full system design
P10=$(python3 -c 'import json; print(json.dumps("Design and implement an HTTP server in C that serves an OpenAI-compatible chat completion API. It should: parse JSON requests with messages array, support streaming SSE responses, handle multiple sessions with unique IDs, and return proper HTTP headers. Write the complete implementation including the HTTP parser, JSON handling, and SSE response formatting."))')
echo -n "Query 10: system design .............. "
R10=$(send_prompt "code10" "$P10" "$TOKENS")
echo "$R10" >> "$RESULTS_FILE"
read toks ttft gen_n pfn <<< "$R10"
printf "%3s tokens, %6s tok/s, TTFT %5sms (prefill %s tok)\n" "$gen_n" "$toks" "$ttft" "$pfn"

TOTAL_END=$(date +%s)
TOTAL_SECS=$((TOTAL_END - TOTAL_START))

echo ""
echo "--- Summary ---"
awk '
BEGIN { n=0; sum_tok=0; sum_ttft=0; sum_gen=0; min_tok=999; max_tok=0; min_ttft=999999; max_ttft=0 }
{
    tok=$1; ttft=$2; gen=$3
    if (tok+0 > 0) {
        toks[n] = tok; ttfts[n] = ttft; n++
        sum_tok += tok; sum_gen += gen; sum_ttft += ttft
        if (tok < min_tok) min_tok = tok
        if (tok > max_tok) max_tok = tok
        if (ttft < min_ttft) min_ttft = ttft
        if (ttft > max_ttft) max_ttft = ttft
    }
}
END {
    if (n == 0) { print "No results"; exit }
    for (i=0; i<n-1; i++) for (j=i+1; j<n; j++) if (toks[i]>toks[j]) { t=toks[i]; toks[i]=toks[j]; toks[j]=t }
    for (i=0; i<n-1; i++) for (j=i+1; j<n; j++) if (ttfts[i]>ttfts[j]) { t=ttfts[i]; ttfts[i]=ttfts[j]; ttfts[j]=t }
    med_tok = (n%2==1) ? toks[int(n/2)] : (toks[n/2-1]+toks[n/2])/2
    med_ttft = (n%2==1) ? ttfts[int(n/2)] : (ttfts[n/2-1]+ttfts[n/2])/2
    printf "Queries:    %d\n", n
    printf "Tokens:     %d total\n", sum_gen
    printf "tok/s:      min=%.2f  avg=%.2f  median=%.2f  max=%.2f\n", min_tok, sum_tok/n, med_tok, max_tok
    printf "TTFT:       min=%dms  avg=%dms  median=%dms  max=%dms\n", min_ttft, sum_ttft/n, med_ttft, max_ttft
}
' "$RESULTS_FILE"

printf "Total time: %ds\n" "$TOTAL_SECS"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo ""
echo "Predictor data: $PREDICTOR_FILE ($(du -h "$PREDICTOR_FILE" 2>/dev/null | awk '{print $1}'))"
echo "Done."
