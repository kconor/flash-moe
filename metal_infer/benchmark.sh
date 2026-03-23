#!/bin/bash
# benchmark.sh — Run 10 diverse prompts against a Flash-MoE model
#
# Usage: ./benchmark.sh <model_path> [--k N] [--tokens N]
#
# Starts the inference server, sends 10 prompts, collects timing stats.

set -euo pipefail

MODEL="${1:?Usage: ./benchmark.sh <model_path> [--k N] [--tokens N]}"
shift

# Parse optional args
K=8
TOKENS=100
PORT=8099
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
SERVER_PID=""
trap 'rm -f $STDERR_LOG $RESULTS_FILE 2>/dev/null; [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null' EXIT

echo "=== Flash-MoE Benchmark ==="
echo "Model:  $MODEL_NAME"
echo "K:      $K"
echo "Tokens: $TOKENS"
echo ""

# Build if needed
make -s -C "$(dirname "$0")" 2>/dev/null || true

# Start server
INFER="$(dirname "$0")/infer"
"$INFER" --model "$MODEL" --serve "$PORT" --k "$K" --timing 2>"$STDERR_LOG" &
SERVER_PID=$!

# Wait for health check
echo -n "Starting server..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo " FAILED (server died)"
        cat "$STDERR_LOG"
        exit 1
    fi
    sleep 0.5
done

# Benchmark prompts
PROMPTS=(
    "What is the capital of France?"
    "Write a Python function to compute fibonacci numbers"
    "Explain how a hash table works"
    "What are the differences between TCP and UDP?"
    "Write a bash script that finds the largest file in a directory"
    "Explain the theory of relativity in simple terms"
    "What is the time complexity of quicksort and why?"
    "Write a SQL query to find duplicate rows in a table"
    "How does garbage collection work in Java?"
    "Explain the CAP theorem and its implications for distributed systems"
)

RESULTS_FILE=${RESULTS_FILE:-$(mktemp)}
TOTAL_START=$(date +%s)

for i in "${!PROMPTS[@]}"; do
    QN=$((i + 1))
    PROMPT="${PROMPTS[$i]}"
    SHORT="${PROMPT:0:40}"
    SESSION="bench-$QN"

    # Clear stderr marker
    MARK_BEFORE=$(wc -l < "$STDERR_LOG")

    # Send request, consume SSE stream, extract content
    RESPONSE=$(curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": $TOKENS,
            \"stream\": true,
            \"session_id\": \"$SESSION\"
        }" 2>/dev/null || echo "ERROR")

    # Parse timing from stderr (new lines since marker)
    STDERR_NEW=$(tail -n +$((MARK_BEFORE + 1)) "$STDERR_LOG")

    PREFILL_MS=$(echo "$STDERR_NEW" | grep -o "prefill=[0-9]* tokens in [0-9]*ms" | head -1 | grep -o "[0-9]*ms" | grep -o "[0-9]*")
    GEN_TOKS=$(echo "$STDERR_NEW" | grep -o "generated=[0-9]*" | head -1 | grep -o "[0-9]*")
    TOK_S=$(echo "$STDERR_NEW" | grep -o "[0-9]*\.[0-9]* tok/s" | head -1 | grep -o "[0-9]*\.[0-9]*")

    PREFILL_MS=${PREFILL_MS:-0}
    GEN_TOKS=${GEN_TOKS:-0}
    TOK_S=${TOK_S:-0}

    printf "Query %2d: %3s tokens, %6s tok/s, TTFT %5sms  \"%s\"\n" \
        "$QN" "$GEN_TOKS" "$TOK_S" "$PREFILL_MS" "$SHORT"

    echo "$TOK_S $PREFILL_MS $GEN_TOKS" >> "$RESULTS_FILE"
done

TOTAL_END=$(date +%s)
TOTAL_SECS=$((TOTAL_END - TOTAL_START))

echo ""
echo "--- Summary ---"

# Compute stats with awk
awk '
BEGIN { n=0; sum_tok=0; sum_ttft=0; sum_gen=0; min_tok=999; max_tok=0; min_ttft=999999; max_ttft=0 }
{
    tok=$1; ttft=$2; gen=$3
    if (tok+0 > 0) {
        toks[n] = tok
        ttfts[n] = ttft
        n++
        sum_tok += tok
        sum_gen += gen
        sum_ttft += ttft
        if (tok < min_tok) min_tok = tok
        if (tok > max_tok) max_tok = tok
        if (ttft < min_ttft) min_ttft = ttft
        if (ttft > max_ttft) max_ttft = ttft
    }
}
END {
    if (n == 0) { print "No results"; exit }

    # Sort for median (simple bubble sort)
    for (i=0; i<n-1; i++)
        for (j=i+1; j<n; j++)
            if (toks[i] > toks[j]) { t=toks[i]; toks[i]=toks[j]; toks[j]=t }
    for (i=0; i<n-1; i++)
        for (j=i+1; j<n; j++)
            if (ttfts[i] > ttfts[j]) { t=ttfts[i]; ttfts[i]=ttfts[j]; ttfts[j]=t }

    med_tok = (n%2==1) ? toks[int(n/2)] : (toks[n/2-1]+toks[n/2])/2
    med_ttft = (n%2==1) ? ttfts[int(n/2)] : (ttfts[n/2-1]+ttfts[n/2])/2

    printf "Queries:    %d\n", n
    printf "Tokens:     %d total\n", sum_gen
    printf "tok/s:      min=%.2f  avg=%.2f  median=%.2f  max=%.2f\n", min_tok, sum_tok/n, med_tok, max_tok
    printf "TTFT:       min=%dms  avg=%dms  median=%dms  max=%dms\n", min_ttft, sum_ttft/n, med_ttft, max_ttft
}
' "$RESULTS_FILE"

printf "Total time: %ds\n" "$TOTAL_SECS"

# Cleanup
rm -f "$RESULTS_FILE"
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "Done."
