# Coding Performance Benchmark

## Goal

Measure tok/s and TTFT on code-heavy workloads. Tests how fast the model generates code with varying prompt sizes — from short requests to prompts with real code context from this repository.

## Prompt Design

10 queries with increasing prefill complexity:

| # | Type | Context | What it tests |
|---|------|---------|---------------|
| 1 | Short codegen | None | Pure generation speed |
| 2 | Shader + context | 100 lines of shaders.metal | Medium prefill + code generation |
| 3 | Code review | 90 lines of infer.m | Large context comprehension speed |
| 4 | Refactor with context | 80 lines of config.h | Context + structured output |
| 5 | Algorithm from scratch | None | Sustained code generation |
| 6 | Multi-file synthesis | shaders.metal + config.h | Large prefill (~180 lines context) |
| 7 | Test generation | None | Repetitive structured output speed |
| 8 | Debug + fix | Detailed problem description | Reasoning + code output |
| 9 | SIMD optimization | None | Domain-specific code generation |
| 10 | System design | None | Long sustained generation |

Each query generates up to 200 tokens. The benchmark reports per-query tok/s and TTFT, plus aggregate min/avg/median/max.

## Running

```bash
cd metal_infer
./benchmark_coding.sh ../models/Qwen3-Coder-Next-4bit --k 10 --tokens 200
./benchmark_coding.sh ../models/Qwen3.5-122B-A10B-4bit --k 8 --tokens 200
```

## What it Measures

- **tok/s**: Code token generation throughput (excludes prefill)
- **TTFT**: Time to first token (dominated by prefill for context-heavy prompts)
- **Prefill tokens**: How many tokens the prompt expands to (code context is token-heavy)
- **Total time**: Wall clock for all 10 queries
