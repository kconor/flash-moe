# Experiment 1: Per-Expert Command Buffers

## Goal
Overlap GPU expert compute with SSD I/O by submitting per-expert command buffers as each pread completes, instead of waiting for all K experts to load before encoding a single CMD3.

## Current Flow
```
routing → async_pread_start(K experts) → wait ALL K → encode ALL K into CMD3 → commit
                                          ^^^^^^^^^^^^
                                          BLOCKED: cached experts idle while cold ones load
```

## Proposed Flow
```
routing → async_pread_start(K experts, per-expert flags)
        → loop:
            poll each expert's completion flag
            if ready: encode into per-expert CMD → commit → GPU starts immediately
        → after all K done: encode combine CMD → commit (deferred)
```

## Implementation Strategy

### Changes to async pread (infer.m)
- Add `volatile int ready[MAX_K]` to `AsyncPreadState`
- Each GCD block sets `ready[k] = 1` when its pread completes
- Remove single `dispatch_group_wait` — caller polls `ready[]` instead

### Changes to CMD3 dispatch (fused_layer_forward)
- Replace `gpu_encode_experts_batched` (one CMD, all K) with per-expert loop:
  ```
  for each k:
    while (!ready[k]) { /* check others first */ }
    encode expert k into cmd_expert[k]
    commit cmd_expert[k]  // GPU starts immediately
  ```
- After all K: encode combine kernel into cmd_combine, commit deferred
- Save all K+1 command buffers in deferred state

### Changes to deferred completion
- `wait_deferred_experts_gpu`: wait for cmd_combine (which depends on all per-expert CMDs via GPU queue ordering)
- Actually: Metal command queue serializes submissions, so if we submit expert CMDs then combine CMD to same queue, the GPU executes them in order. Single wait on combine CMD is sufficient.

### Risk
- Metal command buffer creation overhead (~0.01ms each) × K experts
- K separate commits vs 1 commit
- Queue serialization means GPU still processes experts in submission order, not necessarily optimal

## Metrics to Track
- `expert_io` phase time (should decrease — no longer includes wait-for-all)
- `cmd3_encode` phase time (may increase — K commits vs 1)
- `total_layer` (net improvement?)
- tok/s before vs after
