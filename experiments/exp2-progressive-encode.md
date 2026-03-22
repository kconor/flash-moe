# Experiment 2: Progressive Encoding (single CMD3)

## Goal
Encode expert forward passes into a single CMD3 progressively as each pread completes, overlapping encoding time with remaining I/O. Lower Metal overhead than Experiment 1 (one commit) but GPU can't start until commit.

## Proposed Flow
```
routing → async_pread_start(K experts, per-expert flags)
        → create CMD3
        → loop:
            for each k not yet encoded:
              if ready[k]: encode expert k into CMD3
        → after all K encoded: add combine + commit (deferred)
```

## Key Difference from Experiment 1
- Single CMD3 commit (less Metal overhead)
- GPU can't start any expert compute until all are encoded + committed
- Benefit: encoding time (~0.029ms) overlaps with remaining preads
- Expected smaller improvement than Exp 1, but lower risk

## Metrics to Track
- Same as Experiment 1
