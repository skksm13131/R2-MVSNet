# Hybrid Baseline Rule 2026-06-04

## Why This Exists

The working tree combines a clean baseline model path with the newer training, testing, and evaluation workflow. This keeps the model easy to compare while preserving the scripts needed for the current server workflow.

## Source Rules

- Treat `models/` as the model source of truth.
- Treat `datasets/`, `train.py`, `test.py`, and scripts as workflow support.
- Do not assume old experimental modules are active just because they are present in the repository.
- Any active model change must be wired through `CascadeMVSNet` and exposed through matching train/test flags.

## Safe Additions

Safe additions are flag-gated and easy to ablate:

```bash
--use_view_attention
--view_attention_mode single_pass_reliability_weighted
--use_rafe
--use_adaptive_r2
```

## Unsafe Additions

Avoid directly mixing old dormant modules into the default path. Avoid changing feature extraction and cost-volume aggregation at the same time unless the combined path has a clear flag and a clean baseline fallback.

## Practical Check

Before launching a long training run:

1. Confirm plain baseline can still instantiate.
2. Confirm the new flagged model can instantiate.
3. Run a small tensor smoke test.
4. Keep command tags explicit and reproducible.
