# True Baseline Model Audit 2026-06-04

## Scope

This note records the cleaned true-baseline status used before adding the current reliability modules.

## Baseline Path

When no optional reliability or attention flags are enabled, the active model follows the standard CasMVSNet pipeline:

1. Build reference and source image features with `FeatureNet`.
2. Build the reference volume.
3. Homography-warp source features into the reference camera.
4. Construct an equal-weight variance cost volume.
5. Run 3D CNN cost regularization.
6. Apply softmax and depth regression.

## Active Optional Path

The optional source-view reliability path starts from:

- `models/cas_mvsnet.py`
- `models/modules/view_attention.py`
- `train.py`
- `test.py`

The important runtime flags are:

```bash
--use_view_attention
--view_attention_mode <mode>
```

Later work adds:

```bash
--use_rafe
--use_adaptive_r2
```

## Removed Or Dormant Ideas

Several older ideas existed in earlier working directories, including Direct-SCRF v2, RMFE, UGDR, CADR, RAHS, normal guidance, and geometry guidance. They should not be treated as part of the active baseline unless a flag is explicitly wired through the current `CascadeMVSNet`.

## Rule

The true baseline must stay runnable. New model behavior should be optional, documented, and easy to ablate.
