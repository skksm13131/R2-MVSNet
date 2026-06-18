# Model Improvement Log

## Baseline

Plain CasMVSNet remains the control path:

```text
FeatureNet -> cascade depth sampling -> homography warping -> variance cost volume -> cost regularization -> depth regression
```

The no-attention/no-RAFE path must remain runnable.

## SP-RWCV

Full name:

```text
Single-Pass Reliability-Weighted Cost Volume
```

Runtime flags:

```bash
--use_view_attention \
--view_attention_mode single_pass_reliability_weighted
```

Main files:

- `models/modules/view_attention.py`
- `models/cas_mvsnet.py`

Core idea:

- Replace equal source-view contribution in variance statistics with bounded learned reliability weights.
- Avoid the second homography-warping pass used by earlier residual-fusion designs.
- Keep FeatureNet, CostRegNet, depth regression, and fusion otherwise close to baseline.

Weighted variance:

```text
weighted_mean = sum(w_i * F_i) / sum(w_i)
weighted_variance = sum(w_i * F_i^2) / sum(w_i) - weighted_mean^2
```

## RAFE

Full name:

```text
Reliability-Aware Feature Extraction
```

Runtime flag:

```bash
--use_rafe
```

Main file:

- `models/module.py`

Core idea:

- FeatureNet predicts multi-scale reliability maps:
  - `stage1_reliability`
  - `stage2_reliability`
  - `stage3_reliability`
- Structure prior uses grayscale, gradients, gradient magnitude, local texture variance, and x/y coordinates.
- The feature adapter injects a residual structure-aware feature:

```text
feature_out = feature + residual_scale * gate * reliability_gate * prior_feature
```

RAFE and SP-RWCV link:

- RAFE reliability maps are homography-warped with source features.
- SP-RWCV consumes reference/source feature reliability while predicting source-view weights.

## R2-MVSNet

Full name:

```text
RAFE + SP-RWCV
```

Runtime flags:

```bash
--use_rafe \
--use_view_attention \
--view_attention_mode single_pass_reliability_weighted
```

Observation from official DTU single-scene results:

- Difficult scenes such as scan 29, 33, and 75 improved.
- Several simpler or more stable scenes regressed slightly.
- The average official improvement is real but small because gains and regressions cancel.

## Adaptive R2

Runtime flag:

```bash
--use_adaptive_r2
```

Use together with RAFE and SP-RWCV.

Purpose:

- Keep the difficult-scene gains.
- Reduce simple-scene regressions by making both feature residual injection and source-view reweighting less global.

RAFE change:

```text
feature_out = feature + residual_scale * gate * reliability_gate * difficulty_gate * prior_feature
```

The RAFE `difficulty_gate` combines:

- learned difficulty from feature/prior statistics
- reliability uncertainty
- normalized prior energy

SP-RWCV change:

```text
weight = 1 + residual_ratio * difficulty_gate * pixel_residual
```

The SP-RWCV `difficulty_gate` combines:

- source-view score magnitude
- previous-stage confidence uncertainty when available
- feature reliability uncertainty when available

Expected behavior:

- Easy/high-confidence regions move closer to the baseline variance path.
- Difficult/low-reliability regions retain stronger reliability weighting.

Current adaptive training tag:

```text
20260618_r2_adaptive_rafe_sprwcv_bs4_e16
```
