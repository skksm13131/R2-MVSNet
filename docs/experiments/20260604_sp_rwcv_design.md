# SP-RWCV Design Note

SP-RWCV means:

```text
Single-Pass Reliability-Weighted Cost Volume
```

## Motivation

Earlier residual-fusion directions could improve quality, but they were slow because they required extra source-view processing or repeated homography warping. SP-RWCV keeps the cascade MVS pipeline close to baseline while adding a learned source-view reliability weight during variance cost-volume construction.

## Runtime Flags

```bash
--use_view_attention \
--view_attention_mode single_pass_reliability_weighted
```

## Per-Stage Flow

For each cascade stage:

1. Build the reference volume from the reference feature.
2. Homography-warp each source feature once.
3. Predict a source-view reliability score from reference/source agreement.
4. Convert the score into a bounded reliability weight.
5. Accumulate weighted feature sums and weighted squared sums.
6. Compute the weighted variance cost volume.
7. Send the cost volume to the normal 3D cost regularization and depth regression path.

## Weighting Formula

For source view `i`:

```text
w_i = 1 + residual_ratio * tanh(score_i / temperature)
```

The implementation clamps the final weight to keep training stable.

The weighted variance is:

```text
mean = sum(w_i * F_i) / sum(w_i)
variance = sum(w_i * F_i^2) / sum(w_i) - mean^2
```

The reference view keeps weight `1`.

## Difference From SCRF/RRF

- SP-RWCV performs reliability weighting in a single source-view warping pass.
- It changes the variance statistics directly instead of adding a separate residual cost-volume branch.
- It is designed to be faster and easier to ablate.

## Implementation Files

- `models/modules/view_attention.py`
  - `SinglePassReliabilityWeightedViewAttention`
- `models/cas_mvsnet.py`
  - `DepthNet.forward()` branch for `uses_single_pass_weighted_variance`
- `train.py` and `test.py`
  - expose `--view_attention_mode single_pass_reliability_weighted`

## Current Status

SP-RWCV gave a small but repeatable official DTU improvement over the plain baseline. RAFE and Adaptive R2 build on top of this path.
