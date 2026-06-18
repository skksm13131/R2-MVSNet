# Experiment Results Summary

DTU metrics are distance errors: lower is better.

## Official MATLAB Evaluation

| Method | Output Tag | Acc Mean | Comp Mean | Overall |
| --- | --- | ---: | ---: | ---: |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.334233 | 0.286015 | 0.310124 |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.334978 | 0.278727 | 0.306852 |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.334543 | 0.277197 | 0.305870 |

Official deltas:

```text
SP-RWCV vs baseline Overall: -0.003272
R2-MVSNet vs baseline Overall: -0.004254
R2-MVSNet vs SP-RWCV Overall: -0.000982
```

The R2 official evaluation was rerun once and reproduced exactly to six decimal places.

## Local Python Evaluation

| Method | Output Tag | Acc Mean | Comp Mean | Overall |
| --- | --- | ---: | ---: | ---: |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.313724 | 0.268163 | 0.290943 |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.314965 | 0.261310 | 0.288137 |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.314625 | 0.259278 | 0.286952 |

Local deltas:

```text
SP-RWCV vs baseline Overall: -0.002806
R2-MVSNet vs baseline Overall: -0.003991
R2-MVSNet vs SP-RWCV Overall: -0.001185
```

## R2 Single-Scene Official Comparison

R2 compared with SP-RWCV:

```text
R2 wins 9/22 scenes.
R2 loses 13/22 scenes.
```

R2 compared with baseline:

```text
R2 wins 13/22 scenes.
R2 loses 9/22 scenes.
```

Best R2 improvements versus SP-RWCV:

| Scan | R2 - SP-RWCV Overall |
| ---: | ---: |
| 29 | -0.065576 |
| 75 | -0.017469 |
| 33 | -0.012246 |
| 10 | -0.005690 |
| 32 | -0.003567 |

Worst R2 regressions versus SP-RWCV:

| Scan | R2 - SP-RWCV Overall |
| ---: | ---: |
| 13 | +0.024694 |
| 48 | +0.014188 |
| 4 | +0.013787 |
| 118 | +0.009202 |
| 1 | +0.005729 |

Interpretation:

R2 improves some difficult scenes but regresses several easier or stable scenes, so the average gain is small. Adaptive R2 was introduced to make reliability-aware changes selective rather than global.

## Raw CSVs

- `docs/results/official_plain_baseline_bs6_m015.csv`
- `docs/results/official_sp_rwcv_bs5_m015.csv`
- `docs/results/official_r2_rafe_sprwcv_bs4_m015.csv`
- `docs/results/internal_plain_baseline_bs6_m015.csv`
- `docs/results/internal_sp_rwcv_bs5_m015.csv`
- `docs/results/internal_r2_rafe_sprwcv_bs4_m015.csv`

## Active Experiment

Adaptive R2 is currently training:

```text
tag: 20260618_r2_adaptive_rafe_sprwcv_bs4_e16
flags: --use_rafe --use_adaptive_r2 --use_view_attention --view_attention_mode single_pass_reliability_weighted
batch_size: 4
epochs: 16
```
