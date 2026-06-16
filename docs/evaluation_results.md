# Evaluation Results

DTU distance metrics are error metrics: **lower is better** for Accuracy, Completeness, and Overall.

The tables below separate the local Python evaluation script (`matlab.py`) from the official MATLAB evaluation machine. The official MATLAB numbers should be used for primary comparisons.

## Official MATLAB Evaluation

| Method | Checkpoint / Output Tag | Acc Mean | Comp Mean | Overall | Note |
| --- | --- | ---: | ---: | ---: | --- |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.334233 | 0.286015 | 0.310124 | baseline reference |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.334978 | 0.278727 | 0.306852 | lower Overall than baseline |

Official comparison:

```text
SP-RWCV improves Overall by 0.003272 versus the plain baseline.
Relative change: -1.06% Overall error.
```

## Local Python Evaluation

| Method | Output Tag | Acc Mean | Comp Mean | Overall | Note |
| --- | --- | ---: | ---: | ---: | --- |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.313724 | 0.268163 | 0.290943 | local `matlab.py` |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.314965 | 0.261310 | 0.288137 | local `matlab.py` |

Local comparison:

```text
SP-RWCV improves Overall by 0.002806 versus the plain baseline.
Relative change: -0.96% Overall error.
```

## Raw CSV Files

- [Official plain baseline](results/official_plain_baseline_bs6_m015.csv)
- [Official SP-RWCV](results/official_sp_rwcv_bs5_m015.csv)
- [Local plain baseline](results/internal_plain_baseline_bs6_m015.csv)
- [Local SP-RWCV](results/internal_sp_rwcv_bs5_m015.csv)

## Current Interpretation

The first reliability-weighted source-view aggregation experiment gives a small but consistent reduction in Overall error compared with the plain baseline in both evaluation settings. The improvement mainly comes from lower Completeness error, while Accuracy is roughly unchanged or slightly higher.
