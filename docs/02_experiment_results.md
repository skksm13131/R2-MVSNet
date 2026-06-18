# 02 实验结果

DTU 的 Accuracy、Completeness、Overall 都是距离误差，越低越好。

论文或正式汇报优先使用官方 MATLAB 评估。本地 `matlab.py` 结果只作为快速观察和 sanity check。

## 官方 MATLAB 评估

| 方法 | 输出 tag | Acc Mean | Comp Mean | Overall |
| --- | --- | ---: | ---: | ---: |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.334233 | 0.286015 | 0.310124 |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.334978 | 0.278727 | 0.306852 |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.334543 | 0.277197 | 0.305870 |

官方 Overall delta：

```text
SP-RWCV vs baseline: -0.003272
R2-MVSNet vs baseline: -0.004254
R2-MVSNet vs SP-RWCV: -0.000982
```

说明：R2 官方评估复跑过一次，22 个 scan 的 CSV 结果到 6 位小数完全一致。

## 本地 Python 评估

| 方法 | 输出 tag | Acc Mean | Comp Mean | Overall |
| --- | --- | ---: | ---: | ---: |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.313724 | 0.268163 | 0.290943 |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.314965 | 0.261310 | 0.288137 |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.314625 | 0.259278 | 0.286952 |

本地 Overall delta：

```text
SP-RWCV vs baseline: -0.002806
R2-MVSNet vs baseline: -0.003991
R2-MVSNet vs SP-RWCV: -0.001185
```

## 单场景观察

R2 相比 SP-RWCV：

```text
9/22 个场景更好
13/22 个场景更差
```

R2 相比 plain baseline：

```text
13/22 个场景更好
9/22 个场景更差
```

R2 相比 SP-RWCV 的最大收益：

| Scan | R2 - SP-RWCV Overall |
| ---: | ---: |
| 29 | -0.065576 |
| 75 | -0.017469 |
| 33 | -0.012246 |
| 10 | -0.005690 |
| 32 | -0.003567 |

R2 相比 SP-RWCV 的最大回退：

| Scan | R2 - SP-RWCV Overall |
| ---: | ---: |
| 13 | +0.024694 |
| 48 | +0.014188 |
| 4 | +0.013787 |
| 118 | +0.009202 |
| 1 | +0.005729 |

当前解释：R2 对部分困难场景有效，但在一些简单或稳定场景有回退，所以平均提升不大。Adaptive R2 的目标就是让可靠性增强变成选择性激活，而不是全局作用。

## 原始 CSV

- [官方 plain baseline](data/official_plain_baseline_bs6_m015.csv)
- [官方 SP-RWCV](data/official_sp_rwcv_bs5_m015.csv)
- [官方 R2-MVSNet](data/official_r2_rafe_sprwcv_bs4_m015.csv)
- [本地 plain baseline](data/internal_plain_baseline_bs6_m015.csv)
- [本地 SP-RWCV](data/internal_sp_rwcv_bs5_m015.csv)
- [本地 R2-MVSNet](data/internal_r2_rafe_sprwcv_bs4_m015.csv)

## 当前待跟进实验

Adaptive R2：

```text
tag: 20260618_r2_adaptive_rafe_sprwcv_bs4_e16
flags: --use_rafe --use_adaptive_r2 --use_view_attention --view_attention_mode single_pass_reliability_weighted
batch_size: 4
epochs: 16
```

下一步：确认训练是否完成，然后跑 test、fusion、本地评估和官方 MATLAB 评估。
