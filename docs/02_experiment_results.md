# 02 实验结果

DTU 的 Accuracy、Completeness、Overall 都是距离误差，越低越好。

论文或正式汇报优先使用官方 MATLAB 评估。本地 `matlab.py` 结果只作为快速观察和 sanity check。

## 官方 MATLAB 主消融

| 方法 | 输出 tag | Acc Mean | Comp Mean | Overall |
| --- | --- | ---: | ---: | ---: |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.334233 | 0.286015 | 0.310124 |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.334978 | 0.278727 | 0.306852 |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.334543 | 0.277197 | 0.305870 |
| R2-MVSNet Full | `20260630_r2_anchor_fgdr_candidate_fusion_m015_001` | **0.333268** | **0.267471** | **0.300370** |

官方 Overall delta：

```text
SP-RWCV vs baseline: -0.003272
R2-MVSNet vs baseline: -0.004254
R2-MVSNet vs SP-RWCV: -0.000982
Full model vs baseline: -0.009754
Full model vs SP-RWCV: -0.006482
Full model vs R2-MVSNet: -0.005500
```

说明：R2 官方评估复跑过一次，22 个 scan 的 CSV 结果到 6 位小数完全一致。

R2-MVSNet Full 官方结果取得当前最佳指标：`Acc=0.333268, Comp=0.267471, Overall=0.300370`。相对 RAFE + SP-RWCV，Accuracy 改善 `0.001275`、Completeness 改善 `0.009726`、Overall 改善 `0.005500`。这说明保留 R2 主深度作为级联锚点、只让 FGDR 提供增量候选，能够避免主深度退化，并把候选融合的收益稳定转化为覆盖率提升。

## 本地 Python 主消融

| 方法 | 输出 tag | Acc Mean | Comp Mean | Overall |
| --- | --- | ---: | ---: | ---: |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.313724 | 0.268163 | 0.290943 |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.314965 | 0.261310 | 0.288137 |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.314625 | 0.259278 | 0.286952 |
| R2-MVSNet Full | `20260630_r2_anchor_fgdr_candidate_fusion_m015_001` | **0.312612** | **0.249581** | **0.281097** |

本地 Overall delta：

```text
SP-RWCV vs baseline: -0.002806
R2-MVSNet vs baseline: -0.003991
R2-MVSNet vs SP-RWCV: -0.001185
Full model vs baseline: -0.009846
Full model vs SP-RWCV: -0.007040
Full model vs R2-MVSNet: -0.005855
```

R2-MVSNet Full 本地结果为 `Acc=0.312612, Comp=0.249581, Overall=0.281097`。相对 RAFE + SP-RWCV，Accuracy、Completeness、Overall 分别改善 `0.002013`、`0.009697`、`0.005855`，与官方结果的改善方向一致。

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
- [官方 Adaptive R2](data/official_r2_adaptive_rafe_sprwcv_bs4_m015.csv)
- [官方 R2-MVSNet Full](data/official_r2_anchor_fgdr_candidate_fusion_m015.csv)
- [本地 plain baseline](data/internal_plain_baseline_bs6_m015.csv)
- [本地 SP-RWCV](data/internal_sp_rwcv_bs5_m015.csv)
- [本地 R2-MVSNet](data/internal_r2_rafe_sprwcv_bs4_m015.csv)
- [本地 Adaptive R2](data/internal_r2_adaptive_rafe_sprwcv_bs4_m015.csv)
- [本地 R2-MVSNet Full](data/internal_r2_anchor_fgdr_candidate_fusion_m015.csv)

## Adaptive R2 实验状态

训练与本地评估已于 2026-06-26 完成：

```text
train tag: 20260618_r2_adaptive_rafe_sprwcv_bs4_e16
eval tag: 20260626_r2_adaptive_rafe_sprwcv_bs4_m015_001
checkpoint: model_000015.ckpt
flags: --use_rafe --use_adaptive_r2 --use_view_attention --view_attention_mode single_pass_reliability_weighted
batch_size: 4
epochs: 16
```

官方 MATLAB 评估已补充完成，结论与本地评估一致。下一步优先统计 Adaptive R2 difficulty gate 的分布，并针对 `scan48`、`scan33` 分析有效点覆盖率。

## 完整模型实验状态

```text
train tag: 20260628_r2_anchor_fgdr_rafe_sprwcv_bs4_e16
eval tag: 20260630_r2_anchor_fgdr_candidate_fusion_m015_001
checkpoint: model_000015.ckpt
flags: --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fgdr_anchor_base --fuse_fgdr_candidates
local result:    Acc=0.312612, Comp=0.249581, Overall=0.281097
official result: Acc=0.333268, Comp=0.267471, Overall=0.300370
```
