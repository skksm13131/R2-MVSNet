# 02 实验结果

DTU 的 Accuracy、Completeness、Overall 都是距离误差，越低越好。

论文或正式汇报优先使用官方 MATLAB 评估。本地 `matlab.py` 结果只作为快速观察和 sanity check。

## 官方 MATLAB 评估

| 方法 | 输出 tag | Acc Mean | Comp Mean | Overall |
| --- | --- | ---: | ---: | ---: |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.334233 | 0.286015 | 0.310124 |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.334978 | 0.278727 | 0.306852 |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.334543 | 0.277197 | 0.305870 |
| Adaptive R2 | `20260626_r2_adaptive_rafe_sprwcv_bs4_m015_001` | 0.330682 | 0.283282 | 0.306982 |
| R2-MVSNet + FGDR | `20260628_r2_fgdr_rafe_sprwcv_bs4_m015_001` | 0.332150 | 0.280857 | 0.306503 |
| R2-MVSNet + FGDR candidate fusion | `20260628_r2_fgdr_candidate_fusion_m015_001` | 0.333778 | 0.277980 | 0.305879 |
| R2-MVSNet + Anchor-FGDR candidate fusion | `20260630_r2_anchor_fgdr_candidate_fusion_m015_001` | **0.333268** | **0.267471** | **0.300370** |

官方 Overall delta：

```text
SP-RWCV vs baseline: -0.003272
R2-MVSNet vs baseline: -0.004254
R2-MVSNet vs SP-RWCV: -0.000982
Adaptive R2 vs baseline: -0.003142
Adaptive R2 vs SP-RWCV: +0.000130
Adaptive R2 vs R2-MVSNet: +0.001112
FGDR vs baseline: -0.003621
FGDR vs SP-RWCV: -0.000349
FGDR vs R2-MVSNet: +0.000633
FGDR candidate fusion vs baseline: -0.004245
FGDR candidate fusion vs SP-RWCV: -0.000973
FGDR candidate fusion vs R2-MVSNet: +0.000009
FGDR candidate fusion vs main-depth FGDR: -0.000624
Anchor-FGDR vs baseline: -0.009754
Anchor-FGDR vs SP-RWCV: -0.006482
Anchor-FGDR vs R2-MVSNet: -0.005500
Anchor-FGDR vs FGDR candidate fusion: -0.005509
```

说明：R2 官方评估复跑过一次，22 个 scan 的 CSV 结果到 6 位小数完全一致。

Adaptive R2 官方结果与本地结果结论一致：Accuracy 优于原 R2，但 Completeness 明显回退，导致 Overall 变差。逐场景上 12/22 改善、10/22 回退，但 `scan48`（Overall `+0.028742`）和 `scan33`（`+0.023321`）的大幅回退抵消了多数小幅收益。

main-depth FGDR 官方结果同样表现为 Accuracy 改善、Completeness 回退：相对原 R2，Accuracy 改善 `0.002393`，Completeness 回退 `0.003660`，Overall 回退 `0.000633`。这进一步支持候选融合应优先恢复覆盖率。

候选融合将 main-depth FGDR 的官方 Completeness 改善 `0.002877`，代价是 Accuracy 回退 `0.001628`，最终 Overall 改善 `0.000624`。与原 R2 相比，Accuracy 改善 `0.000765`，Completeness 回退 `0.000783`，Overall 仅差 `+0.000009`，在六位小数口径下几乎持平。逐场景相对 main-depth FGDR 为 11/22 改善、11/22 回退。

Anchor-FGDR 官方结果取得当前最佳指标：`Acc=0.333268, Comp=0.267471, Overall=0.300370`。相对原 R2，Accuracy 改善 `0.001275`、Completeness 改善 `0.009726`、Overall 改善 `0.005500`；相对第一版候选融合，Overall 改善 `0.005509`。这说明保留 R2 主深度作为级联锚点、只让 FGDR 提供增量候选，能够避免主深度先行退化，并把候选融合的收益稳定转化为覆盖率提升。

## 本地 Python 评估

| 方法 | 输出 tag | Acc Mean | Comp Mean | Overall |
| --- | --- | ---: | ---: | ---: |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.313724 | 0.268163 | 0.290943 |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.314965 | 0.261310 | 0.288137 |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.314625 | 0.259278 | 0.286952 |
| Adaptive R2 | `20260626_r2_adaptive_rafe_sprwcv_bs4_m015_001` | 0.310944 | 0.265767 | 0.288355 |
| R2-MVSNet + FGDR | `20260628_r2_fgdr_rafe_sprwcv_bs4_m015_001` | 0.312170 | 0.262030 | 0.287100 |
| R2-MVSNet + FGDR candidate fusion | `20260628_r2_fgdr_candidate_fusion_m015_001` | 0.313151 | 0.259876 | 0.286514 |
| R2-MVSNet + Anchor-FGDR candidate fusion | `20260630_r2_anchor_fgdr_candidate_fusion_m015_001` | **0.312612** | **0.249581** | **0.281097** |

本地 Overall delta：

```text
SP-RWCV vs baseline: -0.002806
R2-MVSNet vs baseline: -0.003991
R2-MVSNet vs SP-RWCV: -0.001185
Adaptive R2 vs baseline: -0.002588
Adaptive R2 vs SP-RWCV: +0.000218
Adaptive R2 vs R2-MVSNet: +0.001403
FGDR vs baseline: -0.003843
FGDR vs SP-RWCV: -0.001037
FGDR vs R2-MVSNet: +0.000148
FGDR candidate fusion vs baseline: -0.004429
FGDR candidate fusion vs SP-RWCV: -0.001623
FGDR candidate fusion vs R2-MVSNet: -0.000438
FGDR candidate fusion vs main-depth FGDR: -0.000586
Anchor-FGDR vs baseline: -0.009846
Anchor-FGDR vs SP-RWCV: -0.007040
Anchor-FGDR vs R2-MVSNet: -0.005855
Anchor-FGDR vs FGDR candidate fusion: -0.005417
```

Adaptive R2 保留了相对 plain baseline 的提升，但没有超过 SP-RWCV 和原 R2-MVSNet。difficulty gate 降低了 Accuracy，却使 Completeness 明显回退，未达到“保留困难区域收益、减少简单场景回退”的预期。当前不安排官方 MATLAB 评估，后续应先分析 gate 的激活分布或缩小其作用强度。

FGDR 第一版本地 Overall 与原 R2-MVSNet 基本持平（差 `+0.000148`）。其中 Accuracy 改善 `-0.002455`，但 Completeness 回退 `+0.002752`，说明深度几何重构提高了已重建点的精度，却损失了部分有效覆盖。22 个场景中 11 个改善、11 个回退。下一步应优先约束 FGDR 的深度偏移或让融合阶段使用 near/far 候选，而不是直接增加模块强度。

候选融合验证了上述判断。相对只使用主深度的 FGDR，Accuracy 回退 `+0.000981`，Completeness 改善 `-0.002154`，Overall 改善 `-0.000586`；22 个场景中 14 个改善、8 个回退。候选融合最终以 `0.286514` 超过原 R2-MVSNet 的 `0.286952`，说明 near/far 候选确实能够恢复部分被主深度融合丢弃的覆盖。

Anchor-FGDR 本地结果为 `Acc=0.312612, Comp=0.249581, Overall=0.281097`。相对原 R2，Accuracy、Completeness、Overall 分别改善 `0.002013`、`0.009697`、`0.005855`，与官方结果的改善方向一致。当前结果支持将 Anchor-FGDR 作为第三创新点的正式主方案。

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
- [官方 R2-MVSNet + FGDR](data/official_r2_fgdr_rafe_sprwcv_bs4_m015.csv)
- [官方 R2-MVSNet + FGDR candidate fusion](data/official_r2_fgdr_candidate_fusion_m015.csv)
- [本地 plain baseline](data/internal_plain_baseline_bs6_m015.csv)
- [本地 SP-RWCV](data/internal_sp_rwcv_bs5_m015.csv)
- [本地 R2-MVSNet](data/internal_r2_rafe_sprwcv_bs4_m015.csv)
- [本地 Adaptive R2](data/internal_r2_adaptive_rafe_sprwcv_bs4_m015.csv)
- [本地 R2-MVSNet + FGDR](data/internal_r2_fgdr_rafe_sprwcv_bs4_m015.csv)
- [本地 R2-MVSNet + FGDR candidate fusion](data/internal_r2_fgdr_candidate_fusion_m015.csv)

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

## FGDR 第一版实验状态

训练、测试、点云融合和本地评估已于 2026-06-28 完成：

```text
train tag: 20260626_r2_fgdr_rafe_sprwcv_bs4_e16_retry
eval tag: 20260628_r2_fgdr_rafe_sprwcv_bs4_m015_001
checkpoint: model_000015.ckpt
flags: --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fgdr_max_radius_factor 2.0
batch_size: 4
epochs: 16
local overall: 0.287100
```

下一步：先分析 Completeness 回退来源并做 FGDR 小范围消融；第一版暂不进入官方 MATLAB 评估。

## FGDR 候选融合实验状态

候选融合、本地点云评估已于 2026-06-28 完成：

```text
eval tag: 20260628_r2_fgdr_candidate_fusion_m015_001
checkpoint: checkpoints/20260626_r2_fgdr_rafe_sprwcv_bs4_e16_retry/model_000015.ckpt
flags: --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fuse_fgdr_candidates
local result: Acc=0.313151, Comp=0.259876, Overall=0.286514
```

该结果已超过原 R2 主线，本地结论值得进入官方 MATLAB 评估。

官方候选融合评估已完成：

```text
official tag: 20260628_r2_fgdr_candidate_fusion_m015_001_w8
official result: Acc=0.333778, Comp=0.277980, Overall=0.305879
```
