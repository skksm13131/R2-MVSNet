# 评估结果

DTU 的 Accuracy、Completeness、Overall 都是距离误差：**越低越好**。

下面分开记录本地 Python 评估脚本 `matlab.py` 和官方 MATLAB 评估机结果。主要论文对比应以官方 MATLAB 结果为准。

## 官方 MATLAB 评估

| 方法 | Checkpoint / 输出 tag | Acc Mean | Comp Mean | Overall | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.334233 | 0.286015 | 0.310124 | baseline reference |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.334978 | 0.278727 | 0.306852 | Overall 低于 baseline |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.334543 | 0.277197 | 0.305870 | 当前官方 Overall 最好 |

官方对比：

```text
SP-RWCV 相比 plain baseline，Overall 降低 0.003272。
相对变化：-1.06% Overall error。

R2-MVSNet 相比 plain baseline，Overall 降低 0.004254。
相对变化：-1.37% Overall error。

R2-MVSNet 相比 SP-RWCV，Overall 降低 0.000982。
相对变化：-0.32% Overall error。
```

## 本地 Python 评估

| 方法 | 输出 tag | Acc Mean | Comp Mean | Overall | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| Plain CasMVSNet baseline | `20260616_plain_baseline_bs6_m015_001` | 0.313724 | 0.268163 | 0.290943 | local `matlab.py` |
| SP-RWCV | `20260615_sp_rwcv_bs5_m015_001` | 0.314965 | 0.261310 | 0.288137 | local `matlab.py` |
| R2-MVSNet (RAFE + SP-RWCV) | `20260618_r2_rafe_sprwcv_bs4_m015_001` | 0.314625 | 0.259278 | 0.286952 | local `matlab.py` |

本地对比：

```text
SP-RWCV 相比 plain baseline，Overall 降低 0.002806。
相对变化：-0.96% Overall error。

R2-MVSNet 相比 plain baseline，Overall 降低 0.003991。
相对变化：-1.37% Overall error。

R2-MVSNet 相比 SP-RWCV，Overall 降低 0.001185。
相对变化：-0.41% Overall error。
```

## 原始 CSV 文件

- [官方 plain baseline](results/official_plain_baseline_bs6_m015.csv)
- [官方 SP-RWCV](results/official_sp_rwcv_bs5_m015.csv)
- [官方 R2-MVSNet](results/official_r2_rafe_sprwcv_bs4_m015.csv)
- [本地 plain baseline](results/internal_plain_baseline_bs6_m015.csv)
- [本地 SP-RWCV](results/internal_sp_rwcv_bs5_m015.csv)
- [本地 R2-MVSNet](results/internal_r2_rafe_sprwcv_bs4_m015.csv)

## 当前解释

SP-RWCV 相比 plain baseline 有小幅但稳定的 Overall error 降低。RAFE 加到 SP-RWCV 之后又带来一点额外降低，主要来自 Completeness。单场景分析显示：R2 在部分困难场景收益明显，但在一些简单或稳定场景上回退，因此平均提升较小。
