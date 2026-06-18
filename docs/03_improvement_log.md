# 03 改进日志

这份文档记录模型为什么演进到现在这个样子。它不替代实验结果表，而是解释每个改进的动机、接入方式和当前判断。

## Plain CasMVSNet

Plain CasMVSNet 是对照路径：

```text
FeatureNet -> cascade depth sampling -> homography warping -> variance cost volume -> cost regularization -> depth regression
```

这个路径不能被破坏。任何新方法都要能和它做清楚的 ablation。

## SP-RWCV

全称：

```text
Single-Pass Reliability-Weighted Cost Volume
```

运行参数：

```bash
--use_view_attention \
--view_attention_mode single_pass_reliability_weighted
```

主要文件：

- `models/modules/view_attention.py`
- `models/cas_mvsnet.py`

动机：

早期 residual-fusion 方向有一定提升，但速度较慢，因为容易引入额外源图处理或重复 homography warping。SP-RWCV 只在 variance cost volume 构建时加入源图可靠性权重，尽量保持主流程接近 baseline。

核心做法：

- 每个源图只做一次 homography warping。
- 根据参考图和源图的一致性预测 source-view reliability score。
- 把 score 转成有界权重。
- 用加权均值和加权方差构建 cost volume。
- 后续仍走原来的 3D cost regularization 和 depth regression。

加权方差：

```text
weighted_mean = sum(w_i * F_i) / sum(w_i)
weighted_variance = sum(w_i * F_i^2) / sum(w_i) - weighted_mean^2
```

当前判断：SP-RWCV 相比 plain baseline 在官方 DTU 上有小幅、可复现提升，是后续 RAFE 和 Adaptive R2 的基础路径。

## RAFE

全称：

```text
Reliability-Aware Feature Extraction
```

运行参数：

```bash
--use_rafe
```

主要文件：

- `models/module.py`

动机：

只在 cost volume 阶段做 source-view reweighting 还不够。RAFE 尝试让 FeatureNet 显式感知局部结构和可靠性，使后续匹配使用的特征更知道哪些区域可信、哪些区域困难。

核心做法：

- FeatureNet 额外预测多尺度 reliability map：
  - `stage1_reliability`
  - `stage2_reliability`
  - `stage3_reliability`
- 结构先验包含灰度、梯度、梯度幅值、局部纹理方差、x/y 坐标。
- RAFE adapter 用结构先验做残差增强：

```text
feature_out = feature + residual_scale * gate * reliability_gate * prior_feature
```

与 SP-RWCV 的关系：

- RAFE 预测的源图 reliability 会跟随 source feature 一起 homography warping。
- SP-RWCV 在预测源图权重时使用参考图和源图 reliability。

## R2-MVSNet

组成：

```text
RAFE + SP-RWCV
```

运行参数：

```bash
--use_rafe \
--use_view_attention \
--view_attention_mode single_pass_reliability_weighted
```

当前结果：

- 官方 DTU Overall 从 baseline 的 `0.310124` 降到 `0.305870`。
- 相比 SP-RWCV 还有 `-0.000982` 的 Overall 改善。
- 收益主要来自 Completeness。

当前问题：

- 困难场景收益明显。
- 一些简单或稳定场景有回退。
- 平均提升真实可复现，但幅度被收益和回退互相抵消。

## Adaptive R2

运行参数：

```bash
--use_adaptive_r2
```

这个参数需要和 RAFE、SP-RWCV 一起使用。

动机：

R2 的问题不是完全无效，而是增强太“全局”。困难区域需要增强，简单高置信区域更应该贴近 baseline。因此 Adaptive R2 希望通过 difficulty gate 让可靠性增强选择性激活。

RAFE 修改：

```text
feature_out = feature + residual_scale * gate * reliability_gate * difficulty_gate * prior_feature
```

RAFE 的 `difficulty_gate` 综合：

- 从 feature / prior 统计中学习到的困难度。
- reliability uncertainty。
- prior feature energy。

SP-RWCV 修改：

```text
weight = 1 + residual_ratio * difficulty_gate * pixel_residual
```

SP-RWCV 的 `difficulty_gate` 综合：

- 源图 score magnitude。
- 上一阶段 confidence uncertainty。
- feature reliability uncertainty。

预期行为：

- 简单、高 confidence 区域更接近 baseline variance。
- 困难、低 reliability 区域保留更强的源图可靠性重加权。

当前训练 tag：

```text
20260618_r2_adaptive_rafe_sprwcv_bs4_e16
```

## 暂缓或废弃想法

旧实验中出现过 Direct-SCRF v2、RMFE、UGDR、CADR、RAHS、normal guidance、geometry guidance 等想法。

除非它们重新接入 `CascadeMVSNet` 并由明确 flag 控制，否则只把它们当作历史想法库，不当作当前有效模型行为。
