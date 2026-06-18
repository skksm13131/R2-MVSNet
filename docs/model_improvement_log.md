# 模型改进日志

## Baseline

Plain CasMVSNet 是对照路径：

```text
FeatureNet -> cascade depth sampling -> homography warping -> variance cost volume -> cost regularization -> depth regression
```

无 attention、无 RAFE 的路径必须一直可运行。

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

核心思想：

- 在 variance cost volume 构建时，不再让所有源图等权贡献。
- 每个源图根据和参考图的匹配可靠性得到一个有界权重。
- 只做一次 homography warping，避免 residual-fusion 类方法的二次 warping 开销。
- FeatureNet、CostRegNet、depth regression、fusion 等部分尽量保持接近 baseline。

加权方差：

```text
weighted_mean = sum(w_i * F_i) / sum(w_i)
weighted_variance = sum(w_i * F_i^2) / sum(w_i) - weighted_mean^2
```

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

核心思想：

- FeatureNet 额外预测多尺度可靠性图：
  - `stage1_reliability`
  - `stage2_reliability`
  - `stage3_reliability`
- 结构先验包含灰度、梯度、梯度幅值、局部纹理方差、x/y 坐标。
- RAFE adapter 使用结构先验做残差增强：

```text
feature_out = feature + residual_scale * gate * reliability_gate * prior_feature
```

RAFE 和 SP-RWCV 的联动：

- RAFE 预测的源图 reliability 会随源图 feature 一起 homography warping。
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

官方 DTU 单场景结果观察：

- scan 29、33、75 等困难场景有明显收益。
- 一些简单或稳定场景有小幅回退。
- 平均提升真实可复现，但幅度较小，因为收益和回退互相抵消。

## Adaptive R2

运行参数：

```bash
--use_adaptive_r2
```

该参数需要和 RAFE、SP-RWCV 一起使用。

目标：

- 保留困难场景收益。
- 通过自适应门控减少简单场景回退。
- 让可靠性模块不要全局强行介入，而是更多作用在困难区域。

RAFE 修改：

```text
feature_out = feature + residual_scale * gate * reliability_gate * difficulty_gate * prior_feature
```

RAFE 的 `difficulty_gate` 综合：

- 从 feature/prior 统计中学习到的困难度。
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

当前 Adaptive R2 训练 tag：

```text
20260618_r2_adaptive_rafe_sprwcv_bs4_e16
```
