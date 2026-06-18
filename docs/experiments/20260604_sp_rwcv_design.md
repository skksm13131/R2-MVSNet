# SP-RWCV 设计说明

SP-RWCV 全称：

```text
Single-Pass Reliability-Weighted Cost Volume
```

## 动机

早期 residual-fusion 方向有一定提升，但速度较慢，因为需要额外的源图处理或重复 homography warping。SP-RWCV 保持 cascade MVS 主流程接近 baseline，只在 variance cost volume 构建时加入源图可靠性权重。

## 运行参数

```bash
--use_view_attention \
--view_attention_mode single_pass_reliability_weighted
```

## 单阶段流程

每个 cascade stage 内：

1. 从参考图特征构建 reference volume。
2. 每个源图特征只做一次 homography warping。
3. 根据参考图和源图的一致性预测源图 reliability score。
4. 将 score 转成有界可靠性权重。
5. 累积加权 feature sum 和 weighted squared sum。
6. 得到 weighted variance cost volume。
7. 后续仍走正常 3D cost regularization 和 depth regression。

## 权重公式

对源图 `i`：

```text
w_i = 1 + residual_ratio * tanh(score_i / temperature)
```

实现中会 clamp 最终权重，保证训练稳定。

加权方差：

```text
mean = sum(w_i * F_i) / sum(w_i)
variance = sum(w_i * F_i^2) / sum(w_i) - mean^2
```

参考图权重保持为 `1`。

## 和 SCRF/RRF 的区别

- SP-RWCV 只做一次源图 warping。
- 它直接修改 variance 统计，不额外开一个 residual cost-volume 分支。
- 设计目标是更快、更容易 ablation。

## 实现文件

- `models/modules/view_attention.py`
  - `SinglePassReliabilityWeightedViewAttention`
- `models/cas_mvsnet.py`
  - `DepthNet.forward()` 中的 `uses_single_pass_weighted_variance` 分支
- `train.py` 和 `test.py`
  - 暴露 `--view_attention_mode single_pass_reliability_weighted`

## 当前状态

SP-RWCV 相比 plain baseline 在官方 DTU 上有小幅、可复现提升。RAFE 和 Adaptive R2 都是在这个路径上继续发展。
