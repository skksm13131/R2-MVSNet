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

当前判断：

Adaptive R2 的本地与官方评估均弱于原 R2-MVSNet。官方 Overall 为 `0.306982`，比 R2 的 `0.305870` 回退 `0.001112`。22 个场景中 12 个改善、10 个回退，但 `scan48` 和 `scan33` 的 Completeness 大幅恶化，抵消了其余场景的小幅收益。这说明仅在特征和 cost volume 阶段做自适应可靠性增强还不够稳定。下一步不继续在特征提取上堆模块，而是转向深度预测和点云融合之间的协同。

## FGDR 设计方向

全称：

```text
Fusion-Guided Depth Refinement
```

中文名：

```text
面向点云融合的深度几何重构模块
```

设计文档：

- `docs/04_third_innovation_fgdr.md`

动机：

MVS 训练通常优化深度图误差，但最终评价的是点云质量。深度图局部几何、置信度、多视角一致性和融合策略会共同影响最终点云。参考 Quadruplex-Depth 论文的 fusion-aware 思路，FGDR 不直接照搬固定四深度波浪几何，而是利用 R2 已有的可靠性信息，自适应决定哪些区域保持单深度，哪些困难区域生成候选深度几何供 fusion 使用。

预期接入位置：

```text
prob_volume -> depth regression -> FGDR -> refined depth / geometry confidence -> fusion
```

预期输出：

```text
D_main, D_near, D_far, geometry confidence, uncertainty radius
```

当前原则：

- 三个 stage 都训练 FGDR 候选，但原 R2 base depth 始终负责后续采样。
- 最后阶段的候选参与最终 fusion。
- 必须由 `--use_fgdr` 独立控制。
- 不破坏 plain CasMVSNet 和 R2 路径。

2026-06-28 R2-MVSNet Full 完整训练：

- 保留原 R2 深度作为三个 stage 的主输出和下一阶段采样中心。
- FGDR 不再覆盖主深度，只训练 `refined main / near / far` 候选。
- 新增 `--fgdr_anchor_base` 独立开关。
- 候选中心增加 Smooth L1 监督，权重由 `--fgdr_center_weight 0.25` 控制。
- 服务器前向/损失/反向检查通过：base 与主输出完全一致，候选 residual head 梯度非零。
- 改动前服务器备份：`backup_anchor_fgdr_20260628`。

正式训练：

```text
tag: 20260628_r2_anchor_fgdr_rafe_sprwcv_bs4_e16
args: --epochs 16 --batch_size 4 --pin_m --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fgdr_anchor_base --fgdr_max_radius_factor 2.0 --fgdr_loss_weight 0.05 --fgdr_radius_weight 0.1 --fgdr_center_weight 0.25
status: completed, model_000015.ckpt
```

2026-06-30 R2-MVSNet Full 测试与评估：

```text
test/fusion tag: 20260630_r2_anchor_fgdr_candidate_fusion_m015_001
checkpoint: checkpoints/20260628_r2_anchor_fgdr_rafe_sprwcv_bs4_e16/model_000015.ckpt
flags: --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fgdr_anchor_base --fuse_fgdr_candidates
local:    Acc=0.312612, Comp=0.249581, Overall=0.281097
official: Acc=0.333268, Comp=0.267471, Overall=0.300370
```

结果：

- 训练、22 个 scan 测试、候选融合、本地评估和官方 MATLAB 评估均已完成。
- 相对原 R2，官方 Accuracy 改善 `0.001275`，Completeness 改善 `0.009726`，Overall 改善 `0.005500`。
- 候选融合日志中的替换比例总体较低，以 refined candidate 为主，near/far 只在少量像素上替换，符合“保留 Base、证据更强才改动”的设计。
- 本地与官方评估方向一致，该配置正式定义为 `R2-MVSNet Full`。
- 官方结果：`docs/data/official_r2_anchor_fgdr_candidate_fusion_m015.csv`。

2026-07-05 RAFE-only 消融：

```text
train tag: 20260630_rafe_only_bs3_e16_val
test/fusion tag: 20260705_rafe_only_bs3_m015_001
flags: --use_rafe
local: Acc=0.311429, Comp=0.261232, Overall=0.286330
```

- 使用独立训练的 RAFE-only checkpoint，不是从组合模型临时关闭 SP-RWCV。
- 测试时 `view_attention=False, rafe=True, fgdr=False`，使用普通点云融合。
- 相对 plain baseline，本地 Overall 改善 `0.004613`，17/22 个场景改善。
- 相对 RAFE + SP-RWCV，本地 Overall 改善 `0.000622`，12/22 个场景改善。
- 官方结果：`Acc=0.331468, Comp=0.279875, Overall=0.305671`。

2026-07-05 RAFE + SP-RWCV epoch 14 复测：

```text
train tag: 20260616_r2_rafe_sprwcv_bs4_e16
checkpoint: model_000014.ckpt
test/fusion tag: 20260705_r2_rafe_sprwcv_ckpt14_m015_001
local: Acc=0.316204, Comp=0.257473, Overall=0.286838
```

- 相对 epoch 15，Completeness 改善 `0.001805`，Accuracy 回退 `0.001579`，Overall 改善 `0.000114`。
- 相对 RAFE-only，本地 Overall 回退 `0.000508`。
- 官方评估机仍无法连接，官方 MATLAB 结果待补。

2026-07-06 RAFE + Anchor-FGDR candidate fusion 消融：

```text
train tag: 20260705_rafe_anchor_fgdr_bs5_e16
checkpoint: model_000015.ckpt
test/fusion tag: 20260706_rafe_anchor_fgdr_candidate_fusion_m015_001
flags: --use_rafe --use_fgdr --fgdr_anchor_base --fuse_fgdr_candidates
disabled: SP-RWCV
local: Acc=0.313301, Comp=0.256046, Overall=0.284673
```

- 完成 22 个 scan 测试、FGDR 候选融合和本地评估。
- 相对 RAFE-only，Accuracy 回退 `0.001872`，Completeness 改善 `0.005186`，Overall 改善 `0.001657`。
- 相对 RAFE + SP-RWCV，Overall 改善 `0.002279`。
- 相对 R2-MVSNet Full，Overall 仍回退 `0.003576`，主要差距来自 Completeness。
- 候选融合确实发生，单视图日志中的 FGDR switch ratio 通常处于低百分比区间，符合保守候选替换设计。
- 原始结果：`docs/data/internal_rafe_anchor_fgdr_candidate_fusion_m015.csv`。

## Decoupled SP-RWCV

2026-07-06 针对 RAFE 与 SP-RWCV 组合略弱于 RAFE-only 的现象，增加可靠性解耦模式：

```text
--use_rafe
--use_view_attention
--view_attention_mode decoupled_reliability_weighted
```

原 SP-RWCV 在启用 RAFE 时，源图可靠性既作为 `ScoreNet` 输入，又在
`score_to_weight` 中乘到权重残差上。后一次乘法会压缩低可靠源图的权重调整幅度，
使本应明显下调的源图更接近等权聚合。

新模式只解除输出端的重复门控：

- 参考图和源图可靠性仍进入 `ScoreNet`，继续参与可靠性判断。
- 上一阶段光度置信度门控保持不变，用于抑制不稳定的跨阶段调整。
- 权重范围、残差初始化和加权方差公式保持不变。
- 原 `single_pass_reliability_weighted` 行为完全保留，便于严格消融与回滚。

该模式不新增可训练参数，checkpoint 结构保持兼容，但应独立训练后再比较，不能只在
旧 checkpoint 测试时切换模式。

2026-07-06 修改后完整模型训练：

```text
train tag: 20260706_r2_decoupled_sprwcv_anchor_fgdr_bs4_e16
flags: --use_rafe --use_view_attention --view_attention_mode decoupled_reliability_weighted --use_fgdr --fgdr_anchor_base
batch_size: 4
epochs: 16
status: running
```

- 训练机部署前已备份 `models/cas_mvsnet.py`、`models/modules/view_attention.py`、`train.py` 和 `test.py`。
- 新旧模式的 `state_dict` 参数键完全一致，旧 checkpoint 可 `strict=True` 加载。
- 训练启动后模型配置打印正确，GPU 利用率正常，未出现 OOM。

## 暂缓或废弃想法

旧实验中出现过 Direct-SCRF v2、RMFE、UGDR、CADR、RAHS、normal guidance、geometry guidance 等想法。

除非它们重新接入 `CascadeMVSNet` 并由明确 flag 控制，否则只把它们当作历史想法库，不当作当前有效模型行为。
