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

- 三个 stage 都接 FGDR；前两阶段改善后续采样，最后阶段服务最终深度和后续 fusion。
- 必须由 `--use_fgdr` 独立控制。
- 先做最小可跑版本，再改 fusion。
- 不破坏 plain CasMVSNet 和 R2 路径。

2026-06-28 实验进展：

- 已完成三阶段 FGDR 第一版接入。
- 训练/测试入口已加入 `--use_fgdr`。
- 服务器纯模型前向、FGDR 三阶段输出、loss 和 backward 已通过。
- 已完成 16 轮训练、测试、点云融合和本地评估：

```text
train tag: 20260626_r2_fgdr_rafe_sprwcv_bs4_e16_retry
eval tag: 20260628_r2_fgdr_rafe_sprwcv_bs4_m015_001
args: --epochs 16 --batch_size 4 --pin_m --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fgdr_loss_weight 0.05 --fgdr_radius_weight 0.1
local result: Acc=0.312170, Comp=0.262030, Overall=0.287100
```

结论：

- 相对原 R2-MVSNet，Accuracy 改善 `0.002455`，Completeness 回退 `0.002752`，Overall 回退 `0.000148`。
- 22 个测试场景中 11 个改善、11 个回退，第一版整体与 R2 主线持平。
- 当前 FGDR 只把 refined main depth 交给旧 fusion，尚未让 fusion 使用 near/far 候选。这可能是精度提升但覆盖率下降的主要原因之一。
- 下一轮优先做深度偏移约束和候选深度融合消融，不直接叠加 Adaptive R2。

2026-06-28 候选融合第一版：

- 服务器改动前备份：
  `/home/u104754251515/baseline/CasMVSNet20260604/backup_fgdr_candidate_fusion_20260628`
- `test.py` 在启用 FGDR 时额外保存：
  `depth_near`、`depth_far`、`geometry_gate`、`uncertainty`、`depth_delta`、`depth_base`。
- `fusion-normal.py` 新增独立开关 `--use_fgdr_candidates`。
- near/far 只有在 geometry gate 达标、候选通过几何筛选，且一致源图支持数至少比主深度多 1 时才替换主深度。
- `confidence > 0.99` 的高置信像素继续保留主深度。
- 默认不启用候选融合，原融合路径保持不变。
- `scan1` 端到端冒烟测试通过：49 组候选图和选择掩码均生成，候选切换比例约为 `0.02%–0.40%`。
- 关闭候选融合时，改动前后 `scan1` PLY 点数一致且 SHA256 完全相同。

正式测试：

```text
tag: 20260628_r2_fgdr_candidate_fusion_m015_001
flags: --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fuse_fgdr_candidates
local result: Acc=0.313151, Comp=0.259876, Overall=0.286514
```

结果：

- 相对 main-depth FGDR，Accuracy 回退 `0.000981`，Completeness 改善 `0.002154`，Overall 改善 `0.000586`。
- 22 个场景中 14 个改善、8 个回退。
- 相对原 R2-MVSNet，本地 Overall 改善 `0.000438`。
- 候选切换比例虽低，但集中在主深度多视图支持不足的位置，证明保守选择策略能够以很小改动恢复有效覆盖。
- 官方 MATLAB 结果为 `Acc=0.333778, Comp=0.277980, Overall=0.305879`。
- 相对 main-depth FGDR，官方 Overall 改善 `0.000624`；相对原 R2 仅差 `+0.000009`，基本持平。
- 下一步统计逐场景候选切换率与 Completeness 变化的相关性，并尝试在不继续损失 Accuracy 的情况下增加有效候选覆盖。

2026-06-28 Anchor-FGDR 完整训练：

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

2026-06-30 Anchor-FGDR 测试与评估：

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
- 相对第一版候选融合，官方 Overall 改善 `0.005509`；本地 Overall 改善 `0.005417`。
- 候选融合日志中的替换比例总体较低，以 refined candidate 为主，near/far 只在少量像素上替换，符合“保留 Base、证据更强才改动”的设计。
- 本地与官方评估方向一致，Anchor-FGDR 成为当前最佳方案，可作为第三创新点主版本。
- 官方原始评估目录：`docs/data/official_eval_20260630_r2_anchor_fgdr_candidate_fusion_m015_001_w8`。

## 暂缓或废弃想法

旧实验中出现过 Direct-SCRF v2、RMFE、UGDR、CADR、RAHS、normal guidance、geometry guidance 等想法。

除非它们重新接入 `CascadeMVSNet` 并由明确 flag 控制，否则只把它们当作历史想法库，不当作当前有效模型行为。
