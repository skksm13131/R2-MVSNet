# 04 第三创新点设计：FGDR

## 名称

建议名称：

```text
FGDR: Fusion-Guided Depth Refinement
```

中文名：

```text
面向点云融合的深度几何重构模块
```

这个创新点不再继续堆特征提取，而是把重点放在深度预测和点云融合之间的断层：网络训练时通常只追求深度图误差，但最终评价的是点云质量。深度图局部几何、边界形态、置信度和多视角一致性都会影响 fusion 后的 Accuracy、Completeness 和 Overall。

## 论文启发

参考文献：

```text
Quadruplex-depth based multi-view stereo network with wave-shaped depth cells and Epipolar Transformer
Engineering Applications of Artificial Intelligence, 2026
```

该文的关键观点：

- 深度预测和点云融合不能完全割裂。
- 即使深度图误差相近，不同局部深度几何也会导致不同点云质量。
- 通过预测多深度并约束局部几何，可以降低融合阶段的插值误差。
- 固定波浪状或鞍状深度几何有收益，但也可能引入额外预测偏差。

我们不直接照搬 Quadruplex-Depth。原因是它需要更重的多深度正则化，对当前 CasMVSNet 主线改动过大，且固定波浪几何未必适合所有区域。我们的方向是：利用 R2 已经产生的可靠性信息，只在需要的区域做融合友好的深度几何重构。

## 核心动机

当前 R2 主线主要解决的是 cost volume 之前和 cost volume 构建时的可靠性问题：

```text
RAFE: 哪些局部图像区域可靠
SP-RWCV: 哪些源视角可靠
Adaptive R2: 哪些区域需要增强，哪些区域应贴近 baseline
```

第三创新点要解决另一个问题：

```text
已经得到深度分布以后，如何让最终深度更适合点云融合？
```

也就是说，前两个创新点关注“匹配是否可靠”，第三个创新点关注“预测结果能否稳定融合成好点云”。

## 模块位置

FGDR 建议插在每个 stage 的 cost regularization 和 depth regression 之后，形成三阶段渐进式深度几何重构：

```text
prob_volume -> depth regression -> FGDR -> refined depth / geometry confidence -> fusion
```

三阶段分工：

1. stage1 输出粗深度和粗不确定性范围，帮助后续阶段避免过早收窄到错误深度。
2. stage2 输出中间深度和中间不确定性范围，继续修正 stage3 的采样中心。
3. stage3 输出最终深度、候选范围和融合风险，供最终点云融合使用。

需要注意：真正生成点云的 fusion 仍然只使用最后阶段输出；前两个阶段的作用是改善深度预测链路和下一阶段搜索范围。

## 模块输入

FGDR 输入尽量复用现有信息：

- 当前 stage 的回归深度 `D`
- 概率体置信度 `photometric_confidence`
- RAFE 输出的参考图可靠性 `ref_reliability`
- SP-RWCV / R2 产生的视角权重或视角可靠性 `view_weights`
- 当前 stage 的图像特征 `ref_feature`
- 可选：深度概率体局部统计，例如 entropy、top-k margin、局部 depth variance

这些输入共同描述：

- 当前深度是否可信。
- 当前区域是否低纹理或边界复杂。
- 多视角是否一致。
- 深度分布是否尖锐或多峰。

## 模块输出

FGDR 不只输出一个深度，而是输出一个可用于融合的局部深度几何表达：

```text
D_main: 主深度
D_near: 偏近候选深度
D_far: 偏远候选深度
G: geometry confidence / fusion gate
U: uncertainty / search radius
```

推荐实现方式：

```text
delta = softplus(delta_raw) * max_radius
D_near = D_main - delta
D_far = D_main + delta
```

其中 `delta` 不应全局放大，而应由 `G` 和不确定性控制。高置信区域 delta 小，困难区域 delta 大。

## 自适应几何选择

FGDR 的核心不是强行让所有像素波动，而是区域自适应：

### 高置信区域

行为：

```text
D_main 接近 baseline depth
delta 接近 0
fusion 主要使用 D_main
```

目的：

- 避免破坏简单区域。
- 保住 Accuracy。
- 减少 Adaptive R2 中出现的简单场景回退。

### 困难区域

包括：

- 低纹理区域
- 遮挡边界
- 反光区域
- 多视角权重分歧大区域
- 概率体多峰区域

行为：

```text
允许 D_near / D_far 提供上下界或候选几何
fusion 根据多视角一致性选择更可靠候选
```

目的：

- 降低融合阶段插值误差。
- 提高 Completeness。
- 减少错误深度在点云融合中扩散。

## 训练损失

FGDR 至少需要四类损失，组成一个完整创新点。

### 1. 主深度监督

保持原始深度学习目标：

```text
L_main = |D_main - D_gt|
```

作用：保证模块不是为了几何而牺牲基本深度准确性。

### 2. 上下界覆盖损失

困难区域中，希望 ground truth 落在 `D_near` 和 `D_far` 之间：

```text
L_cover = relu(D_near - D_gt) + relu(D_gt - D_far)
```

作用：让候选几何真的覆盖正确深度，而不是随意扩张。

### 3. 自适应半径约束

高置信区域不应产生大范围候选，困难区域才允许更大范围：

```text
L_radius = confidence * delta + difficult_mask * relu(min_radius - delta)
```

其中 `confidence` 可由 photometric confidence、RAFE reliability、view agreement 综合得到。

作用：

- 高置信区域贴近单深度。
- 困难区域保留必要几何弹性。

### 4. 多视角融合一致性损失

训练时用可微或近似可微的重投影一致性约束：

```text
L_reproj = weighted reprojection consistency(D_candidate, source views)
```

候选深度可从 `D_main / D_near / D_far` 中软选择：

```text
D_fusion = alpha_main * D_main + alpha_near * D_near + alpha_far * D_far
```

作用：让训练目标更接近测试阶段点云融合。

## 测试阶段融合改动

测试阶段需要和 FGDR 输出配套，否则训练出的候选几何用不上。

建议融合逻辑：

1. 对最后 stage 的每个参考像素生成 `D_main / D_near / D_far` 三个候选。
2. 将候选深度分别投影到源视角，计算几何一致性和置信度。
3. 高置信区域直接使用 `D_main`。
4. 困难区域选择多视角一致性最好的候选深度。
5. 多源视角融合时使用 reliability-weighted average，而不是简单平均。

这样第三创新点同时包含：

- 网络输出改动。
- 损失函数改动。
- 点云融合策略改动。
- 可解释的中间图：`G`、`U`、候选选择图、困难区域图。

## 与现有 R2 的关系

FGDR 不替代 R2，而是接在 R2 后面：

```text
Plain CasMVSNet
-> SP-RWCV
-> RAFE + SP-RWCV
-> Adaptive R2
-> FGDR
```

逻辑关系：

- RAFE 提供像素/区域可靠性。
- SP-RWCV 提供源视角可靠性。
- Adaptive R2 判断增强是否应该激活。
- FGDR 使用这些可靠性信息决定深度几何和融合策略。

这能把整篇论文主线串成一句话：

```text
从特征、视角、深度几何到点云融合的可靠性一致建模。
```

## 消融实验设计

至少需要以下 ablation：

1. Plain CasMVSNet
2. R2-MVSNet
3. Adaptive R2
4. R2 + FGDR depth head only
5. R2 + FGDR loss only
6. R2 + FGDR fusion only
7. R2 + full FGDR

如果时间允许，再做：

- 只 stage3 接 FGDR
- stage2 + stage3 接 FGDR
- stage1 + stage2 + stage3 接 FGDR
- 不使用 RAFE reliability
- 不使用 view weights
- 固定 delta vs adaptive delta
- 简单平均 fusion vs reliability-weighted fusion

## 可视化材料

FGDR 适合做可解释图：

- `G`: 几何融合门控图
- `U`: 深度不确定性/候选半径图
- `D_near / D_main / D_far` 可视化
- 候选深度选择图
- 高置信区域和困难区域对比
- 点云边界和低纹理区域局部放大图

这些图可以支撑论文叙事：模块不是盲目扰动深度，而是在 fusion 容易出错的地方自适应重构深度几何。

## 实施路线

### 阶段一：三阶段训练内版本

- 三个 stage 都接 FGDR。
- 输出 `delta`、`G`、`D_near`、`D_far`。
- 训练先只加 `L_main + L_cover + L_radius`。
- 前两阶段使用 FGDR 后的主深度继续指导下一阶段采样。
- 测试先不改 fusion，只看 depth 和 local eval 是否稳定。

目标：确认三阶段深度几何重构不会破坏 baseline/R2。

### 阶段二：融合联动版本

- 修改 `fusion-normal.py`，接收 FGDR 输出。
- 在困难区域启用候选深度选择。
- 引入 reliability-weighted fusion。

目标：观察 Completeness 是否提升，边界和低纹理点云是否更完整。

2026-06-28 已实现阶段二的保守候选融合 v1：

```text
D_main / D_near / D_far
        -> 分别与源图主深度做多视图一致性检查
        -> gate 允许且候选支持数至少比主深度多 1
        -> 选择候选，否则保持 D_main
```

该版本先只改变参考图候选，不做 near/far 的全组合匹配，以控制错误匹配和点云噪声。运行开关：

```bash
--use_fgdr \
--fuse_fgdr_candidates
```

其中 `--fuse_fgdr_candidates` 由 `scripts/test_fuse_eval.sh` 转换为 fusion 侧的 `--use_fgdr_candidates`，不会传给 `test.py`。关闭该开关时，原融合结果保持不变。

第一版本地结果：

```text
main-depth FGDR:      Acc=0.312170, Comp=0.262030, Overall=0.287100
FGDR candidate fuse: Acc=0.313151, Comp=0.259876, Overall=0.286514
```

候选融合以 `0.000981` 的 Accuracy 代价换取 `0.002154` 的 Completeness 改善，最终 Overall 改善 `0.000586`。这与设计预期一致：候选深度的主要作用不是继续提高单点精度，而是恢复主深度在多视图检查中被拒绝的有效点。

官方 MATLAB 结果：

```text
main-depth FGDR:      Acc=0.332150, Comp=0.280857, Overall=0.306503
FGDR candidate fuse: Acc=0.333778, Comp=0.277980, Overall=0.305879
original R2:          Acc=0.334543, Comp=0.277197, Overall=0.305870
```

候选融合在官方评估中将 main-depth FGDR 的 Overall 改善 `0.000624`，几乎完全追回 Completeness 损失；最终与原 R2 仅差 `0.000009`。第一版已证明“候选参与 fusion”有效，但尚未形成显著超过 R2 的主指标提升。

### 阶段二点五：Anchor-FGDR

第一版的问题是 FGDR refined main 先替换了原 R2 深度，导致 fusion 之前已经产生 Completeness 回退。Anchor-FGDR 改为：

```text
R2 base depth -> 保持为主深度和级联采样中心
              -> FGDR 生成 refined / near / far 候选
              -> fusion 仅在候选获得更多多视图支持时替换 base
```

训练损失：

```text
L = L_R2_depth + lambda_fgdr * (
    L_cover + 0.25 * L_candidate_center + 0.1 * L_high_conf_radius
)
```

该设计使第三模块只提供增量候选，不再先改变 R2 主路径。运行参数：

```bash
--use_fgdr \
--fgdr_anchor_base \
--fgdr_center_weight 0.25
```

### 阶段三：完整论文版本

- 加入多视角重投影一致性损失。
- 做完整 ablation。
- 输出可视化图。
- 决定是否扩展到 stage2。

目标：形成完整、独立、可写作的第三创新点。

## 风险与控制

主要风险：

- 候选深度范围过大，Accuracy 变差。
- fusion 逻辑过于激进，产生噪点。
- 训练损失过多，权重难调。
- stage 全接导致显存和不稳定性增加。

控制策略：

- 所有行为必须由 flag 控制，例如 `--use_fgdr`。
- 默认路径保持 plain CasMVSNet 可运行。
- 三阶段都有独立输出，但先不急着改 fusion。
- 先轻损失，后重投影损失。
- 先 local eval，再决定是否 official MATLAB。

## 初始判断

FGDR 比直接照搬 Quadruplex-Depth 更适合当前项目，因为它：

- 能形成单独的大创新点。
- 与 R2 的可靠性主线天然连接。
- 改动足够大，包括模型、损失和融合。
- 仍能保持 ablation 清晰。
- 可以解释 Adaptive R2 的问题：前端可靠性增强还不够，需要把可靠性延伸到最终深度几何和点云融合。
