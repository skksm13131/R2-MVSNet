# 01 工作习惯

这份文档记录我们继续做这个项目时默认遵守的习惯。它比具体实验更稳定，新的智能体接手后应该先按这里执行。

## 协作习惯

- 默认用中文沟通和记录。
- 先确认项目当前状态，再动代码或跑实验。
- 文档只记录能帮助接手、复现实验、解释模型演进的信息。
- 不把明文密码、私密路径说明、临时聊天内容写进 git。

## Baseline 规则

plain CasMVSNet 是所有改进的对照组。不开 reliability / attention flag 时，模型应该保持标准流程：

```text
FeatureNet -> cascade depth sampling -> homography warping -> variance cost volume -> cost regularization -> depth regression
```

要求：

- baseline 必须一直可运行。
- 新行为必须放在显式 flag 后面。
- 新模块要能独立 ablation。
- 不要直接修改默认路径来“偷偷启用”新方法。

## 模型改动规则

当前主要改动面：

- `models/cas_mvsnet.py`
- `models/modules/view_attention.py`
- `models/module.py`
- `train.py`
- `test.py`

习惯：

- `train.py` 和 `test.py` 的参数要同步。
- 一次只做一个清楚的结构改动。
- 长训练前先做 import、模型实例化、小 batch forward 检查。
- 旧实验模块只有接入 `CascadeMVSNet` 并有 flag 控制时，才算当前有效模型行为。
- 困难场景提升、简单场景回退时，优先考虑 adaptive gate，而不是全局增强模块强度。

当前安全 flag：

```bash
--use_view_attention
--view_attention_mode single_pass_reliability_weighted
--use_rafe
--use_adaptive_r2
```

## 实验记录规则

实验 tag 尽量包含日期、模型、batch size、epoch、checkpoint：

```text
20260618_r2_adaptive_rafe_sprwcv_bs4_e16
20260618_r2_rafe_sprwcv_bs4_m015_001
```

常用约定：

- `bs4` 表示 batch size 4。
- `e16` 表示训练 16 个 epoch。
- `m015` 表示使用 `model_000015.ckpt`。
- `_001` 表示同一实验的第一个测试输出版本。

每次重要实验至少记录：

- 训练 tag 和 checkpoint。
- 使用的关键 flags。
- 官方 MATLAB 指标。
- 本地 `matlab.py` 指标。
- 相比 baseline / 上一版本的 delta。
- 单场景明显收益和明显回退。

## 评估规则

- 论文主对比以官方 MATLAB 评估为准。
- 本地 `matlab.py` 结果只用于快速观察和 sanity check。
- Accuracy、Completeness、Overall 都是距离误差，越低越好。
- 官方评估和本地评估不要混在同一张结论表里。

## Git 规则

可以提交：

- 源代码。
- 小型文档。
- 用于汇总结果的小型 CSV。
- 可复现实验的脚本。

不要提交：

- `checkpoints/`
- 点云文件。
- TensorBoard events。
- 完整测试输出。
- 完整官方/本地评估输出目录。
- `__pycache__/`
