# 00 项目交接

这份文档给新智能体快速接手项目用。先读这里，再读工作习惯、实验结果和改进日志。

## 当前项目状态

本项目基于 CasMVSNet，当前研究方向是把可靠性建模加入多视图立体匹配流程。

当前主线：

```text
Plain CasMVSNet -> SP-RWCV -> RAFE + SP-RWCV -> Adaptive R2
```

核心要求：plain CasMVSNet baseline 必须一直可运行，所有新模块都要能通过 flag 单独开关和做 ablation。

## 仓库与服务器

GitHub 仓库：

```text
skksm13131/R2-MVSNet
```

训练机：

```bash
ssh -p 21785 u104754251515@10.91.28.4
```

官方评估机：

```bash
ssh -p 23466 root@10.91.28.4
```

主要训练目录：

```text
/home/u104754251515/baseline/CasMVSNet20260604
```

旧参考目录：

```text
/home/u104754251515/baseline/CasMVSNet
```

不要把明文密码写进仓库。

## 当前实验状态

Adaptive R2 已在 2026-06-26 完成训练、测试、点云融合和本地评估：

```text
train tag: 20260618_r2_adaptive_rafe_sprwcv_bs4_e16
eval tag: 20260626_r2_adaptive_rafe_sprwcv_bs4_m015_001
repo: /home/u104754251515/baseline/CasMVSNet20260604
checkpoint: checkpoints/20260618_r2_adaptive_rafe_sprwcv_bs4_e16/model_000015.ckpt
local overall: 0.288355
official overall: 0.306982
```

该结果弱于原 R2-MVSNet（本地 `0.286952`、官方 `0.305870`）。本地与官方评估都表现为 Accuracy 改善、Completeness 回退，说明 Adaptive difficulty gate 会在少数场景损失有效覆盖。训练机上已启动低强度 GPU 保活 demo，接手后先确认其进程和心跳日志是否仍在。

下一条研究线已转向第三创新点：`FGDR: Fusion-Guided Depth Refinement`。该方向不继续堆特征提取，而是把可靠性信息延伸到深度几何重构和点云融合阶段。设计文档见 `docs/04_third_innovation_fgdr.md`。

FGDR 第一版已于 2026-06-28 完成训练、测试、点云融合和本地评估：

```text
train tag: 20260626_r2_fgdr_rafe_sprwcv_bs4_e16_retry
eval tag: 20260628_r2_fgdr_rafe_sprwcv_bs4_m015_001
repo: /home/u104754251515/baseline/CasMVSNet20260604
checkpoint: checkpoints/20260626_r2_fgdr_rafe_sprwcv_bs4_e16_retry/model_000015.ckpt
local accuracy: 0.312170
local completeness: 0.262030
local overall: 0.287100
gpu keepalive pid file: outputs_retest/20260628_r2_fgdr_rafe_sprwcv_bs4_m015_001/gpu_keepalive.pid
```

训练使用 R2 主线加 `--use_fgdr`，未叠加 Adaptive R2，最终以 `batch_size=4` 完成。main-depth FGDR 官方结果为 `0.306503`，相对原 R2 的 `0.305870` 回退 `+0.000633`，表现为准确性改善但完整性回退。低强度 GPU 保活已恢复，接手后先检查 PID 和心跳日志。

2026-06-28 已完成 FGDR 候选融合第一版本地评估：

```text
eval tag: 20260628_r2_fgdr_candidate_fusion_m015_001
local accuracy: 0.313151
local completeness: 0.259876
local overall: 0.286514
```

候选融合相对 main-depth FGDR 改善 `0.000586`，并超过原 R2 的本地 `0.286952`。代码改动前备份位于 `backup_fgdr_candidate_fusion_20260628`，默认融合路径经 SHA256 回归验证保持不变。

候选融合官方 MATLAB 评估已完成：

```text
official tag: 20260628_r2_fgdr_candidate_fusion_m015_001_w8
eval machine dir: /root/official_eval_20260628_r2_fgdr_candidate_fusion_m015_001_w8
official accuracy: 0.333778
official completeness: 0.277980
official overall: 0.305879
```

该结果相对 main-depth FGDR 的 `0.306503` 改善 `0.000624`，与原 R2 的 `0.305870` 仅差 `+0.000009`。候选融合方向有效，但当前尚不能宣称超过原 R2。

2026-06-28 已启动 Anchor-FGDR 从头完整训练：

```text
train tag: 20260628_r2_anchor_fgdr_rafe_sprwcv_bs4_e16
checkpoint dir: checkpoints/20260628_r2_anchor_fgdr_rafe_sprwcv_bs4_e16
flags: --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fgdr_anchor_base
batch_size: 4
epochs: 16
status: completed, final checkpoint model_000015.ckpt
```

Anchor-FGDR 保持原 R2 depth 为主输出，只训练 refined/near/far 候选。改动前备份位于 `backup_anchor_fgdr_20260628`。

2026-06-30 已完成 Anchor-FGDR 测试、候选融合、本地评估和官方 MATLAB 评估：

```text
eval tag: 20260630_r2_anchor_fgdr_candidate_fusion_m015_001
local:    Acc=0.312612, Comp=0.249581, Overall=0.281097
official: Acc=0.333268, Comp=0.267471, Overall=0.300370
official eval dir: /root/official_eval_20260630_r2_anchor_fgdr_candidate_fusion_m015_001_w8
```

相对原 R2，官方 Overall 改善 `0.005500`，Accuracy 和 Completeness 同时改善。Anchor-FGDR 是当前最佳方案，后续第三创新点实验应以该 checkpoint 和融合逻辑为主线。

## 常用训练命令

进入服务器项目目录：

```bash
cd /home/u104754251515/baseline/CasMVSNet20260604
```

通用训练 helper：

```bash
bash scripts/train_baseline.sh <tag> <train.py 参数...>
```

R2 训练模板：

```bash
bash scripts/train_baseline.sh <tag> \
  --epochs 16 \
  --batch_size 4 \
  --pin_m \
  --use_rafe \
  --use_view_attention \
  --view_attention_mode single_pass_reliability_weighted
```

Adaptive R2 训练模板：

```bash
bash scripts/train_baseline.sh <tag> \
  --epochs 16 \
  --batch_size 4 \
  --pin_m \
  --use_rafe \
  --use_adaptive_r2 \
  --use_view_attention \
  --view_attention_mode single_pass_reliability_weighted
```

## 常用测试与评估命令

优先使用现有 helper：

```bash
bash scripts/test_fuse_eval.sh <checkpoint> <output_tag> <test.py 参数...>
```

R2 测试参数：

```bash
--use_rafe \
--use_view_attention \
--view_attention_mode single_pass_reliability_weighted
```

Adaptive R2 额外参数：

```bash
--use_adaptive_r2
```

官方 MATLAB 评估结果同步回训练机时，通常放在：

```text
/home/u104754251515/baseline/CasMVSNet20260604/results_m/official_matlab_<tag>...
```

## 接手后的优先级

1. 确认 active run 状态和最新 checkpoint。
2. 如果训练完成，跑 test + fusion + local eval。
3. 如果本地指标值得看，再跑官方 MATLAB 评估。
4. 把关键结果更新到 `docs/02_experiment_results.md`。
5. 如果改了模型设计，把原因更新到 `docs/03_improvement_log.md`。
6. 如果开始 FGDR，实现前先读 `docs/04_third_innovation_fgdr.md`，并保持 `--use_fgdr` 独立开关。

## 容易踩的坑

- DTU 的 Accuracy、Completeness、Overall 都是距离误差，越低越好。
- 本地 `matlab.py` 和官方 MATLAB 评估要分开记录。
- 服务器项目目录可能不是 git 仓库，同步前先确认当前文件来源。
- 不要提交 checkpoint、点云、TensorBoard events、完整评估输出目录。
- 不要把旧目录里的模块默认当成当前有效模型行为。
