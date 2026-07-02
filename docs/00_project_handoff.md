# 00 项目交接

这份文档给新智能体快速接手项目用。先读这里，再读工作习惯、实验结果和改进日志。

## 当前项目状态

本项目基于 CasMVSNet，当前研究方向是把可靠性建模加入多视图立体匹配流程。

当前主线：

```text
Plain CasMVSNet -> SP-RWCV -> RAFE + SP-RWCV -> R2-MVSNet Full
```

核心要求：plain CasMVSNet baseline 必须一直可运行，所有新模块都要能通过 flag 单独开关和做 ablation。

完整模型定义：

```text
R2-MVSNet Full = RAFE + SP-RWCV + Anchor-FGDR candidate fusion
```

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

当前完整模型已完成从头训练：

```text
train tag: 20260628_r2_anchor_fgdr_rafe_sprwcv_bs4_e16
checkpoint dir: checkpoints/20260628_r2_anchor_fgdr_rafe_sprwcv_bs4_e16
flags: --use_rafe --use_view_attention --view_attention_mode single_pass_reliability_weighted --use_fgdr --fgdr_anchor_base
batch_size: 4
epochs: 16
status: completed, final checkpoint model_000015.ckpt
```

Anchor-FGDR 保持原 R2 depth 为主输出和三级采样中心，只训练 refined/near/far 增量候选。

2026-06-30 已完成完整模型测试、候选融合、本地评估和官方 MATLAB 评估：

```text
eval tag: 20260630_r2_anchor_fgdr_candidate_fusion_m015_001
local:    Acc=0.312612, Comp=0.249581, Overall=0.281097
official: Acc=0.333268, Comp=0.267471, Overall=0.300370
official eval dir: /root/official_eval_20260630_r2_anchor_fgdr_candidate_fusion_m015_001_w8
```

相对 RAFE + SP-RWCV，官方 Overall 改善 `0.005500`，Accuracy 和 Completeness 同时改善。该 checkpoint 是当前完整模型，后续实验和消融均以它对应的模块定义为准。

主消融链只保留：

```text
Plain
SP-RWCV
RAFE + SP-RWCV
R2-MVSNet Full
```

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
