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

## 当前 active run

Adaptive R2 在 2026-06-18 启动过训练：

```text
tag: 20260618_r2_adaptive_rafe_sprwcv_bs4_e16
pid: 57663
repo: /home/u104754251515/baseline/CasMVSNet20260604
checkpoint dir: checkpoints/20260618_r2_adaptive_rafe_sprwcv_bs4_e16
```

接手后优先确认这个 run 是否已经完成、是否有最新 checkpoint、是否需要测试/融合/官方评估。

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

## 容易踩的坑

- DTU 的 Accuracy、Completeness、Overall 都是距离误差，越低越好。
- 本地 `matlab.py` 和官方 MATLAB 评估要分开记录。
- 服务器项目目录可能不是 git 仓库，同步前先确认当前文件来源。
- 不要提交 checkpoint、点云、TensorBoard events、完整评估输出目录。
- 不要把旧目录里的模块默认当成当前有效模型行为。
