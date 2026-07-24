# R2-MVSNet

本仓库基于 CasMVSNet，当前只保留三项经过官方评估验证的改进：

```text
Plain CasMVSNet -> SP-RWCV -> RAFE + SP-RWCV -> R2-MVSNet Full
```

默认无增强路径必须始终保持可运行，用来作为所有实验的 baseline。
Adaptive R2、Edge-View 和未接线实验入口已从运行代码中移除；文档中的相关内容仅作为历史负实验记录。

`R2-MVSNet Full` 表示：

```text
RAFE + SP-RWCV + Anchor-FGDR candidate fusion
```

## 文档阅读顺序

新智能体或新对话接手时，按下面顺序阅读：

1. [项目交接](docs/00_project_handoff.md)：当前状态、服务器目录、常用命令、下一步。
2. [工作习惯](docs/01_working_rules.md)：协作方式、实验记录规则、模型改动边界。
3. [实验结果](docs/02_experiment_results.md)：官方/本地 DTU 指标、关键结论、原始 CSV。
4. [改进日志](docs/03_improvement_log.md)：模型从 baseline 到当前主线的演进与负实验记录。
5. [第三创新点设计](docs/04_third_innovation_fgdr.md)：FGDR 深度几何重构与点云融合协同方案。

原始实验 CSV 放在 [docs/data](docs/data)。

## 最新评估

最新 22-scan DTU 官方复评结果：

```text
Accuracy     0.327053
Completeness 0.261178
Overall      0.294116
```

逐场景结果见 [docs/results/dtu_official_latest.csv](docs/results/dtu_official_latest.csv)。
仓库只记录指标，不提交对应 checkpoint、深度图、PFM、点云或完整评估工作目录。

## 常用入口

训练完整模型：

```bash
python train.py \
  --epochs 16 \
  --batch_size 4 \
  --pin_m \
  --use_rafe \
  --use_view_attention \
  --use_fgdr \
  --fgdr_anchor_base
```

测试完整模型：

```bash
python test.py \
  --loadckpt checkpoints/<tag>/model_000015.ckpt \
  --outdir outputs_retest/<tag> \
  --use_rafe \
  --use_view_attention \
  --use_fgdr \
  --fgdr_anchor_base
```

融合与评估：

```bash
python fusion-normal.py --outdir outputs_retest/<tag> --use_fgdr_candidates
python matlab.py --plyPath outputs_retest/<tag> --resultPath results_m/retest_<tag>
```

## 注意

`checkpoints/`、点云、TensorBoard 日志、完整评估输出等运行产物不要提交到 git。
