# R2-MVSNet

本仓库基于 CasMVSNet，当前主线是研究可靠性引导的多视图立体匹配改进：

```text
Plain CasMVSNet -> SP-RWCV -> RAFE + SP-RWCV -> Adaptive R2
```

默认无增强路径必须始终保持可运行，用来作为所有实验的 baseline。

## 文档阅读顺序

新智能体或新对话接手时，按下面顺序阅读：

1. [项目交接](docs/00_project_handoff.md)：当前状态、服务器目录、常用命令、下一步。
2. [工作习惯](docs/01_working_rules.md)：协作方式、实验记录规则、模型改动边界。
3. [实验结果](docs/02_experiment_results.md)：官方/本地 DTU 指标、关键结论、原始 CSV。
4. [改进日志](docs/03_improvement_log.md)：模型从 baseline 到 Adaptive R2 的演进原因。
5. [第三创新点设计](docs/04_third_innovation_fgdr.md)：FGDR 深度几何重构与点云融合协同方案。

原始实验 CSV 放在 [docs/data](docs/data)。

## 常用入口

训练 R2-MVSNet：

```bash
python train.py \
  --epochs 16 \
  --batch_size 4 \
  --pin_m \
  --use_rafe \
  --use_view_attention \
  --view_attention_mode single_pass_reliability_weighted
```

测试 R2-MVSNet：

```bash
python test.py \
  --loadckpt checkpoints/<tag>/model_000015.ckpt \
  --outdir outputs_retest/<tag> \
  --use_rafe \
  --use_view_attention \
  --view_attention_mode single_pass_reliability_weighted
```

融合与评估：

```bash
python fusion-normal.py --outdir outputs_retest/<tag>
python matlab.py --plyPath outputs_retest/<tag> --resultPath results_m/retest_<tag>
```

## 注意

`checkpoints/`、点云、TensorBoard 日志、完整评估输出等运行产物不要提交到 git。
