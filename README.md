# R2-MVSNet

Reliability-aware Representation and Reliability-weighted Cost Volume Network for multi-view stereo.

This repository is based on CasMVSNet and keeps the original cascade MVS pipeline:

```text
FeatureNet -> cascade depth sampling -> homography warping -> cost-volume regularization -> depth regression
```

The current research branch adds two reliability-driven components:

- **RAFE**: Reliability-Aware Feature Extraction. FeatureNet predicts multi-scale feature reliability maps and injects structure-aware residual cues into stage features.
- **SP-RWCV**: Single-Pass Reliability-Weighted Cost Volume. Source-view warped features are weighted during variance cost-volume construction using learned reliability scores.

When both are enabled, RAFE reliability maps are homography-warped together with source features and are consumed by SP-RWCV while computing source-view weights.

## 评估

DTU 指标是距离误差，所以越低越好。当前 baseline、SP-RWCV、R2-MVSNet 的评估汇总记录在 [docs/evaluation_results.md](docs/evaluation_results.md)，原始 CSV 保存在 [docs/results](docs/results)。

## 项目记忆

- [交接指南](docs/handoff_guide.md)：服务器连接方式、工作习惯、训练/评估约定、当前 active run。
- [模型改进日志](docs/model_improvement_log.md)：baseline、SP-RWCV、RAFE、R2-MVSNet、Adaptive R2 的设计记录。
- [实验结果汇总](docs/experiment_results_summary.md)：官方/本地结果、单场景分析、当前实验状态。
- [实验目录布局](EXPERIMENT_LAYOUT.md)：训练、测试、融合、本地评估和官方评估的目录约定。

## Main Flags

Train plain baseline:

```bash
python train.py --epochs 16 --batch_size 6 --pin_m
```

Train RAFE only:

```bash
python train.py --epochs 16 --batch_size 5 --pin_m --use_rafe
```

Train R2-MVSNet:

```bash
python train.py \
  --epochs 16 \
  --batch_size 5 \
  --pin_m \
  --use_rafe \
  --use_view_attention \
  --view_attention_mode single_pass_reliability_weighted
```

Test R2-MVSNet:

```bash
python test.py \
  --loadckpt checkpoints/<tag>/model_000015.ckpt \
  --outdir outputs_retest/<tag> \
  --use_rafe \
  --use_view_attention \
  --view_attention_mode single_pass_reliability_weighted
```

Fuse and evaluate:

```bash
python fusion-normal.py --outdir outputs_retest/<tag>
python matlab.py --plyPath outputs_retest/<tag> --resultPath results_m/retest_<tag>
```

## Notes

- Dataset paths in `train.py`, `test.py`, and evaluation scripts are local defaults and should be overridden for a new environment.
- Runtime artifacts such as checkpoints, point clouds, tensorboard logs, and evaluation outputs are intentionally excluded from git.
