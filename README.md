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

## Evaluation

DTU metrics are distance errors, so lower is better. Current baseline, SP-RWCV, and R2-MVSNet evaluation summaries are tracked in [docs/evaluation_results.md](docs/evaluation_results.md), with raw CSV files in [docs/results](docs/results).

## Project Memory

- [Handoff guide](docs/handoff_guide.md): server access pattern, working habits, training/evaluation conventions, and current active run.
- [Model improvement log](docs/model_improvement_log.md): baseline, SP-RWCV, RAFE, R2-MVSNet, and Adaptive R2 design notes.
- [Experiment results summary](docs/experiment_results_summary.md): official/local results, single-scene analysis, and active experiment status.

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
