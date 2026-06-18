# 实验目录布局

这个文档说明训练、测试、融合、本地评估和官方评估的目录约定。

## 训练输出

训练结果保存在：

```text
checkpoints/<tag>/
```

常见文件：

- `RUN_INFO.md`：训练启动信息和完整命令。
- `run_train.sh`：可复现的训练命令脚本。
- `train.pid`：后台训练进程 PID。
- `logs/stdout.log`：训练 stdout 日志。
- `events.out.tfevents.*`：TensorBoard 日志。
- `model_*.ckpt`：每个 epoch 保存的 checkpoint。

## 测试和融合输出

测试、融合、本地评估流程输出保存在：

```text
outputs_retest/<tag>/
```

常见文件：

- `RUN_INFO.md`：测试/融合/评估启动信息。
- `run_workflow.sh`：可复现的 test + fusion + local eval 脚本。
- `workflow.pid`：后台 workflow 进程 PID。
- `logs/test.log`：测试日志。
- `logs/fusion.log`：融合日志。
- `logs/eval.log`：本地评估日志。
- `mvsnet*_l3.ply`：融合后的点云。

## 本地 Python 评估结果

本地 `matlab.py` 评估结果保存在：

```text
results_m/retest_<tag>/
```

常见文件：

- `evaluation_results_*.csv`：本地 Python 评估 CSV。

注意：本地 `matlab.py` 结果只用于辅助观察，论文主对比应优先使用官方 MATLAB 评估结果。

## 官方 MATLAB 评估结果

官方 MATLAB 评估在评估机上运行，通常目录是：

```text
/root/official_eval_<tag>_w8/
```

评估完成后同步回训练机：

```text
results_m/official_matlab_<tag>_w8/
```

复评时使用单独目录，避免覆盖第一次结果，例如：

```text
results_m/official_matlab_<tag>_w8_rerun1/
```

常见文件：

- `SUMMARY.md`：官方评估汇总。
- `official_results.csv`：22 个 DTU scan 的官方结果和平均值。
- `metrics/scan*.csv`：单场景结果。
- `metrics/scan*.matlab.log`：单场景 MATLAB 日志。

## 命名习惯

tag 尽量包含日期、模型、batch size、epoch、checkpoint 信息，例如：

```text
20260618_r2_adaptive_rafe_sprwcv_bs4_e16
20260618_r2_rafe_sprwcv_bs4_m015_001
```

约定：

- `bs4` 表示 batch size 为 4。
- `e16` 表示训练 16 个 epoch。
- `m015` 表示使用 `model_000015.ckpt`。
- `_001` 表示同一实验的第一个测试输出版本。
