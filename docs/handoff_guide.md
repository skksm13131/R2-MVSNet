# R2-MVSNet 交接指南

这个文档用于新开对话时快速继承项目记忆。

## 服务器连接

即使仓库是私有仓库，也不要把明文密码提交到 git。

训练机：

```bash
ssh -p 21785 u104754251515@10.91.28.4
```

官方评估机：

```bash
ssh -p 23466 root@10.91.28.4
```

本地 Windows 环境里，如果普通 SSH 不方便，优先用 Paramiko 连接服务器。密码不要写入仓库，需要时从私密聊天记录或本地 secret note 获取。

主要训练目录：

```text
/home/u104754251515/baseline/CasMVSNet20260604
```

旧参考目录，只作为参考：

```text
/home/u104754251515/baseline/CasMVSNet
```

GitHub 仓库：

```text
skksm13131/R2-MVSNet
```

## 协作习惯

- 默认使用中文沟通。
- 主要关注 `CasMVSNet20260604`，旧目录只用于查历史实现。
- DTU 的 Accuracy、Completeness、Overall 都是距离误差，越低越好。
- 官方 MATLAB 评估和本地 `matlab.py` 评估必须分开记录。
- 远程操作常用本地 PowerShell + Paramiko。
- 不要在公开消息或 git 文档里暴露明文密码。
- checkpoints、点云、TensorBoard events、融合输出、完整评估目录等大文件不要进 git。

## 模型修改规则

- 保留可运行的 plain CasMVSNet 路径。
- 新功能必须放在显式 flag 后面，不要改默认行为。
- `train.py` 和 `test.py` 的参数要同步。
- 一次只做一个清晰的结构改动。
- 长训练前先做小张量 import/forward 检查。
- 主要改动面：
  - `models/cas_mvsnet.py`
  - `models/modules/view_attention.py`
  - `models/module.py`
  - `train.py`
  - `test.py`
- `models/module.py` 里未接入 `CascadeMVSNet` 的模块只能当作想法库，不要默认认为它们生效。
- 如果困难场景提升但简单场景回退，优先考虑自适应门控，而不是全局增强模块强度。

## 训练习惯

默认训练 16 个 epoch。

已知 batch size：

- Plain baseline：`batch_size=6` 可跑。
- SP-RWCV：`batch_size=5` 可跑。
- R2 / RAFE + SP-RWCV：`batch_size=5` 会 OOM，`batch_size=4` 可跑。
- Adaptive R2：当前用 `batch_size=4`。

常用启动脚本：

```bash
cd /home/u104754251515/baseline/CasMVSNet20260604
bash scripts/train_baseline.sh <tag> <train.py 参数...>
```

R2 训练命令模板：

```bash
bash scripts/train_baseline.sh <tag> \
  --epochs 16 \
  --batch_size 4 \
  --pin_m \
  --use_rafe \
  --use_view_attention \
  --view_attention_mode single_pass_reliability_weighted
```

Adaptive R2 训练命令模板：

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

## 测试和评估习惯

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

Adaptive R2 额外加：

```bash
--use_adaptive_r2
```

官方 MATLAB 评估在评估机上跑，结果同步回训练机：

```text
/home/u104754251515/baseline/CasMVSNet20260604/results_m/official_matlab_<tag>...
```

## 当前训练

Adaptive R2 已在 2026-06-18 启动训练：

```text
tag: 20260618_r2_adaptive_rafe_sprwcv_bs4_e16
pid: 57663
repo: /home/u104754251515/baseline/CasMVSNet20260604
checkpoint dir: checkpoints/20260618_r2_adaptive_rafe_sprwcv_bs4_e16
```

启动该训练前，GPU 占用 demo 已停止。
