# R2-MVSNet Handoff Guide

This document is a compact memory handoff for a new Codex conversation.

## Server Access

Do not commit plaintext passwords to git, even for a private repository.

Training machine:

```bash
ssh -p 21785 u104754251515@10.91.28.4
```

Official evaluation machine:

```bash
ssh -p 23466 root@10.91.28.4
```

Use Paramiko from the local Windows workspace when normal SSH is awkward. The password is intentionally omitted from this repository; get it from the private chat or a local secret note.

Primary training repo:

```text
/home/u104754251515/baseline/CasMVSNet20260604
```

Old reference repo only:

```text
/home/u104754251515/baseline/CasMVSNet
```

GitHub repository:

```text
skksm13131/R2-MVSNet
```

## Collaboration Habits

- Communicate in Chinese unless asked otherwise.
- Focus on `CasMVSNet20260604`; use the old `CasMVSNet` directory only as a reference.
- DTU Accuracy, Completeness, and Overall are distance errors: lower is better.
- Keep official MATLAB evaluation separate from local `matlab.py` evaluation.
- Prefer server-side Paramiko commands from Windows PowerShell for remote work.
- Do not expose server passwords or raw credentials in public messages or committed docs.
- Keep heavy runtime artifacts out of git: checkpoints, point clouds, TensorBoard events, fused outputs, and full eval folders.

## Model Change Rules

- Preserve a runnable plain CasMVSNet path.
- Add new behavior behind explicit flags rather than changing defaults.
- Keep `train.py` and `test.py` flags synchronized.
- Make one controlled architectural change at a time.
- Before training, run a small import/forward smoke test.
- Prefer rollback-friendly edits in:
  - `models/cas_mvsnet.py`
  - `models/modules/view_attention.py`
  - `models/module.py`
  - `train.py`
  - `test.py`
- Treat dormant modules in `models/module.py` as idea bank only unless they are explicitly wired into `CascadeMVSNet`.
- When a method improves difficult scenes but regresses simple scenes, prefer adaptive gating instead of globally stronger weighting.

## Training Habits

Default training is 16 epochs.

Known batch sizes:

- Plain baseline: `batch_size=6` worked.
- SP-RWCV: `batch_size=5` worked.
- R2 / RAFE + SP-RWCV: `batch_size=5` OOM, `batch_size=4` worked.
- Adaptive R2: started with `batch_size=4`.

Useful wrapper:

```bash
cd /home/u104754251515/baseline/CasMVSNet20260604
bash scripts/train_baseline.sh <tag> <train.py args...>
```

R2 training command pattern:

```bash
bash scripts/train_baseline.sh <tag> \
  --epochs 16 \
  --batch_size 4 \
  --pin_m \
  --use_rafe \
  --use_view_attention \
  --view_attention_mode single_pass_reliability_weighted
```

Adaptive R2 command pattern:

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

## Test And Evaluation Habits

Use the existing helper when possible:

```bash
bash scripts/test_fuse_eval.sh <checkpoint> <output_tag> <test.py args...>
```

Typical R2 test flags:

```bash
--use_rafe \
--use_view_attention \
--view_attention_mode single_pass_reliability_weighted
```

Adaptive R2 test flags add:

```bash
--use_adaptive_r2
```

Official MATLAB evaluation runs on the eval machine and should sync results back under:

```text
/home/u104754251515/baseline/CasMVSNet20260604/results_m/official_matlab_<tag>...
```

## Current Active Training

Adaptive R2 training was started on 2026-06-18:

```text
tag: 20260618_r2_adaptive_rafe_sprwcv_bs4_e16
pid: 57663
repo: /home/u104754251515/baseline/CasMVSNet20260604
checkpoint dir: checkpoints/20260618_r2_adaptive_rafe_sprwcv_bs4_e16
```

The GPU demo holder was stopped before this training started.
