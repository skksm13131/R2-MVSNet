# Hybrid Baseline 规则 2026-06-04

## 为什么有这个规则

当前工作树结合了 clean baseline 模型路径和较新的训练、测试、评估 workflow。这样可以保持模型对比干净，同时保留服务器上已经习惯使用的脚本流程。

## 来源规则

- `models/` 是模型源码的主要依据。
- `datasets/`、`train.py`、`test.py` 和 scripts 主要服务训练测试流程。
- 仓库里出现的旧实验模块不等于当前模型已经启用。
- 有效模型改动必须接入 `CascadeMVSNet`，并且在 train/test 中有一致的 flag。

## 安全新增项

安全新增项必须有明确 flag，便于 ablation：

```bash
--use_view_attention
--view_attention_mode single_pass_reliability_weighted
--use_rafe
--use_adaptive_r2
```

## 不安全新增项

不要把旧 dormant module 直接混进默认路径。除非有明确 flag 和 baseline fallback，否则不要同时大改特征提取和 cost-volume 聚合。

## 实用检查

长训练前：

1. 确认 plain baseline 能实例化。
2. 确认新 flag 模型能实例化。
3. 跑一个小张量 smoke test。
4. 训练 tag 要清晰可复现。
