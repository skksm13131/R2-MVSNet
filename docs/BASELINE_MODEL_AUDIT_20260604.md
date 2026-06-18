# True Baseline 模型审计 2026-06-04

## 范围

这个文档记录在加入当前可靠性模块之前，我们清理出来的 true baseline 状态。

## Baseline 路径

不启用任何可靠性或 attention flag 时，模型走标准 CasMVSNet 流程：

1. 使用 `FeatureNet` 提取参考图和源图特征。
2. 构建参考图 volume。
3. 对源图特征进行 homography warping。
4. 构建等权 variance cost volume。
5. 使用 3D CNN 做 cost regularization。
6. 使用 softmax + depth regression 得到深度。

## 当前有效的可选路径

源图可靠性方向主要从这些文件接入：

- `models/cas_mvsnet.py`
- `models/modules/view_attention.py`
- `train.py`
- `test.py`

核心运行参数：

```bash
--use_view_attention
--view_attention_mode <mode>
```

后续新增参数：

```bash
--use_rafe
--use_adaptive_r2
```

## 已废弃或暂未接入的想法

早期工作目录里出现过 Direct-SCRF v2、RMFE、UGDR、CADR、RAHS、normal guidance、geometry guidance 等想法。除非它们在当前 `CascadeMVSNet` 中明确通过 flag 接入，否则不要把它们当作当前有效模型行为。

## 规则

True baseline 必须一直可运行。新的模型行为必须可选、可解释、可 ablation。
