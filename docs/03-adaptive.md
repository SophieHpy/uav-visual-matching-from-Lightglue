# 03 · 自适应计算：LightGlue 为什么快

`src/lightglue.py` 的 `forward` 主循环

LightGlue 的"Light"来自两层自适应：**深度**方向早停 +
**宽度**方向丢点。简单图像对少算几层、难的多算。

## 深度自适应：早停（early stopping）

每层之后，每个点过一个 `TokenConfidence`（Linear→Sigmoid，
输入 detach 的 desc——只读不回传梯度）。

```python
threshold_i = clip(0.8 + 0.1 * exp(-4 i / L), 0, 1)   # 逐层递减
ratio = 1 - #(conf < threshold_i) / (M + N)
if ratio > depth_confidence:  # 默认 0.95
    break
```

直觉：如果 95% 的点的"置信度"都超过阈值，说明特征状态已经收敛，
后面的层不会再改变结果——直接跳到匹配头。阈值随层数递减，
越到后面要求越松（保证最终能停）。

## 宽度自适应：关键点剪枝（point pruning）

没触发早停时，每层结束后做一轮清理：

```python
keep = matchability(desc) > (1 - width_confidence)   # 0.99 → 丢掉 <0.01 的
keep |= token <= confidence_thresholds[i]            # 还没收敛的点不许丢
desc, enc, ind = index_select(keep)
prune[:, ind] += 1   # 记录每个原始点走到了第几层（验证用）
```

- 剪枝依据是 `matchability`——"这个点能被匹配"的分数。
  配不上的点（遮挡、模糊区域）越早丢越省算力。
- `keep |= token <= th` 保护：仍在演化的点不会被误剪。
- 被剪掉点的原始索引存在 `ind` 里，最终输出的 match 索引
  映射回原始下标（`i0 = ind[i0]`）。

## 验证覆盖

`verify_lightglue.py` 默认开启两个机制
（depth_confidence=0.95, width_confidence=0.99），对比结果：

| 输出 | graf1-2 | graf1-4 |
|------|---------|---------|
| stop 层 | 5 == 5 | 7 == 7 |
| 匹配数 | 1162 == 1162 | 817 == 817 |
| 分数最大差 | 0.0 | 0.0 |
| prune 轨迹差 | 0 | 0 |

即：**每个点在第几层被剪掉、模型在第几层决定停**都逐位一致。

## 一个值得注意的实现细节

官方仓库在 CPU 上 pruning 阈值是 -1（"任何点数都剪"），
CUDA + Flash 时是 1536。本实现面向 CPU eager 路径，默认全开；
如需严格对齐官方行为可在推理时传 `width_confidence=-1` 关闭。

下一步：[04 · 如何验证复现正确](04-verification.md)
