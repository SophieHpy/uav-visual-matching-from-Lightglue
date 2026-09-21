# 00 · 整体管线与问题定义

## 问题

给定两张图片，找出其中的对应点（correspondences）。这是 SfM（运动恢复结构）、
视觉定位、图像拼接等任务的核心子问题。

经典方法分两步：
1. **检测+描述**：在每张图上找关键点（keypoint）并为每个点算一个描述子
   （descriptor，一个向量，"长得像的点描述子距离近"）。
2. **匹配**：比较两张图的描述子集合，决定谁和谁对应。

LightGlue 做的是第二步：输入不是图片本身，而是**两张图各自的
{keypoints: [N,2], descriptors: [N,256]}**，输出匹配对索引。

## 数据流总览

```
img0 ──[SuperPoint]──> kpts0 [M,2], desc0 [M,256] ─┐
                                                 ├─[LightGlue]─> matches [K,2]
img1 ──[SuperPoint]──> kpts1 [N,2], desc1 [N,256] ─┘
```

- M、N 不相等：匹配器处理变长集合（把"无法匹配"建模为 dustbin）。
- SuperPoint 是 CNN（本地处理像素）；LightGlue 是 Transformer
  （在**特征集合**上做注意力，完全没有图像卷积）。

## 为什么值得复现

- 规模小：核心 ~600 行，但包含完整 SOTA 思想——attention、可学习位置编码、
  optimal-transport 风格分配、自适应计算。
- 验证容易：官方权重可下载，能逐位对照自己的实现是否正确。
- 含金量高：ICCV 2023，比 SuperGlue 快 4–10 倍且更准。

## 关键论文思想（对应代码位置）

| 思想 | 代码 |
|------|------|
| keypoint 归一化到 [-1,1] | `normalize_keypoints` |
| 可学习傅里叶位置编码（RoPE 风格） | `LearnableFourierPositionalEncoding` |
| 每层 self-attn + 双向 cross-attn | `TransformerLayer` |
| cross-attn 共享 qk 投影（对称相似度） | `CrossBlock.to_qk` |
| matchability 头：这个点"配被匹配吗" | `MatchAssignment.matchability` |
| 双 softmax + certainty + dustbin | `sigmoid_log_double_softmax` |
| mutual-argmax + 阈值过滤 | `filter_matches` |
| 逐层早停（深度自适应） | `TokenConfidence` + `check_if_stop` |
| 逐层剪枝掉不可匹配点（宽度自适应） | `_prune` |

下一步：[01 · SuperPoint 特征提取器](01-superpoint.md)
