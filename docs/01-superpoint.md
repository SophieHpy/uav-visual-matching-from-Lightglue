# 01 · SuperPoint：一个 CNN 同时做检测和描述

`src/superpoint.py`

## 网络结构

```
[1,H,W] 灰度图
  └─ enc1a/enc1b (64ch) → pool ── /2
  └─ enc2a/enc2b (64ch) → pool ── /4
  └─ enc3a/enc3b (128ch) → pool ── /8
  └─ enc4a/enc4b (128ch)
        │                        x: [B,128,H/8,W/8]
        ├─ 检测头: det_a(256) → det_b(65)   ── 3x3 + 1x1 conv
        └─ 描述头: desc_a(256) → desc_b(256) ── 3x3 + 1x1 conv
```

所有 conv 都是 3x3 + ReLU（除了输出层是 1x1）。整张图编码一次，
特征图缩小 8 倍——所以两个头都在 **H/8 × W/8** 的格子上工作。

## 检测头的巧妙之处：65 = 64 + 1

`det_b` 输出 65 个通道。每个 H/8×W/8 格子对应原图一个 8×8 patch：
65 个通道 = patch 里 64 个像素位置 + 1 个"这里没有关键点"的 dustbin。

```python
scores = softmax(det_out, dim=1)[:, :-1]   # [B,64,H/8,W/8]，丢掉 dustbin
# reshape (h,w,c8x8) -> 换回全分辨率 [B,H,W]
scores = scores.permute(0,2,3,1).reshape(b,h,w,8,8)
        .permute(0,1,3,2,4).reshape(b,h*8,w*8)
```

等价于 `pixel_unshuffle`：通道维的 8×8 折回空间维。之后：

1. `simple_nms`：max_pool 抑制非局部极大值（半径 4），两轮修正平局。
2. 边缘 4px 丢弃（特征在边缘不可靠）。
3. 阈值 0.0005 筛掉弱响应 → top-k 保留最多 `max_num_keypoints` 个。

## 描述头 + 亚像素采样

`desc_b` 给每个格子一个 256 维描述子（L2 归一化）。但关键点的
坐标是全分辨率的，落在格子"之间"——用 `grid_sample` 双线性插值：

```python
sample_descriptors(kpts, dense_desc_map, s=8)
```

坐标换算 `(kpts - s/2 + 0.5) / (w*s - s/2 - 0.5) * 2 - 1` 把像素坐标
对齐到格子中心再归一化到 [-1,1]。这是复现时最容易错的地方之一。

## 验证

`scripts/verify_superpoint.py` 用 `convert_weights.py` 把官方权重
改名后灌进我们的模块，同一张图上逐元素对比：

```
keypoints/scores/descriptors 最大绝对误差 = 0.0（逐位一致）
```

权重对应表（官方 → 本 repo）见 `scripts/convert_weights.py` 的
`SUPERPOINT_KEYMAP`。

## 许可证提醒

SuperPoint 的代码与权重属于 Magic Leap 的**限制性许可证**
（仅限研究/非商用）。本 repo 只重写架构实现并加载官方权重做验证，
权重文件不进入仓库。简历展示/学习用途没问题，商用需要另行授权。

下一步：[02 · LightGlue 匹配器](02-lightglue.md)
