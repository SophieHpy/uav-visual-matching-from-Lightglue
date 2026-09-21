# 06 · UAV Visual Matching Benchmark

`scripts/uav_benchmark.py` · 原始数据 `results/uav_benchmark.csv` · 图 `results/uav_benchmark.png`

## 评测集

12 张真实 UAV 航拍图（OpenDroneMap Aukerman 集，~4000×3000px），
每张随机采样 2 个单应矩阵 warp 出第二张图 → **24 对带精确 GT
单应的评测对**。GT 同时监督"匹配的精度"和"RANSAC 估计的好坏"。

## 四个方法

| 方法 | 提取器 | 匹配器 |
|------|--------|--------|
| SIFT+NN | OpenCV DoG+RootSIFT | mutual NN + ratio 0.8 |
| SuperPoint+NN | SuperPoint (conv) | mutual NN + ratio 0.8 |
| SIFT+LightGlue | 同上 | LightGlue (sift 权重, add_scale_ori) |
| SuperPoint+LightGlue | 同上 | LightGlue (superpoint 权重) |

## 主结果（resize=1024，24 对平均，CPU）

| 方法 | 匹配数 | GT 精度 | RANSAC inlier 率 | H 误差(px) | 提取 | 匹配 |
|------|--------|---------|------------------|------------|------|------|
| SIFT+NN | 545 | 88.7% | 88.7% | 0.09 | 66ms | 2ms |
| SuperPoint+NN | 718 | 95.5% | 95.4% | 0.14 | 632ms | 2ms |
| SIFT+LightGlue | 620 | **96.8%** | 96.8% | 0.10 | 59ms | 173ms |
| SuperPoint+LightGlue | **801** | **97.1%** | 97.1% | 0.18 | 611ms | 97ms |

（precision = 预测匹配经 GT 单应 warp 后误差 <4px 的比例；
H_err = 用匹配 RANSAC 估计的单应 vs GT 的图像四角重投影误差）

## 结论（可直接写进简历的观察）

1. **学习匹配器对老式特征提升最大**：SIFT+NN 88.7% → SIFT+LightGlue
   96.8%（+8pt），而 SuperPoint 只 +1.6pt——LightGlue 把 SIFT 缺的那部分
   "语义判别力"补了回来。
2. **匹配数与精度同时提升**：SP+LG 比 SIFT+NN 多 47% 匹配且更准。
3. **代价是延迟**：NN 匹配 ~2ms，LightGlue ~100-170ms——嵌入式 UAV
   场景下这就是论文自适应机制存在的理由。
4. **分辨率敏感度不对称**（下方扫描）：SuperPoint+LightGlue 匹配数随
   分辨率近似线性增长（268→855），SIFT 很快饱和在 ~500——
   高分辨率航拍图上 learned extractor 更受益。

### 分辨率扫描（6 对子集）

| resize | SP+LG 匹配数 / 精度 | SIFT+NN 匹配数 / 精度 |
|--------|---------------------|------------------------|
| 512 | 268 / 83.1% | 513 / 89.5% |
| 768 | 491 / 91.6% | 490 / 86.3% |
| 1024 | 800 / 96.5% | 483 / 85.0% |
| 1536 | 855 / 98.2% | 498 / 87.0% |

## 复跑

```bash
python scripts/uav_benchmark.py --pairs-per-image 2 --sweep-res
python scripts/plot_benchmark.py
```
