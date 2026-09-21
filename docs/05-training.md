# 05 · 极简训练 demo：不靠 glue-factory 也能训起来

`scripts/train_mini.py`

官方训练在 [glue-factory](https://github.com/cvg/glue-factory) 里，
需要 MegaDepth 级别的大数据和 GPU。这个脚本把同样的思想压缩到
**14 对带真值单应矩阵的图片 + CPU**，证明整条训练 pipeline 是通的。

## 监督信号从哪来

Oxford Affine 数据集每个序列有 `img1..img6` 和 `H1toNp`——
img1→imgN 的 3×3 单应矩阵真值。于是：

```
kpts0 --warp(H)--> 预测在 img1 中的落点
与真实 kpts1 做 mutual-NN（3px 内互为最近邻）→ GT 匹配对
越界的点 → 监督到 dustbin
```

## 损失函数

分配矩阵 S ∈ log 概率，[M+1, N+1]：

```
loss = -( Σ_{(i,j)∈GT} S[i,j]
        + Σ_{i unmatched} S[i, dustbin]
        + Σ_{j unmatched} S[dustbin, j] ) / count
```

对**每一层**的 `assignments[i]` 都加 loss 再取平均——
这就是早停能 work 的原因：浅层的匹配头也被训练到能用。

## 设计选择

- **冻结 SuperPoint**：官方也是这么训的（描述子 detach），
  matcher 只需要学"在固定描述子空间里做匹配"。
- **关掉早停/剪枝**（两个 confidence=-1）：训练时 9 层全跑。
- 默认 `--init converted` 从官方权重微调（loss 起点低）；
  `--init scratch` 从零开始，更能直观看到"匹配数 0→有、loss ↓"。

## 实测（CPU，500 步，lr 3e-4，512 kpts/图，从零初始化）

```
eval graf:1-6 训练前: 0 matches, 精度 0%
step 0    loss 1.97
step 330  loss 0.18   eval: 26 matches, 26.9% correct   <- 峰值
step 499  loss 0.17   eval: 21 matches, 19.0% correct
```

loss 单调下降、匹配数 0→数十、留出对上的 GT 精度从 0% 升到 ~20–27%。
量还不大（CPU 上只跑了 500 步、14 对图），但**证明了完整链路**：
GT 构造 → 逐层分配损失 → 反传只更新 matcher → 泛化到未训练对。
日志存于 `results/train_scratch.log`，产物 `weights/lightglue_trained_scratch.pth`。

要逼近论文指标请换 GPU + glue-factory + MegaDepth；这个 demo 的价值是
"pipeline 我全懂且能跑通"。

## 学到的

- 分配矩阵的 dustbin 监督只需要 logNLL，一行 einsum/索引就能写完。
- 每层都挂 assignment 头 + 逐层平均损失 = 免费的 early-exit 训练。
- 图像对特征的缓存很关键：提取器结果不变，整个 150 步共用一次提取。
