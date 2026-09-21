# 02 · LightGlue 匹配器：在特征集合上做 Transformer

`src/lightglue.py`

## 输入与预处理

输入每张图的 `keypoints [N,2]`（像素坐标）和 `descriptors [N,256]`。

```python
kpts = normalize_keypoints(kpts, image_size)
# (kpts - size/2) / (max(W,H)/2)  -> 长边映到 [-1,1]，保持宽高比
```

归一化让位置编码与图像分辨率解耦——这是它泛化到任意尺寸的关键。

## 位置编码：可学习傅里叶特征（RoPE 用法）

`LearnableFourierPositionalEncoding`：一个无 bias 的 Linear 把 2 维
坐标投影到 head_dim/2=32 维频率，再取 cos/sin 对：

```
proj = Wr @ (x,y)          # [B,N,32]
enc  = [cos(proj), sin(proj)]  → repeat_interleave(2) → [B,N,64]
```

与标准 RoPE 一样**乘进 attention 的 q/k**（`q*cos + rotate_half(q)*sin`），
而不是加到 token 上。旋转式编码让 attention 感知的是**相对**位置关系。

## 每层 = SelfBlock + CrossBlock

```
desc0 ──SelfBlock(enc0)──> desc0' ──┐
                                    ├─ CrossBlock ──> 两边同时更新
desc1 ──SelfBlock(enc1)──> desc1' ──┘
```

### SelfBlock：各自图像内部的注意力

```
qkv = Wqkv(x) -> [B,heads,N,3,head_dim]  (768 = 4头×64维×3)
q,k 乘旋转编码 -> SDPA -> out_proj
x = x + FFN([x, message])      # FFN 输入是拼接，2D->2D->D
```

作用：让同一张图里分布合理的点互相"打招呼"（比如均匀铺开）。

### CrossBlock：两张图之间的注意力（本实现踩过的坑）

```python
qk0 = to_qk(x0);  qk1 = to_qk(x1)   # 同一投影既当 q 又当 k！
sim = qk0 · qk1ᵀ / sqrt(d)
attn01 = softmax(sim, dim=-1)       # 图0 的每个点看图1
attn10 = softmax(simᵀ, dim=-1)      # 图1 的每个点看图0
```

注意 `to_qk` 输出**一份**张量，对称地充当 query 和 key——
不是分开的 q/k。第一版实现我误把它 chunk 成两半，导致
early-stop 在第 2 层就触发、只输出 2 个匹配。教训：读代码时
"`qk` 这个名字就是字面意思"。

## 匹配分配：sigmoid log double softmax

`MatchAssignment` 产出两张东西：

1. `sim = (final_proj(d0) / d^0.25) · (final_proj(d1) / d^0.25)ᵀ`
   —— 配对相似度矩阵 [M,N]
2. `z = matchability(d)` —— 每个点"配不配被匹配"的 logit [M],[N]

然后组装**对数分配矩阵** [M+1, N+1]：

```
logA[i,j] = logsoftmax_row(sim) + logsoftmax_col(sim) + logsigmoid(z_i) + logsigmoid(z_j)
logA[i, N] = logsigmoid(-z_i)     # dustbin 列：i 配不上任何人
logA[M, j] = logsigmoid(-z_j)     # dustbin 行
```

多出来的第 M+1 行 / N+1 列是 **dustbin**（垃圾桶）：匹配不上时
概率流向那里。这正是 SuperGlue 系列处理"集合大小不等 +
部分点无对应"的核心机制。

## 输出：mutual argmax + 阈值

`filter_matches`：只保留**互为对方最优**的对子
（mutual nearest neighbor），且分数 `exp(logA[i,j]) > 0.1`。

## 验证结果

`scripts/verify_lightglue.py`：两对真实图像上，匹配索引、
匹配分数、早停层数、剪枝轨迹与官方实现**完全一致**（0 误差）。

下一步：[03 · 自适应计算：快在哪里](03-adaptive.md)
