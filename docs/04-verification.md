# 04 · 如何验证"复现"是对的

复现一个模型，可信度的阶梯大致是：

1. ~~跑起来不报错~~（门槛）
2. 输出形状对、demo 图看起来像（弱证据）
3. **加载官方权重后输出逐位一致**（强证据）← 本 repo 做到这一级
4. 从随机初始化训练到官方指标（最强，需完整训练资源）

## 方法：权重转换 + 双模型对照

官方发布的 checkpoint 沿用了旧版扁平命名
（`self_attn.0.Wqkv`、`cross_attn.0.to_qk`…），
`scripts/convert_weights.py` 做了三件事：

- 建立 `SUPERPOINT_KEYMAP` / LightGlue 的逐层重命名规则，
  把每个张量映射到本 repo 的模块名下——写这张表本身就是
  "我理解了每个参数在哪"的证明。
- 丢掉 checkpoint 里的遗留 buffer `log_assignment.{i}.r`
  （旧版本残留，当前架构不使用）。
- 补进 `confidence_thresholds`（官方用 `strict=False` 加载，
  buffer 由代码即时计算；我们把它算好写进 ckpt 使加载可 strict=True）。

## 逐位一致的边界在哪

同样权重、同样数学，为什么第一次对比仍有 1e-4 的分差？

- `F.scaled_dot_product_attention` 对**非连续**输入会选不同 kernel
  （累计顺序不同 → 浮点尾差）。补上 `.contiguous()` 后与官方 0 误差。
- 官方的 cross-attention 在 CPU 上走手写 einsum 路径而非 SDPA
  （它只在 CUDA+Flash 时用 SDPA）——本实现复刻了这条路径，
  因此跨层累计误差也是 0。
- 剩下的潜在差异源：`einsum` 的 reduction 顺序、autocast
  （本实现不含 AMP 路径）。要严格等价，逐层对照 desc 张量即可定位。

## 本 repo 的验证清单

| 检查 | 脚本 | 结果 |
|------|------|------|
| SuperPoint 逐点对比 | `verify_superpoint.py` | kpts/scores/desc 最大误差 0.0 |
| LightGlue 匹配对 | `verify_lightglue.py` | 索引/分数/早停/剪枝全一致 |
| 端到端可视化 | `run_match.py` | `results/graf12.png` |
| CPU 速度对照 | `benchmark.py` | 与官方 ±10% 内互有胜负 |

下一步：[05 · 极简训练 demo](05-training.md)
