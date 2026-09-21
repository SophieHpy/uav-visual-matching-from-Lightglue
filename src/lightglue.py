"""LightGlue matcher — from-scratch implementation.

Pipeline (Lindenberger et al., ICCV 2023):
    keypoints + descriptors per image
      -> keypoint normalization to roughly [-1, 1]
      -> input projection of descriptors to the model dimension
      -> learnable Fourier positional encoding of keypoint (x, y)
      -> L transformer layers, each = self-attention then cross-attention
         between the two feature sets
      -> per-layer MatchAssignment head: similarity matrix +
         matchability logits -> log assignment matrix with a dustbin
      -> mutual-argmax match filtering
    Adaptive inference: a confidence token can stop the stack early
    (depth) and low-matchability keypoints are pruned on the fly
    (width). Both are disabled when the corresponding confidence
    threshold is set to -1.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def normalize_keypoints(kpts, size=None):
    """Center keypoints and scale the larger image side to [-1, 1]."""
    if size is None:
        size = 1 + kpts.amax(-2) - kpts.amin(-2)
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    shift = size / 2
    scale = size.amax(-1) / 2
    return (kpts - shift[..., None, :]) / scale[..., None, None]


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), -1).flatten(-2)


class LearnableFourierPositionalEncoding(nn.Module):
    """sin/cos encoding with a learned projection of the position vector."""

    def __init__(self, in_dim: int, out_dim: int, gamma: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim // 2, bias=False)
        nn.init.normal_(self.proj.weight, std=gamma**-2)

    def forward(self, x):
        """x: [B, N, in_dim] -> [2, B, heads, N, head_dim] (cos, sin pair)."""
        projected = self.proj(x)
        emb = torch.stack((projected.cos(), projected.sin()), 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    """Predicts how "settled" each keypoint state is -> early stopping."""

    def __init__(self, dim: int):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0, desc1):
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )


class Attention(nn.Module):
    """Scaled dot-product attention; uses fused SDPA when available."""

    def forward(self, q, k, v):
        if q.shape[-2] == 0 or k.shape[-2] == 0:
            return q.new_zeros((*q.shape[:-1], v.shape[-1]))
        return F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous()
        )


class SelfBlock(nn.Module):
    """Multi-head self-attention over one image's features + FFN."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attn = Attention()
        self.out = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x, encoding):
        qkv = self.qkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = q * encoding[0] + rotate_half(q) * encoding[1]
        k = k * encoding[0] + rotate_half(k) * encoding[1]
        context = self.attn(q, k, v)
        message = self.out(context.transpose(1, 2).flatten(-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    """Bidirectional cross-attention between the two feature sets.

    One shared query/key projection per side makes the similarity
    symmetric; the two softmax directions are computed from the same
    score matrix.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.to_qk = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x0, x1):
        split = lambda t: t.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)  # noqa
        qk0, qk1 = split(self.to_qk(x0)), split(self.to_qk(x1))
        v0, v1 = split(self.to_v(x0)), split(self.to_v(x1))
        qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
        sim = torch.einsum("bhid,bhjd->bhij", qk0, qk1)
        attn01 = F.softmax(sim, dim=-1)  # image0 queries -> image1
        attn10 = F.softmax(
            sim.transpose(-2, -1).contiguous(), dim=-1
        )  # image1 queries -> image0
        m0 = torch.einsum("bhij,bhjd->bhid", attn01, v1)
        m1 = torch.einsum("bhji,bhjd->bhid", attn10.transpose(-2, -1), v0)
        m0 = self.to_out(m0.transpose(1, 2).flatten(-2))
        m1 = self.to_out(m1.transpose(1, 2).flatten(-2))
        return (
            x0 + self.ffn(torch.cat([x0, m0], -1)),
            x1 + self.ffn(torch.cat([x1, m1], -1)),
        )


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads)
        self.cross_attn = CrossBlock(embed_dim, num_heads)

    def forward(self, desc0, desc1, enc0, enc1):
        desc0 = self.self_attn(desc0, enc0)
        desc1 = self.self_attn(desc1, enc1)
        return self.cross_attn(desc0, desc1)


def sigmoid_log_double_softmax(sim, z0, z1):
    """Log assignment matrix: joint softmax over both axes + certainty
    scaling, padded with a dustbin row/column for unmatched keypoints."""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))  # dustbin col
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))  # dustbin row
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.final_proj = nn.Linear(dim, dim)
        self.matchability = nn.Linear(dim, 1)

    def forward(self, desc0, desc1):
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        d = mdesc0.shape[-1]
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0, z1 = self.matchability(desc0), self.matchability(desc1)
        return sigmoid_log_double_softmax(sim, z0, z1), sim

    def get_matchability(self, desc):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, threshold: float):
    """Mutual-argmax over the assignment matrix + confidence threshold."""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    idx0 = torch.arange(m0.shape[1], device=m0.device)[None]
    idx1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = idx0 == m1.gather(1, m0)
    mutual1 = idx1 == m0.gather(1, m1)
    mscores0 = torch.where(mutual0, max0.values.exp(), 0.0)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), 0.0)
    valid0 = mutual0 & (mscores0 > threshold)
    valid1 = mutual1 & valid0.gather(1, m1)
    return (
        torch.where(valid0, m0, -1),
        torch.where(valid1, m1, -1),
        mscores0,
        mscores1,
    )


class LightGlue(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        descriptor_dim: int = 256,
        n_layers: int = 9,
        num_heads: int = 4,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        add_scale_ori: bool = False,
    ):
        super().__init__()
        self.conf = {
            "input_dim": input_dim,
            "descriptor_dim": descriptor_dim,
            "n_layers": n_layers,
            "num_heads": num_heads,
            "depth_confidence": depth_confidence,
            "width_confidence": width_confidence,
            "filter_threshold": filter_threshold,
            "add_scale_ori": add_scale_ori,
        }
        d = descriptor_dim
        self.input_proj = (
            nn.Linear(input_dim, d) if input_dim != d else nn.Identity()
        )
        head_dim = d // num_heads
        # SIFT-style features also encode per-point scale/orientation,
        # which ride along as two extra positional channels
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * add_scale_ori, head_dim
        )
        self.layers = nn.ModuleList(
            [TransformerLayer(d, num_heads) for _ in range(n_layers)]
        )
        self.assignments = nn.ModuleList(
            [MatchAssignment(d) for _ in range(n_layers)]
        )
        self.confidence_heads = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n_layers - 1)]
        )
        self.register_buffer(
            "confidence_thresholds",
            torch.tensor(
                [self.confidence_threshold(i) for i in range(n_layers)]
            ),
        )

    def confidence_threshold(self, layer_index: int) -> float:
        """Per-layer bar for "this point's state is final": decays from
        ~0.9 toward 0.8 as layers progress."""
        return np.clip(
            0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf["n_layers"]), 0, 1
        )

    def forward(self, data: dict) -> dict:
        """data: {"image0"/"image1": {"keypoints", "descriptors",
        "image_size"}} -> match indices + scores + pruning info."""
        kpts0 = normalize_keypoints(
            data["image0"]["keypoints"], data["image0"].get("image_size")
        )
        kpts1 = normalize_keypoints(
            data["image1"]["keypoints"], data["image1"].get("image_size")
        )
        if self.conf["add_scale_ori"]:
            kpts0 = torch.cat(
                [kpts0, data["image0"]["scales"][..., None],
                 data["image0"]["oris"][..., None]],
                -1,
            )
            kpts1 = torch.cat(
                [kpts1, data["image1"]["scales"][..., None],
                 data["image1"]["oris"][..., None]],
                -1,
            )
        desc0 = self.input_proj(data["image0"]["descriptors"].detach())
        desc1 = self.input_proj(data["image1"]["descriptors"].detach())
        enc0 = self.posenc(kpts0)
        enc1 = self.posenc(kpts1)

        b, m = kpts0.shape[:2]
        n = kpts1.shape[1]
        device = kpts0.device
        do_early_stop = self.conf["depth_confidence"] > 0
        do_pruning = self.conf["width_confidence"] > 0

        if do_pruning:
            ind0 = torch.arange(m, device=device)[None]
            ind1 = torch.arange(n, device=device)[None]
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)

        last_i = 0
        for i, layer in enumerate(self.layers):
            if desc0.shape[1] == 0 or desc1.shape[1] == 0:
                break
            desc0, desc1 = layer(desc0, desc1, enc0, enc1)
            last_i = i
            if i == self.conf["n_layers"] - 1:
                break
            if do_early_stop or do_pruning:
                tok0, tok1 = self.confidence_heads[i](desc0, desc1)
            if do_early_stop:
                conf = torch.cat([tok0[..., :m], tok1[..., :n]], -1)
                th = self.confidence_thresholds[i]
                ratio_confident = 1.0 - (conf < th).float().sum() / (m + n)
                if ratio_confident > self.conf["depth_confidence"]:
                    break
            if do_pruning:
                gate = tok0 if do_early_stop else None
                desc0, enc0, ind0, prune0 = self._prune(
                    desc0, enc0, ind0, prune0, gate, i
                )
                gate = tok1 if do_early_stop else None
                desc1, enc1, ind1, prune1 = self._prune(
                    desc1, enc1, ind1, prune1, gate, i
                )

        if desc0.shape[1] == 0 or desc1.shape[1] == 0:
            empty = {
                "matches": kpts0.new_empty((b, 0, 2), dtype=torch.long),
                "scores": kpts0.new_empty((b, 0)),
                "stop": last_i + 1,
            }
            return empty

        scores, _ = self.assignments[last_i](desc0, desc1)
        m0, m1, ms0, _ = filter_matches(
            scores, self.conf["filter_threshold"]
        )
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            i0, i1 = torch.where(valid)[0], m0[k][valid]
            if do_pruning:
                i0 = ind0[k, i0]
                i1 = ind1[k, i1]
            matches.append(torch.stack([i0, i1], -1))
            mscores.append(ms0[k][valid])
        return {
            "matches": matches,
            "scores": mscores,
            "stop": last_i + 1,
            "prune0": prune0 if do_pruning else None,
            "prune1": prune1 if do_pruning else None,
        }

    def _prune(self, desc, enc, ind, prune, token, layer_i):
        scores = self.assignments[layer_i].get_matchability(desc)
        keep = scores > (1 - self.conf["width_confidence"])
        if token is not None:
            # never drop points whose state is still evolving
            keep |= token <= self.confidence_thresholds[layer_i]
        keep_idx = torch.where(keep[0])[0] if keep.dim() > 1 else torch.where(keep)[0]
        prune = prune.clone()
        prune[:, ind.index_select(1, keep_idx)] += 1
        return (
            desc.index_select(1, keep_idx),
            enc.index_select(-2, keep_idx),
            ind.index_select(1, keep_idx),
            prune,
        )
