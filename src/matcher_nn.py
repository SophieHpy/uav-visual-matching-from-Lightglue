"""Baseline matcher: mutual nearest neighbor + Lowe ratio test.

The classic approach — no learned components. Used in the benchmark
to show what the learned matcher buys you.
"""

import torch


def nn_match(desc0, desc1, ratio=0.8, mutual=True):
    """desc0 [M,D], desc1 [N,D] L2-normalized -> matches [K,2], scores [K].

    score = 1 - cosine distance (higher is better), compatible with
    LightGlue's [0,1] match-score convention.
    """
    sim = desc0 @ desc1.T  # cosine similarity, [M, N]
    val12, idx12 = sim.topk(2, dim=1)
    # Lowe ratio test on cosine sim: s1 > ratio... convert: dist ratio
    # keeps i only if best match is much better than second best
    keep = (1 - val12[:, 0]) < ratio * (1 - val12[:, 1] + 1e-9)
    m0 = torch.where(keep, idx12[:, 0], torch.full_like(idx12[:, 0], -1))
    if mutual:
        _, j_best = sim.max(dim=0)  # best source for each target
        m0 = torch.where(
            j_best[m0.clamp(min=0)] == torch.arange(len(m0), device=sim.device),
            m0,
            torch.full_like(m0, -1),
        )
    valid = m0 > -1
    idx0 = torch.where(valid)[0]
    matches = torch.stack([idx0, m0[valid]], -1)
    scores = sim[idx0, m0[valid]]
    return matches, scores
