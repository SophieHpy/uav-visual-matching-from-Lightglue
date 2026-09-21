"""Match visualization helpers (matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np


def plot_match_pair(
    image0,
    image1,
    kpts0,
    kpts1,
    matches,
    scores=None,
    path=None,
    max_lines=300,
):
    """Draw side-by-side images with match lines.

    image0/1: [3,H,W] tensors; kpts/matches: tensors from match_pair.
    """
    im = [t.permute(1, 2, 0).cpu().numpy() for t in (image0, image1)]
    k0, k1 = kpts0.cpu().numpy(), kpts1.cpu().numpy()
    m = matches.cpu().numpy()
    s = scores.cpu().numpy() if scores is not None else np.ones(len(m))

    h0, w0 = im[0].shape[:2]
    h1, w1 = im[1].shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3))
    canvas[:h0, :w0] = im[0]
    canvas[:h1, w0:] = im[1]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(canvas)
    if len(m) > max_lines:  # keep the strongest matches readable
        top = np.argsort(-s)[:max_lines]
        m, s = m[top], s[top]
    for (i, j), c in zip(m, s):
        ax.plot(
            [k0[i, 0], k1[j, 0] + w0],
            [k0[i, 1], k1[j, 1]],
            c=plt.cm.viridis(c),
            lw=0.7,
            alpha=0.8,
        )
        ax.scatter([k0[i, 0]], [k0[i, 1]], c="r", s=4)
        ax.scatter([k1[j, 0] + w0], [k1[j, 1]], c="r", s=4)
    ax.set_title(f"{len(m)} matches")
    ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
