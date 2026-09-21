"""Minimal LightGlue training loop on homography-verified pairs.

Idea (same as glue-factory, scaled down):
  * Fixed extractor (SuperPoint) -> frozen descriptors per pair.
  * Ground-truth matches come from the dataset's homographies:
    warp keypoints of image0 into image1, keep mutual nearest
    neighbors within a radius; everything else is supervised onto the
    dustbin row/column of the log-assignment matrix.
  * Loss = negative log-likelihood of the assignment matrix, applied
    at *every* layer (so early exits stay meaningful), like the paper.

Runs on CPU in a few minutes. Not meant to reach paper accuracy —
it demonstrates the full training pipeline end-to-end.

Usage: python scripts/train_mini.py [--steps 150] [--init converted]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from superpoint import SuperPoint  # noqa: E402
from lightglue import LightGlue, normalize_keypoints  # noqa
from utils import load_image  # noqa: E402

DATA = ROOT / "data/oxford"


def load_homography(seq: str, k: int) -> np.ndarray:
    """Oxford 'H1toNp' maps img1 pixel coords -> imgN pixel coords."""
    return np.loadtxt(DATA / seq / f"H1to{k}p").reshape(3, 3)


def warp_points(pts: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(pts[:, :1])
    p = torch.cat([pts, ones], -1) @ H.T
    return p[:, :2] / p[:, 2:3].clamp_min(1e-8)


def gt_assignment(kpts0, kpts1, H01, size1, thresh=3.0):
    """mutual-NN GT: gt[i]=j or -1;  gt_col[j]=i or -1."""
    w, h = size1
    warped = warp_points(kpts0, H01)
    inb = (
        (warped[:, 0] >= 0) & (warped[:, 0] < w)
        & (warped[:, 1] >= 0) & (warped[:, 1] < h)
    )
    dist = torch.cdist(warped, kpts1)  # [M, N]
    d01, j01 = dist.min(dim=1)
    d10, i10 = dist.min(dim=0)
    mutual = torch.arange(len(kpts0)) == i10[j01]
    ok = inb & mutual & (d01 < thresh)
    gt0 = torch.where(ok, j01, torch.full_like(j01, -1))
    gt1 = torch.full((len(kpts1),), -1, dtype=torch.long)
    pos = ok.nonzero().squeeze(-1)
    gt1[j01[pos]] = pos
    return gt0, gt1


def assignment_loss(scores, gt0, gt1):
    """NLL of the log assignment matrix for a single pair (B=1).

    Positive cells get s[i, j]; unmatched rows get s[i, dustbin];
    unmatched columns get s[dustbin, j].
    """
    s = scores[0]  # [M+1, N+1]
    M, N = s.shape[0] - 1, s.shape[1] - 1
    pos_rows = torch.where(gt0 > -1)[0]
    col_matched = torch.zeros(N, dtype=torch.bool)
    col_matched[gt0[pos_rows]] = True
    loss = -(
        s[pos_rows, gt0[pos_rows]].sum()
        + s[:M, N][gt0 < 0].sum()
        + s[M, :N][~col_matched].sum()
    )
    count = len(pos_rows) + int((gt0 < 0).sum()) + int((~col_matched).sum())
    return loss / max(count, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kpts", type=int, default=512)
    p.add_argument("--init", choices=["converted", "scratch"], default="converted")
    p.add_argument("--out", default="weights/lightglue_finetuned.pth")
    args = p.parse_args()
    torch.manual_seed(0)

    extractor = SuperPoint(max_num_keypoints=args.kpts).eval()
    extractor.load_state_dict(
        torch.load(ROOT / "weights/superpoint_converted.pth", weights_only=True)
    )
    for q in extractor.parameters():
        q.requires_grad_(False)

    matcher = LightGlue(
        depth_confidence=-1, width_confidence=-1  # train all layers
    )
    if args.init == "converted":
        matcher.load_state_dict(
            torch.load(
                ROOT / "weights/lightglue_superpoint_converted.pth",
                weights_only=True,
            )
        )
    opt = torch.optim.Adam(matcher.parameters(), lr=args.lr)

    # ---- build the pair pool: (img1, imgk) for each sequence ----
    train_pairs = []
    for seq in ["graf", "boat", "wall"]:
        for k in [2, 3, 4, 5, 6]:
            if (seq, k) == ("graf", 6):
                continue  # held out for evaluation
            train_pairs.append((seq, 1, k))
    eval_pair = ("graf", 1, 6)

    cache = {}

    def feats_for(seq, i):
        key = (seq, i)
        if key not in cache:
            img = load_image(DATA / seq / f"img{i}.{'pgm' if seq=='boat' else 'ppm'}")
            with torch.no_grad():
                f = extractor.extract(img)
            f["image_size"] = torch.tensor(
                [[img.shape[-1], img.shape[-2]]], dtype=torch.float
            )
            cache[key] = f
        return cache[key]

    def prepare(seq, i, j):
        f0, f1 = feats_for(seq, i), feats_for(seq, j)
        if i == 1:
            H = torch.tensor(load_homography(seq, j), dtype=torch.float)
        else:
            H = torch.tensor(load_homography(seq, j), dtype=torch.float) @ torch.linalg.inv(
                torch.tensor(load_homography(seq, i), dtype=torch.float)
            )
        k0 = f0["keypoints"][0]
        k1 = f1["keypoints"][0]
        gt0, gt1 = gt_assignment(k0, k1, H, f1["image_size"][0])
        return f0, f1, gt0, gt1

    def nll_all_layers(f0, f1, gt0, gt1):
        """Forward pass computing the assignment loss at every layer."""
        kpts0 = f0["keypoints"]; kpts1 = f1["keypoints"]
        kpts0 = normalize_keypoints(kpts0, f0["image_size"])
        kpts1 = normalize_keypoints(kpts1, f1["image_size"])
        desc0 = matcher.input_proj(f0["descriptors"].detach())
        desc1 = matcher.input_proj(f1["descriptors"].detach())
        enc0 = matcher.posenc(kpts0)
        enc1 = matcher.posenc(kpts1)
        loss = 0
        for i, layer in enumerate(matcher.layers):
            desc0, desc1 = layer(desc0, desc1, enc0, enc1)
            scores, _ = matcher.assignments[i](desc0, desc1)
            loss = loss + assignment_loss(scores, gt0, gt1)
        return loss / len(matcher.layers)

    def evaluate():
        """Match count + precision vs GT homography on the held-out pair."""
        seq, i, j = eval_pair
        f0, f1, gt0, _ = prepare(seq, i, j)
        with torch.no_grad():
            out = matcher({"image0": f0, "image1": f1})
        m = out["matches"][0]
        if len(m) == 0:
            return 0, 0.0
        correct = (gt0[m[:, 0]] == m[:, 1]).float().mean().item()
        return len(m), correct

    print(f"training pairs: {len(train_pairs)}, kpts/img: {args.kpts}")
    n0, acc0 = evaluate()
    print(f"eval graf:1-6 before training: {n0} matches, GT precision {acc0:.2%}")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train_pairs))
    for step in range(args.steps):
        seq, i, j = train_pairs[order[step % len(train_pairs)]]
        f0, f1, gt0, gt1 = prepare(seq, i, j)
        loss = nll_all_layers(f0, f1, gt0, gt1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            n, acc = evaluate()
            print(
                f"step {step:4d}  pair {seq}:{i}-{j}  loss {loss.item():.4f}"
                f"   eval: {n} matches, {acc:.1%} correct"
            )
    n1, acc1 = evaluate()
    print(f"eval after training: {n1} matches, GT precision {acc1:.2%}")

    Path(args.out).parent.mkdir(exist_ok=True)
    torch.save(matcher.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
