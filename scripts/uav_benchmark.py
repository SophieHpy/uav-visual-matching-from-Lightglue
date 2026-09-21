"""UAV Visual Matching Benchmark.

Evaluation set: real UAV photos (OpenDroneMap Aukerman) warped by
random homographies -> pairs with exact ground truth.

Methods compared:
  SIFT+NN, SuperPoint+NN, SIFT+LightGlue, SuperPoint+LightGlue

Metrics per pair:
  * n_matches
  * precision: fraction of predicted matches whose warp error < thresh
  * RANSAC: homography inlier ratio, and corner reprojection error of
    the estimated homography vs ground truth
  * latency: extract / match wall time
  * resolution sweep over the extractor's resize parameter

Run: python scripts/uav_benchmark.py [--pairs-per-image 2] [--quick]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from superpoint import SuperPoint  # noqa: E402
from sift import SIFTExtractor  # noqa: E402
from lightglue import LightGlue  # noqa: E402
from matcher_nn import nn_match  # noqa: E402
from utils import load_image  # noqa: E402

DATA = ROOT / "data/uav"


def random_homography(rng, w, h):
    """Rotation+scale+translation+slight perspective around identity."""
    ang = np.deg2rad(rng.uniform(-30, 30))
    s = rng.uniform(0.7, 1.4)
    c, sn = np.cos(ang) * s, np.sin(ang) * s
    tx = rng.uniform(-0.15, 0.15) * w
    ty = rng.uniform(-0.15, 0.15) * h
    H = np.array([[c, -sn, tx], [sn, c, ty], [0, 0, 1.0]], np.float64)
    if rng.random() < 0.5:  # mild perspective
        H += np.array(
            [[0, 0, 0], [0, 0, 0],
             [rng.uniform(-2e-4, 2e-4), rng.uniform(-2e-4, 2e-4), 0]]
        )
    return H


def make_pairs(paths, per_image, rng, resize):
    pairs = []
    for p in paths:
        img = load_image(p, resize=resize)
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        h, w = arr.shape[:2]
        for k in range(per_image):
            H = random_homography(rng, w, h)
            warped = cv2.warpPerspective(arr, H, (w, h))
            im2 = torch.tensor(
                warped.transpose(2, 0, 1) / 255.0, dtype=torch.float
            )
            pairs.append((img, im2, torch.tensor(H, dtype=torch.float)))
    return pairs


def warp(pts, H):
    ones = torch.ones_like(pts[:, :1])
    ph = torch.cat([pts, ones], -1) @ H.T
    return ph[:, :2] / ph[:, 2:3].clamp_min(1e-8)


def eval_matches(kpts0, kpts1, matches, H, px_thresh=4.0):
    """Precision under GT homography + RANSAC stats."""
    out = {"n": len(matches)}
    if len(matches) == 0:
        return out | {"precision": 0.0, "ransac_inliers": 0,
                      "ransac_ratio": 0.0, "h_err": np.nan}
    p0 = kpts0[matches[:, 0]]
    p1 = kpts1[matches[:, 1]]
    err = (warp(p0, H) - p1).norm(dim=-1)
    out["precision"] = (err < px_thresh).float().mean().item()

    n0 = p0.numpy().astype(np.float64)
    n1 = p1.numpy().astype(np.float64)
    if len(matches) >= 4:
        H_est, inl = cv2.findHomography(
            n0, n1, cv2.RANSAC, ransacReprojThreshold=4.0
        )
        if inl is not None:
            out["ransac_inliers"] = int(inl.sum())
            out["ransac_ratio"] = float(inl.mean())
            if H_est is not None:
                # mean reprojection error of the image corners under
                # the estimated vs ground-truth homography
                corners = np.array(
                    [[0, 0], [640, 0], [640, 480], [0, 480]], np.float64
                )
                c_h = np.concatenate(
                    [corners, np.ones((4, 1))], axis=1
                )  # 4x3
                p_gt = (H.numpy() @ c_h.T).T
                p_gt = p_gt[:, :2] / p_gt[:, 2:3]
                p_est = (H_est @ c_h.T).T
                p_est = p_est[:, :2] / p_est[:, 2:3]
                out["h_err"] = float(np.abs(p_gt - p_est).mean())
            else:
                out["h_err"] = np.nan
        else:
            out["ransac_inliers"], out["ransac_ratio"], out["h_err"] = 0, 0.0, np.nan
    else:
        out["ransac_inliers"], out["ransac_ratio"], out["h_err"] = (
            0, 0.0, np.nan,
        )
    return out


class NNMatcher:
    def __call__(self, data):
        m, s = nn_match(
            data["image0"]["descriptors"][0],
            data["image1"]["descriptors"][0],
        )
        return {"matches": [m], "scores": [s], "stop": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-per-image", type=int, default=2)
    ap.add_argument("--resize", type=int, default=1024)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sweep-res", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    paths = sorted(DATA.glob("*.jpg"))
    if args.quick:
        paths = paths[:4]
    pairs = make_pairs(paths, args.pairs_per_image, rng, args.resize)
    print(f"{len(pairs)} pairs from {len(paths)} UAV images")

    sp = SuperPoint(max_num_keypoints=2048).eval()
    sp.load_state_dict(
        torch.load(ROOT / "weights/superpoint_converted.pth", weights_only=True)
    )
    sift = SIFTExtractor(max_num_keypoints=2048)
    lg_sp = LightGlue().eval()
    lg_sp.load_state_dict(
        torch.load(
            ROOT / "weights/lightglue_superpoint_converted.pth", weights_only=True
        )
    )
    lg_sift = LightGlue(input_dim=128, add_scale_ori=True).eval()
    lg_sift.load_state_dict(
        torch.load(ROOT / "weights/lightglue_sift_converted.pth", weights_only=True)
    )
    nnm = NNMatcher()

    methods = {
        "SIFT+NN": (sift, nnm),
        "SuperPoint+NN": (sp, nnm),
        "SIFT+LightGlue": (sift, lg_sift),
        "SuperPoint+LightGlue": (sp, lg_sp),
    }

    rows = []
    for name, (ext, mtc) in methods.items():
        agg = {k: [] for k in
               ["precision", "ransac_ratio", "h_err", "n", "t_ext", "t_match"]}
        for img0, img1, H in pairs:
            t0 = time.perf_counter()
            with torch.no_grad():
                f0 = ext.extract(img0, resize=args.resize)
                f1 = ext.extract(img1, resize=args.resize)
            t1 = time.perf_counter()
            with torch.no_grad():
                out = mtc({"image0": f0, "image1": f1})
            t2 = time.perf_counter()
            r = eval_matches(
                f0["keypoints"][0], f1["keypoints"][0],
                out["matches"][0], H,
            )
            for k in ["precision", "ransac_ratio", "h_err", "n"]:
                agg[k].append(r[k])
            agg["t_ext"].append((t1 - t0) * 1000 / 2)
            agg["t_match"].append((t2 - t1) * 1000)
        row = {"method": name}
        for k, v in agg.items():
            row[k] = float(np.nanmean(np.asarray(v, dtype=float)))
        rows.append(row)
        print(
            f"{name:22s} matches={row['n']:7.1f}  prec={row['precision']:.3f}  "
            f"ransac_inlier={row['ransac_ratio']:.3f}  "
            f"H_err={row['h_err']:6.2f}px  "
            f"extract={row['t_ext']:6.1f}ms match={row['t_match']:6.1f}ms"
        )

    out_csv = ROOT / "results/uav_benchmark.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out_csv}")

    if args.sweep_res:
        print("\n== resolution sweep (SuperPoint+LightGlue, SIFT+NN) ==")
        for res in [512, 768, 1024, 1536]:
            for name in ["SuperPoint+LightGlue", "SIFT+NN"]:
                ext, mtc = methods[name]
                ps, ns, ts = [], [], []
                for img0, img1, H in pairs[:6]:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        f0 = ext.extract(img0, resize=res)
                        f1 = ext.extract(img1, resize=res)
                        out = mtc({"image0": f0, "image1": f1})
                    ts.append(time.perf_counter() - t0)
                    r = eval_matches(
                        f0["keypoints"][0], f1["keypoints"][0],
                        out["matches"][0], H,
                    )
                    ps.append(r["precision"])
                    ns.append(r["n"])
                print(
                    f"  res={res:4d} {name:22s} matches={np.mean(ns):7.1f}  "
                    f"prec={np.mean(ps):.3f}  total={np.mean(ts)*1000:7.1f}ms"
                )


if __name__ == "__main__":
    main()
