"""Match an image pair and save a visualization.

Usage:
    python scripts/run_match.py data/oxford/graf/img1.ppm \
        data/oxford/graf/img2.ppm -o results/graf12.png
"""

import argparse
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def _import(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


superpoint_mod = _import("superpoint")
lightglue_mod = _import("lightglue")
utils = _import("utils")
viz = _import("viz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image0")
    p.add_argument("image1")
    p.add_argument("-o", "--output", default="results/matches.png")
    p.add_argument("--max-kpts", type=int, default=2048)
    p.add_argument("--resize", type=int, default=None)
    p.add_argument("--depth-confidence", type=float, default=0.95)
    p.add_argument("--width-confidence", type=float, default=0.99)
    args = p.parse_args()

    extractor = superpoint_mod.SuperPoint(max_num_keypoints=args.max_kpts).eval()
    extractor.load_state_dict(
        torch.load(ROOT / "weights/superpoint_converted.pth", weights_only=True)
    )
    matcher = lightglue_mod.LightGlue(
        depth_confidence=args.depth_confidence,
        width_confidence=args.width_confidence,
    ).eval()
    matcher.load_state_dict(
        torch.load(
            ROOT / "weights/lightglue_superpoint_converted.pth", weights_only=True
        )
    )

    image0 = utils.load_image(args.image0, resize=args.resize)
    image1 = utils.load_image(args.image1, resize=args.resize)
    with torch.no_grad():
        feats0, feats1, matches01 = utils.match_pair(
            extractor, matcher, image0, image1
        )
    m = matches01["matches"]
    print(
        f"kpts: {len(feats0['keypoints'])}/{len(feats1['keypoints'])}, "
        f"matches: {len(m)}, stopped at layer {matches01['stop']}"
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    viz.plot_match_pair(
        image0,
        image1,
        feats0["keypoints"],
        feats1["keypoints"],
        m,
        matches01["scores"],
        path=args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
