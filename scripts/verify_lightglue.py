"""Verify our LightGlue against the official implementation.

Runs both matchers on identical extracted features and compares the
match indices, scores and adaptive-inference statistics.
Run: python scripts/verify_lightglue.py
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/home/ubuntu/repos/LightGlue")
from lightglue import LightGlue as OfficialLG, SuperPoint as OfficialSP  # noqa
from lightglue.utils import load_image  # noqa: E402


def load_ours():
    """Load src/lightglue.py under a private name (the official package
    already occupies the ``lightglue`` module name)."""
    spec = importlib.util.spec_from_file_location(
        "ours_lightglue", ROOT / "src/lightglue.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LightGlue


def compare(name, ours_out, ref_out):
    print(f"\n== {name}")
    print(f"  stop layer: ours={ours_out['stop']}  ref={ref_out['stop']}")
    mo = ours_out["matches"][0]
    mr = ref_out["matches"][0]
    print(f"  matches: ours={mo.shape}  ref={mr.shape}")
    if mo.shape != mr.shape:
        print("  MISMATCH: different match count")
        return False
    same = (mo == mr).all().item()
    ds = (ours_out["scores"][0] - ref_out["scores"][0]).abs().max().item()
    print(f"  identical match indices: {same}   max|dscore|={ds:.2e}")
    if ours_out.get("prune0") is not None and "prune0" in ref_out:
        dp = (ours_out["prune0"] - ref_out["prune0"]).abs().max().item()
        print(f"  max|dprune|={dp:.2e}")
    return same and ds < 1e-5


def main():
    extractor = OfficialSP(max_num_keypoints=2048).eval()

    ref = OfficialLG(features="superpoint").eval()
    ours = load_ours()().eval()
    ours.load_state_dict(
        torch.load(ROOT / "weights/lightglue_superpoint_converted.pth",
                   weights_only=True),
        strict=True,
    )
    print("converted checkpoint loaded strict=True")

    pairs = [("graf/img1.ppm", "graf/img2.ppm"), ("graf/img1.ppm", "graf/img4.ppm")]
    ok = True
    with torch.no_grad():
        for n0, n1 in pairs:
            feats = []
            for n in (n0, n1):
                img = load_image(ROOT / "data/oxford" / n)[None]
                feats.append(extractor({"image": img}))
            data = {"image0": feats[0], "image1": feats[1]}
            out_o = ours(data)
            out_r = ref(data)
            ok &= compare(f"{n0} vs {n1}", out_o, out_r)
    print("\nALL EQUIVALENT" if ok else "\nMISMATCHES FOUND")


if __name__ == "__main__":
    main()
