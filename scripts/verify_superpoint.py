"""Verify our SuperPoint against the official implementation.

Loads the same pretrained weights into both models and compares
keypoints, scores and descriptors on a real image pair.
Run: PYTHONPATH=src python scripts/verify_superpoint.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/home/ubuntu/repos/LightGlue")

from superpoint import SuperPoint  # ours
from lightglue import SuperPoint as OfficialSP  # noqa: E402
from lightglue.utils import load_image  # noqa: E402

DEV = torch.device("cpu")


def main():
    ours = SuperPoint(max_num_keypoints=2048).eval()
    sd = torch.load(ROOT / "weights/superpoint_converted.pth", weights_only=True)
    missing, unexpected = ours.load_state_dict(sd, strict=True), None
    print("state dict loaded strict=True")

    ref = OfficialSP(max_num_keypoints=2048).eval()

    with torch.no_grad():
        for name in ["graf/img1.ppm", "graf/img2.ppm"]:
            img = load_image(ROOT / "data/oxford" / name)[None].to(DEV)
            a = ours(img)
            b = ref({"image": img})
            print(f"\n== {name}")
            print(f"  keypoints:  ours {a['keypoints'].shape}  ref {b['keypoints'].shape}")
            if a["keypoints"].shape != b["keypoints"].shape:
                print("  MISMATCH in count!")
                continue
            dk = (a["keypoints"] - b["keypoints"]).abs().max().item()
            ds = (a["keypoint_scores"] - b["keypoint_scores"]).abs().max().item()
            dd = (a["descriptors"] - b["descriptors"]).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                a["descriptors"], b["descriptors"], dim=-1
            ).min()
            print(f"  max|dkeypoint|={dk:.2e}  max|dscore|={ds:.2e}")
            print(f"  max|ddesc|={dd:.2e}  min cosine={cos.item():.6f}")


if __name__ == "__main__":
    main()
