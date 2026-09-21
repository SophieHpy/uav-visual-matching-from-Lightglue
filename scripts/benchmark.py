"""CPU benchmark: our implementation vs the official one.

Times the matcher alone on synthetic feature sets at several keypoint
counts. Run: python scripts/benchmark.py [--kpts 256 512 1024 2048]
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/home/ubuntu/repos/LightGlue")
from lightglue import LightGlue as Official  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "ours_lg", ROOT / "src/lightglue.py"
)
ours_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ours_mod)
Ours = ours_mod.LightGlue


def bench(model, data, iters=20, warmup=3):
    with torch.no_grad():
        for _ in range(warmup):
            model(data)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(data)
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kpts", type=int, nargs="+",
                   default=[256, 512, 1024, 2048])
    args = p.parse_args()

    torch.manual_seed(0)
    ref = Official(features="superpoint").eval()
    ours = Ours().eval()
    ours.load_state_dict(
        torch.load(ROOT / "weights/lightglue_superpoint_converted.pth",
                   weights_only=True)
    )

    print(f"{'kpts':>6} | {'ours (ms)':>10} | {'official (ms)':>13} | {'stop':>4}")
    for k in args.kpts:
        data = {
            "image0": {
                "keypoints": torch.rand(1, k, 2) * 640,
                "descriptors": torch.nn.functional.normalize(
                    torch.randn(1, k, 256), dim=-1
                ),
                "image_size": torch.tensor([[640.0, 480.0]]),
            },
            "image1": {
                "keypoints": torch.rand(1, k, 2) * 640,
                "descriptors": torch.nn.functional.normalize(
                    torch.randn(1, k, 256), dim=-1
                ),
                "image_size": torch.tensor([[640.0, 480.0]]),
            },
        }
        t_o = bench(ours, data)
        t_r = bench(ref, data)
        print(f"{k:>6} | {t_o:>10.1f} | {t_r:>13.1f} |")


if __name__ == "__main__":
    main()
