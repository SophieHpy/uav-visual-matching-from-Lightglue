"""Turn results/uav_benchmark.csv (+ sweep numbers) into report plots."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = list(csv.DictReader(open(ROOT / "results/uav_benchmark.csv")))
    names = [r["method"] for r in rows]
    prec = [float(r["precision"]) * 100 for r in rows]
    inl = [float(r["ransac_ratio"]) * 100 for r in rows]
    n = [float(r["n"]) for r in rows]
    t_ext = [float(r["t_ext"]) for r in rows]
    t_m = [float(r["t_match"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(names))
    colors = ["#6baed6", "#9ecae1", "#3182bd", "#08519c"]

    axes[0].bar(x, prec, color=colors)
    axes[0].bar(x, [p - i for p, i in zip(prec, inl)], bottom=inl,
                color="#fdae6b", alpha=0.6)
    axes[0].set_ylabel("%")
    axes[0].set_title("GT precision (blue) / RANSAC inlier (orange delta)")
    axes[0].set_xticks(x, names, rotation=20, ha="right")
    axes[0].set_ylim(0, 105)

    axes[1].bar(x, n, color=colors)
    axes[1].set_title("Matches per pair")
    axes[1].set_xticks(x, names, rotation=20, ha="right")

    axes[2].bar(x - 0.2, t_ext, width=0.4, label="extract", color="#74c476")
    axes[2].bar(x + 0.2, t_m, width=0.4, label="match", color="#fd8d3c")
    axes[2].set_title("Latency (ms, CPU)")
    axes[2].set_xticks(x, names, rotation=20, ha="right")
    axes[2].legend()
    axes[2].set_yscale("log")

    fig.tight_layout()
    out = ROOT / "results/uav_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
