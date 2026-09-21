"""I/O and pipeline helpers for the matcher."""

from pathlib import Path

import cv2
import numpy as np
import torch


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    return image if grayscale else image[..., ::-1]  # BGR -> RGB


def load_image(path: Path, resize: int = None) -> torch.Tensor:
    """Read an image as a float tensor [3,H,W] in [0,1]."""
    image = read_image(path)
    if resize is not None:
        h, w = image.shape[:2]
        scale = resize / max(h, w)
        image = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float)


def rbd(data: dict) -> dict:
    """Remove the leading batch dimension from every tensor/list."""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def match_pair(extractor, matcher, image0, image1):
    """Extract features on both images and run the matcher.

    image0/image1: unbatched float tensors [3,H,W] in [0,1].
    Returns feats0, feats1, matches01 — all without batch dim.
    """
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    return rbd(feats0), rbd(feats1), rbd(matches01)
