"""SIFT feature extractor (OpenCV backend) — from-scratch wrapper.

Classic DoG keypoints + RootSIFT descriptors (L1-normalize, sqrt, L2),
plus duplicate removal / NMS on the detector lattice. Returns each
keypoint's scale and orientation too — LightGlue's `add_scale_ori`
mode consumes them as extra positional channels.
"""

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_grayscale


def filter_dog_points(points, scales, angles, image_shape, nms_radius, scores):
    """Dedup identical integer-lattice points (highest scale wins, then
    lowest angle), optionally followed by radius-NMS on the score map."""
    h, w = image_shape
    ij = np.round(points - 0.5).astype(int).T[::-1]

    buffer = np.zeros((h, w))
    np.maximum.at(buffer, tuple(ij), scores)
    keep = np.where(buffer[tuple(ij)] == scores)[0]
    ij = ij[:, keep]

    buffer[:] = np.inf
    o_abs = np.abs(angles[keep])
    np.minimum.at(buffer, tuple(ij), o_abs)
    keep = keep[buffer[tuple(ij)] == o_abs]
    ij = ij[:, buffer[tuple(ij)] == o_abs] if False else ij  # recomputed below

    if nms_radius > 0:
        buffer[:] = 0
        buffer[tuple(ij)] = scores[keep]
        local_max = F.max_pool2d(
            torch.from_numpy(buffer)[None],
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        )[0].numpy()
        keep = keep[(buffer == local_max)[tuple(ij)]]
    return keep


def rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = F.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return F.normalize(x, p=2, dim=-1, eps=eps)


class SIFTExtractor:
    """cv2.SIFT wrapper producing the same dict layout as SuperPoint."""

    descriptor_dim = 128

    def __init__(
        self,
        max_num_keypoints: int = 2048,
        detection_threshold: float = 0.0066667,
        edge_threshold: float = 10,
        num_octaves: int = 4,
        nms_radius: int = 0,
        use_rootsift: bool = True,
    ):
        self.conf = {
            "max_num_keypoints": max_num_keypoints,
            "nms_radius": nms_radius,
            "rootsift": use_rootsift,
        }
        self.sift = cv2.SIFT_create(
            contrastThreshold=detection_threshold,
            nfeatures=max_num_keypoints,
            edgeThreshold=edge_threshold,
            nOctaveLayers=num_octaves,
        )

    def _single(self, image: np.ndarray) -> dict:
        det, desc = self.sift.detectAndCompute(image, None)
        if desc is None:
            desc = np.zeros((0, self.descriptor_dim), np.float32)
        pred = {
            "keypoints": np.array([k.pt for k in det], np.float32).reshape(-1, 2),
            "keypoint_scores": np.array([k.response for k in det], np.float32),
            "scales": np.array([k.size for k in det], np.float32),
            "oris": np.deg2rad(
                np.array([k.angle for k in det], np.float32)
            ),
            "descriptors": desc.astype(np.float32),
        }
        if self.conf["nms_radius"] is not None:
            keep = filter_dog_points(
                pred["keypoints"],
                pred["scales"],
                pred["oris"],
                image.shape,
                self.conf["nms_radius"],
                scores=pred["keypoint_scores"],
            )
            pred = {k: v[keep] for k, v in pred.items()}
        pred = {k: torch.from_numpy(v) for k, v in pred.items()}
        n = self.conf["max_num_keypoints"]
        if n is not None and len(pred["keypoints"]) > n:
            idx = torch.topk(pred["keypoint_scores"], n).indices
            pred = {k: v[idx] for k, v in pred.items()}
        return pred

    def forward(self, image: torch.Tensor) -> dict:
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
        out = self._single((image[0, 0].numpy() * 255).astype(np.uint8))
        out = {k: v[None] for k, v in out.items()}
        if self.conf["rootsift"]:
            out["descriptors"] = rootsift(out["descriptors"])
        return out

    def extract(self, image: torch.Tensor, resize: int = 1024) -> dict:
        """Detect on a resized image (long edge = `resize`), report
        keypoints in original-image coordinates."""
        if image.dim() == 3:
            image = image[None]
        h, w = image.shape[-2:]
        if resize is not None:
            image = kornia.geometry.transform.resize(
                image, resize, side="long", antialias=True
            )
        scale = torch.tensor(
            [image.shape[-1] / w, image.shape[-2] / h], dtype=torch.float
        )
        out = self.forward(image)
        out["keypoints"] = (out["keypoints"] + 0.5) / scale[None, None] - 0.5
        out["image_size"] = torch.tensor([w, h], dtype=torch.float)[None]
        return out
