"""SuperPoint feature extractor — from-scratch implementation.

Architecture (DeTone et al., CVPRW 2018):
    grayscale image
      -> shared VGG-style encoder (8 conv blocks, 3x stride-2 maxpool)
      -> detector head: 1x1 conv -> 65 channels -> softmax over 64 bins
         + dustbin -> pixel-unshuffle to HxW score map -> NMS -> top-k
      -> descriptor head: conv -> L2-normalized 256-d dense map
         -> bilinear grid-sample at keypoint locations

The 65 detector channels encode an 8x8 super-pixel grid plus a "no
keypoint" dustbin; dropping the dustbin after softmax yields a dense
score map at 1/8 resolution that is unfolded back to image resolution.
"""

import kornia
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_grayscale
from torch import nn


def simple_nms(scores: torch.Tensor, radius: int) -> torch.Tensor:
    """Non-maximum suppression with max-pooling.

    Keeps a score only where it equals the local max, with two extra
    refinement rounds to break ties on plateaus.
    """

    def max_pool(x):
        return F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max & ~supp_mask)
    return torch.where(max_mask, scores, zeros)


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Bilinearly interpolate the dense descriptor map at keypoint xy's."""
    b, c, h, w = descriptors.shape
    # align the sampling grid to the centers of descriptor cells
    keypoints = keypoints - s / 2 + 0.5
    keypoints = keypoints / torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
        device=keypoints.device,
    )[None]
    keypoints = keypoints * 2 - 1  # to [-1, 1] for grid_sample
    sampled = F.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=True
    )
    return F.normalize(sampled.reshape(b, c, -1), p=2, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class SuperPoint(nn.Module):
    """Convolutional keypoint detector + descriptor, stride-8 output heads."""

    descriptor_dim = 256

    def __init__(
        self,
        nms_radius: int = 4,
        max_num_keypoints: int = 2048,
        detection_threshold: float = 0.0005,
        remove_borders: int = 4,
    ):
        super().__init__()
        self.conf = {
            "nms_radius": nms_radius,
            "max_num_keypoints": max_num_keypoints,
            "detection_threshold": detection_threshold,
            "remove_borders": remove_borders,
        }
        if max_num_keypoints is not None and max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive or None")

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        # shared encoder: /2, /2, /2 -> H/8 x W/8 feature map
        self.enc1a = ConvBlock(1, c1)
        self.enc1b = ConvBlock(c1, c1)
        self.enc2a = ConvBlock(c1, c2)
        self.enc2b = ConvBlock(c2, c2)
        self.enc3a = ConvBlock(c2, c3)
        self.enc3b = ConvBlock(c3, c3)
        self.enc4a = ConvBlock(c3, c4)
        self.enc4b = ConvBlock(c4, c4)
        self.pool = nn.MaxPool2d(2, 2)

        self.det_a = nn.Conv2d(c4, c5, kernel_size=3, padding=1)
        self.det_b = nn.Conv2d(c5, 65, kernel_size=1)  # 8x8 grid + dustbin

        self.desc_a = nn.Conv2d(c4, c5, kernel_size=3, padding=1)
        self.desc_b = nn.Conv2d(c5, self.descriptor_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict:
        """image: [B,1|3,H,W] in [0,1] -> keypoints/scores/descriptors."""
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)

        x = self.enc1a(image)
        x = self.enc1b(x)
        x = self.pool(x)
        x = self.enc2a(x)
        x = self.enc2b(x)
        x = self.pool(x)
        x = self.enc3a(x)
        x = self.enc3b(x)
        x = self.pool(x)
        x = self.enc4a(x)
        x = self.enc4b(x)

        # --- detector: softmax over 65 bins, drop dustbin, unshuffle ---
        scores = self.det_b(F.relu(self.det_a(x)))
        scores = F.softmax(scores, dim=1)[:, :-1]  # [B,64,H/8,W/8]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.conf["nms_radius"])

        if self.conf["remove_borders"]:
            p = self.conf["remove_borders"]
            scores[:, :p] = -1
            scores[:, :, :p] = -1
            scores[:, -p:] = -1
            scores[:, :, -p:] = -1

        idx = torch.where(scores > self.conf["detection_threshold"])
        scores = scores[idx]
        keypoints = [
            torch.stack(idx[1:3], dim=-1)[idx[0] == i] for i in range(b)
        ]
        scores = [scores[idx[0] == i] for i in range(b)]

        kmax = self.conf["max_num_keypoints"]
        if kmax is not None:
            keypoints, scores = list(
                zip(
                    *[
                        (
                            (lambda t, i: (k[i], s[i]))(
                                *torch.topk(s, min(kmax, len(s)), sorted=True)
                            )
                            if len(s) > 0
                            else (k, s)
                        )
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]  # -> (x,y)

        # --- descriptor head: dense map + sampling at keypoints ---
        descriptors = self.desc_b(F.relu(self.desc_a(x)))
        descriptors = F.normalize(descriptors, p=2, dim=1)
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, descriptors)
        ]

        return {
            "keypoints": torch.stack(keypoints),
            "keypoint_scores": torch.stack(scores),
            "descriptors": torch.stack(descriptors).transpose(-1, -2),
        }

    @torch.no_grad()
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
            [image.shape[-1] / w, image.shape[-2] / h],
            device=image.device, dtype=torch.float,
        )
        out = self.forward(image)
        out["keypoints"] = (out["keypoints"] + 0.5) / scale[None, None] - 0.5
        out["image_size"] = torch.tensor(
            [w, h], device=image.device, dtype=torch.float
        )[None]
        return out
