"""Convert official pretrained checkpoints to this repo's naming.

Downloads the official SuperPoint / LightGlue checkpoints and rewrites
every parameter key to match the modules in ``src/``, proving each
tensor's role was understood during reimplementation.

Usage:  python scripts/convert_weights.py          # all checkpoints
Outputs to ``weights/`` (git-ignored, regenerated on demand).
"""

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
BASE = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/"

SUPERPOINT_KEYMAP = {
    "conv1a": "enc1a.conv",
    "conv1b": "enc1b.conv",
    "conv2a": "enc2a.conv",
    "conv2b": "enc2b.conv",
    "conv3a": "enc3a.conv",
    "conv3b": "enc3b.conv",
    "conv4a": "enc4a.conv",
    "conv4b": "enc4b.conv",
    "convPa": "det_a",
    "convPb": "det_b",
    "convDa": "desc_a",
    "convDb": "desc_b",
}


def convert_superpoint() -> Path:
    sd = torch.hub.load_state_dict_from_url(
        BASE + "superpoint_v1.pth", file_name="official_superpoint_v1.pth"
    )
    out = {}
    for key, value in sd.items():
        stem, _, suffix = key.rpartition(".")
        out[f"{SUPERPOINT_KEYMAP[stem]}.{suffix}"] = value
    dest = WEIGHTS / "superpoint_converted.pth"
    torch.save(out, dest)
    print(f"superpoint: {len(out)} tensors -> {dest.name}")
    return dest


def convert_lightglue(feature: str = "superpoint") -> Path:
    """Map official keys (old flat + new nested) onto src/lightglue.py."""
    sd = torch.hub.load_state_dict_from_url(
        BASE + f"{feature}_lightglue.pth",
        file_name=f"official_{feature}_lightglue.pth",
    )
    out = {}
    for key, value in sd.items():
        k = key
        # released checkpoint predates the ModuleList refactor:
        # self_attn.{i}.X -> transformers.{i}.self_attn.X -> ours layers.{i}.self_attn.X
        parts = k.split(".")
        if parts[0] in ("self_attn", "cross_attn"):
            k = f"layers.{parts[1]}.{parts[0]}." + ".".join(parts[2:])
        k = k.replace("transformers.", "layers.")
        k = k.replace("log_assignment.", "assignments.")
        k = k.replace("token_confidence.", "confidence_heads.")
        k = k.replace("posenc.Wr.", "posenc.proj.")
        k = k.replace("self_attn.Wqkv.", "self_attn.qkv.")
        k = k.replace("self_attn.out_proj.", "self_attn.out.")
        if k.endswith(".r") or k == "r":
            continue  # legacy buffer unused by the current architecture
        out[k] = value
    # the buffer is absent from the released ckpt; store it so the
    # converted file is self-contained under strict=True
    import numpy as np
    out["confidence_thresholds"] = torch.tensor(
        [np.clip(0.8 + 0.1 * np.exp(-4.0 * i / 9), 0, 1) for i in range(9)]
    )
    dest = WEIGHTS / f"lightglue_{feature}_converted.pth"
    torch.save(out, dest)
    print(f"lightglue[{feature}]: {len(out)} tensors -> {dest.name}")
    leftover = [k for k in out if k.startswith(("self_attn.", "cross_attn."))]
    assert not leftover, leftover
    return dest


if __name__ == "__main__":
    WEIGHTS.mkdir(exist_ok=True)
    convert_superpoint()
    convert_lightglue("superpoint")
    convert_lightglue("sift")
