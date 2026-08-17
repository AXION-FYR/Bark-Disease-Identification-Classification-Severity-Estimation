"""
PyTorch Dataset over the processed cache. Stage 1 (U-Net) plugs straight in.

    from src.dataset import BarkSegDataset
    train_ds = BarkSegDataset("train", augment=True)
    val_ds   = BarkSegDataset("valid", augment=False)

Augmentation defaults are deliberately conservative:
rotation is capped at +/-15 deg and shear/perspective are absent, because
stripe canker's signature is vertical anisotropy and heavy geometric warping
destroys the very signal Stage 2 and Stage 3 depend on. Flips (both axes) are
safe -- a vertical streak stays vertical under either.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, PROJECT_ROOT
from .imaging import read_image_rgb, read_mask_png

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def default_augment(size: int):
    """Albumentations pipeline. Returns None if albumentations isn't installed."""
    try:
        import albumentations as A
    except ImportError:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, border_mode=0, fill=0, fill_mask=0, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.3),
        A.GaussNoise(p=0.2),
    ])


class BarkSegDataset:
    """Yields (image CHW float32 normalised, mask 1HW float32) for Stage 1."""

    def __init__(self, split: str, augment: bool = False,
                 config_path=None, transform=None, return_meta: bool = False):
        cfg = load_config(config_path)
        proc = cfg.path("paths", "processed_root")
        manifest = proc / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{manifest} missing — run scripts/02_build_masks.py first")

        df = pd.read_csv(manifest)
        self.df = df[df.split == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no rows for split={split!r} in {manifest}")

        self.fmt = str(cfg["preprocess"]["save_format"]).lower()
        self.size = int(cfg["preprocess"]["size"])
        self.return_meta = return_meta
        self.transform = transform if transform is not None else (
            default_augment(self.size) if augment else None)

        if augment and self.transform is None:
            print("albumentations not installed — running without augmentation. "
                  "pip install albumentations")

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, row):
        ip, mp = PROJECT_ROOT / row.image_path, PROJECT_ROOT / row.mask_path
        if self.fmt == "png":
            return read_image_rgb(ip), read_mask_png(mp)
        return np.load(ip), np.load(mp).astype(np.uint8)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img, mask = self._load(row)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        y = np.ascontiguousarray(mask.astype(np.float32)[None])

        try:
            import torch
            x, y = torch.from_numpy(x), torch.from_numpy(y)
        except ImportError:
            pass

        if self.return_meta:
            return x, y, {"stem": row.stem, "class_idx": int(row.class_idx),
                          "class_name": row.class_name}
        return x, y


def class_weights(split: str = "train", config_path=None) -> np.ndarray:
    """Inverse-frequency weights over the per-image disease class (for Stage 2)."""
    cfg = load_config(config_path)
    df = pd.read_csv(cfg.path("paths", "processed_root") / "manifest.csv")
    df = df[df.split == split]
    counts = df.class_idx.value_counts().sort_index().values.astype(np.float32)
    w = counts.sum() / (len(counts) * counts)
    return w
