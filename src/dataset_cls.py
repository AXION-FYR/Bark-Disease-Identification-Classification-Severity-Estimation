"""
Stage 2 dataset — bark-bbox crop + scale jitter.

Two confounds are handled here, in this order.

1. AREA SHORTCUT. Raw bark fraction differs 2.7x across classes (healthy 0.113,
   rough bark 0.169, stripe canker 0.302) because stripe canker was shot from
   roughly 1.6x closer. A single number predicts the class at 85%. Cropping to
   the bark bounding box makes the trunk fill the frame regardless of shooting
   distance, so absolute area stops carrying the label.

2. THE CONFOUND THE CROP INTRODUCES. A distant trunk cropped small and upscaled
   to 224 is blurry; a close trunk cropped large and downscaled is sharp. Bark
   grain then lands at a different spatial frequency per class, which is exactly
   what an LBP branch measures. RandomResizedCrop with a wide scale range
   randomises that, so absolute texture scale is no longer informative and the
   branch is forced toward scale-invariant features.

Returns (image, mask, label). The mask is needed by the mask-conditioned SE
fusion, and is transformed in lockstep with the image throughout.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import load_config, PROJECT_ROOT
from .imaging import read_image_rgb, read_mask_png

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------
def crop_to_bbox(img: np.ndarray, mask: np.ndarray,
                 margin: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Crop both to the mask's bounding box, expanded by `margin` of its size."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return img, mask
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    H, W = mask.shape[:2]
    x0, x1 = max(0, x0 - mx), min(W, x1 + mx)
    y0, y1 = max(0, y0 - my), min(H, y1 + my)
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def clahe_lab(img_rgb: np.ndarray, clip: float = 2.0,
              grid: int = 8) -> np.ndarray:
    """CLAHE on L in LAB. Fixed parameters, identical across every split."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clip, (grid, grid)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def build_transform(size: int, train: bool):
    """
    Scale jitter is the load-bearing part. scale=(0.4, 1.0) means the crop
    covers 40-100% of the bbox area, so the model sees the same trunk at a
    ~1.6x range of apparent scales and cannot use absolute scale as a cue.

    Rotation stays at +/-15 and shear/perspective are absent: stripe canker's
    signature is vertical anisotropy and heavy warping destroys it. Flips on
    both axes are safe -- a vertical streak stays vertical under either.
    """
    try:
        import albumentations as A
    except ImportError:
        return None

    if not train:
        return A.Compose([A.Resize(size, size)])

    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.4, 1.0),
                            ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT,
                 fill=0, fill_mask=0, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.3),
    ])


# --------------------------------------------------------------------------
class BarkClsDataset:
    """Stage 2: disease classification from the masked, bbox-cropped trunk."""

    def __init__(self, split: str, size: int = 224, train: bool = False,
                 margin: float = 0.05, apply_clahe: bool = True,
                 apply_mask: bool = True, config_path=None,
                 drop_empty: bool = True, mask_source: str = "gt"):
        cfg = load_config(config_path)
        df = pd.read_csv(cfg.path("paths", "processed_root") / "manifest.csv")
        df = df[df.split == split]

        # rows with no surviving polygon carry class_idx == -1, which is an
        # invalid target. Drop them here so a stale config.exclude cannot
        # silently poison training.
        if drop_empty:
            before = len(df)
            df = df[(df.class_idx >= 0) & (df.bark_frac > 0)]
            if before != len(df):
                print(f"[{split}] dropped {before - len(df)} row(s) with no "
                      f"valid mask or label")

        self.df = df.reset_index(drop=True)
        self.fmt = str(cfg["preprocess"]["save_format"]).lower()
        self.size, self.margin = size, margin
        self.apply_clahe, self.apply_mask = apply_clahe, apply_mask
        # mask_source: "gt"   -> the ground-truth mask from build_masks
        #              "pred" -> the Stage-1 predicted mask (honest end-to-end;
        #                        falls back to gt if a pred file is missing)
        self.mask_source = mask_source
        self.proc_root = cfg.path("paths", "processed_root")
        self.split = split
        self.transform = build_transform(size, train)
        if self.transform is None:
            print("albumentations not installed — no augmentation or scale "
                  "jitter. pip install albumentations")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ip = PROJECT_ROOT / row.image_path
        mp = PROJECT_ROOT / row.mask_path

        if self.mask_source == "pred":
            pred_mp = self.proc_root / self.split / "pred_masks" / f"{row.stem}.png"
            if pred_mp.exists():
                mp = pred_mp
            # else: silently fall back to gt mask (mp unchanged)

        if self.fmt == "png":
            img, mask = read_image_rgb(ip), read_mask_png(mp)
        else:
            img = np.load(ip)
            # pred masks are always written as png even in npy mode
            if str(mp).endswith(".png"):
                mask = read_mask_png(mp)
            else:
                mask = np.load(mp).astype(np.uint8)

        # 1. crop to the trunk -> removes the absolute-area shortcut
        img, mask = crop_to_bbox(img, mask, self.margin)

        # 2. CLAHE before masking, so the black background does not distort
        #    the local histograms
        if self.apply_clahe:
            img = clahe_lab(img)

        # 3. geometric + photometric augmentation, image and mask in lockstep
        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        else:
            img = cv2.resize(img, (self.size, self.size), cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.size, self.size), cv2.INTER_NEAREST)

        # 4. zero the background
        if self.apply_mask:
            img = img * mask[:, :, None]

        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        m = np.ascontiguousarray(mask.astype(np.float32)[None])
        y = int(row.class_idx)

        try:
            import torch
            return torch.from_numpy(x), torch.from_numpy(m), y
        except ImportError:
            return x, m, y
