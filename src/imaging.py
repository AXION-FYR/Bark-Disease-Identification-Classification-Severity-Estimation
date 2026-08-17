"""
Aspect-ratio-preserving letterbox + resize.

Why letterbox rather than a plain square resize:
stripe canker's signature is vertical anisotropy. Squashing a tall trunk into a
square changes the aspect ratio and therefore the apparent orientation and
spacing of the streaks. Padding first keeps the geometry intact; the padded
region is background and is excluded by the bark mask anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class LetterboxParams:
    """Everything needed to map coordinates back to the original image."""
    orig_w: int
    orig_h: int
    scale: float        # applied to the original before padding
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    out_size: int

    def to_dict(self) -> dict:
        return asdict(self)


def letterbox_image(img: np.ndarray, size: int,
                    pad_value: int = 0) -> tuple[np.ndarray, LetterboxParams]:
    """Resize the long side to `size`, then pad the short side symmetrically."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_x, pad_y = size - new_w, size - new_h
    left, top = pad_x // 2, pad_y // 2
    right, bottom = pad_x - left, pad_y - top

    border_val = [pad_value] * (img.shape[2] if img.ndim == 3 else 1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=border_val)

    return out, LetterboxParams(w, h, scale, left, top, right, bottom, size)


def letterbox_mask(mask: np.ndarray, params: LetterboxParams) -> np.ndarray:
    """
    Apply the *same* geometric transform to a mask.

    INTER_NEAREST is mandatory: bilinear interpolation of a binary mask produces
    fractional labels along every boundary.
    """
    new_w = params.out_size - params.pad_left - params.pad_right
    new_h = params.out_size - params.pad_top - params.pad_bottom

    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = cv2.copyMakeBorder(
        resized, params.pad_top, params.pad_bottom,
        params.pad_left, params.pad_right,
        cv2.BORDER_CONSTANT, value=0,
    )
    return out


def read_image_rgb(path, apply_exif: bool = True) -> np.ndarray:
    """
    Read as RGB, applying EXIF orientation by default.

    This matters. Roboflow's annotation UI displays images EXIF-corrected, so
    the width/height in _annotations.coco.json — and therefore the polygon
    coordinates — are in the *rotated* frame. The file bytes on disk are often
    still in the camera's native frame with an orientation tag. Reading with
    cv2 alone ignores that tag and every polygon lands 90 degrees out.

    PIL's exif_transpose applies the tag; cv2 is the fallback for files whose
    EXIF was stripped on export.
    """
    if apply_exif:
        try:
            from PIL import Image, ImageOps
            with Image.open(str(path)) as im:
                im = ImageOps.exif_transpose(im)
                return np.asarray(im.convert("RGB"))
        except Exception:
            pass

    data = np.fromfile(str(path), dtype=np.uint8)      # unicode-safe on Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image_rgb(path, img_rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise IOError(f"could not encode image: {path}")
    buf.tofile(str(path))


def write_mask_png(path, mask01: np.ndarray) -> None:
    """Write a {0,1} mask as a single-channel 0/255 PNG."""
    ok, buf = cv2.imencode(".png", (mask01.astype(np.uint8) * 255))
    if not ok:
        raise IOError(f"could not encode mask: {path}")
    buf.tofile(str(path))


def read_mask_png(path) -> np.ndarray:
    """Read a 0/255 PNG back as a {0,1} uint8 mask."""
    data = np.fromfile(str(path), dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"could not decode mask: {path}")
    return (m > 127).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight (x0, y0, x1, y1) around non-zero pixels; None if mask is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.45,
                 colour: tuple[int, int, int] = (255, 40, 40)) -> np.ndarray:
    """Tinted fill + solid contour, for visual QC."""
    out = img_rgb.copy()
    m = mask.astype(bool)
    if m.any():
        tint = np.zeros_like(out)
        tint[:] = colour
        out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 2)
    return out
