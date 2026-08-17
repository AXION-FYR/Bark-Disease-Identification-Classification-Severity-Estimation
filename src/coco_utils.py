"""
COCO segmentation parsing + polygon rasterisation.

Deliberately avoids pycocotools for the common case (polygon segmentation),
because pycocotools is awkward to install on Windows. pycocotools is imported
lazily only if compressed-RLE annotations are encountered.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# --------------------------------------------------------------------------
# name normalisation
# --------------------------------------------------------------------------
def norm_name(s: str) -> str:
    """'Rough bark' -> 'roughbark'; 'stripe_canker' -> 'stripecanker'."""
    return re.sub(r"[\s_\-]+", "", str(s)).lower()


# --------------------------------------------------------------------------
# data containers
# --------------------------------------------------------------------------
@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    polygons: list[np.ndarray] = field(default_factory=list)   # each (N,2) float32
    rles: list[dict] = field(default_factory=list)             # raw RLE dicts, if any
    areas: list[float] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)

    @property
    def n_ann(self) -> int:
        return len(self.polygons) + len(self.rles)


@dataclass
class CocoData:
    records: dict[int, ImageRecord]
    categories: dict[int, str]          # id -> raw name (dummy classes removed)
    source: Path

    def __iter__(self):
        return iter(self.records.values())

    def __len__(self):
        return len(self.records)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def find_annotation_file(folder: Path) -> Path | None:
    """Locate _annotations.coco.json (or any *.json holding COCO keys) in folder."""
    direct = folder / "_annotations.coco.json"
    if direct.exists():
        return direct
    for cand in sorted(folder.glob("*.json")):
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                head = json.load(fh)
        except Exception:
            continue
        if {"images", "annotations", "categories"} <= set(head):
            return cand
    return None


def load_coco(ann_path: Path, min_area_frac: float = 0.0) -> CocoData:
    with open(ann_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    # --- categories: drop ids that never appear in annotations (Roboflow dummy)
    used_cat_ids = {a["category_id"] for a in raw.get("annotations", [])}
    categories = {
        c["id"]: c["name"]
        for c in raw.get("categories", [])
        if c["id"] in used_cat_ids
    }

    records: dict[int, ImageRecord] = {}
    for img in raw["images"]:
        records[img["id"]] = ImageRecord(
            image_id=img["id"],
            file_name=img["file_name"],
            width=int(img["width"]),
            height=int(img["height"]),
        )

    by_img: dict[int, list[dict]] = defaultdict(list)
    for a in raw.get("annotations", []):
        by_img[a["image_id"]].append(a)

    for img_id, anns in by_img.items():
        rec = records.get(img_id)
        if rec is None:
            continue
        img_area = float(rec.width * rec.height)
        for a in anns:
            if a.get("iscrowd", 0) and isinstance(a.get("segmentation"), dict):
                rec.rles.append(a["segmentation"])
                rec.areas.append(float(a.get("area", 0.0)))
                rec.class_names.append(categories.get(a["category_id"], "unknown"))
                continue

            seg = a.get("segmentation")
            if not seg:
                continue

            # dict segmentation without iscrowd -> still RLE
            if isinstance(seg, dict):
                rec.rles.append(seg)
                rec.areas.append(float(a.get("area", 0.0)))
                rec.class_names.append(categories.get(a["category_id"], "unknown"))
                continue

            area = float(a.get("area", 0.0))
            if img_area > 0 and min_area_frac > 0 and area / img_area < min_area_frac:
                continue

            for ring in seg:
                if len(ring) < 6:          # need >= 3 points
                    continue
                pts = np.asarray(ring, dtype=np.float32).reshape(-1, 2)
                rec.polygons.append(pts)
                rec.areas.append(area)
                rec.class_names.append(categories.get(a["category_id"], "unknown"))

    return CocoData(records=records, categories=categories, source=ann_path)


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------
def polygons_to_mask(rec: ImageRecord, h: int | None = None,
                     w: int | None = None) -> np.ndarray:
    """
    Union of all polygons for this image -> uint8 mask {0,1} at (h, w).

    All classes are collapsed to a single foreground label: for Stage 1 the task
    is bark-vs-background. The disease class is carried separately.
    """
    h = int(h or rec.height)
    w = int(w or rec.width)
    mask = np.zeros((h, w), dtype=np.uint8)

    if rec.polygons:
        polys = [np.round(p).astype(np.int32) for p in rec.polygons]
        cv2.fillPoly(mask, polys, color=1)

    if rec.rles:
        mask |= _decode_rles(rec.rles, h, w)

    return mask


def _decode_rles(rles: list[dict], h: int, w: int) -> np.ndarray:
    try:
        from pycocotools import mask as mask_util  # noqa: PLC0415
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "This export contains RLE segmentation. Install pycocotools "
            "(`pip install pycocotools`) or re-export from Roboflow as "
            "polygon COCO segmentation."
        ) from exc

    out = np.zeros((h, w), dtype=np.uint8)
    for rle in rles:
        r = dict(rle)
        if isinstance(r.get("counts"), list):
            r = mask_util.frPyObjects(r, h, w)
        out |= mask_util.decode(r).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# per-image class label (majority polygon area)
# --------------------------------------------------------------------------
def image_class(rec: ImageRecord) -> tuple[str, bool]:
    """
    Returns (class_name, is_mixed).

    is_mixed is True when the image carries polygons of more than one class —
    a real trunk with two co-occurring diseases, or an annotation error.
    """
    if not rec.class_names:
        return ("__none__", False)

    per_class: dict[str, float] = defaultdict(float)
    for name, area in zip(rec.class_names, rec.areas):
        per_class[name] += max(area, 1.0)

    winner = max(per_class.items(), key=lambda kv: kv[1])[0]
    return (winner, len(per_class) > 1)


def resolve_class_order(found: list[str], configured: list[str] | None) -> list[str]:
    """Map configured canonical order onto the names actually present."""
    if not configured:
        return sorted(found, key=norm_name)

    lookup = {norm_name(f): f for f in found}
    ordered = [lookup[norm_name(c)] for c in configured if norm_name(c) in lookup]
    leftovers = [f for f in found if f not in ordered]
    return ordered + sorted(leftovers, key=norm_name)
