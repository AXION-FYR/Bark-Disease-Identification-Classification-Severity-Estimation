"""
Step 16 — region-constrained patch refinement.

The Stage-3 decoder localises the diseased REGION well, but at 7x7 encoder
resolution it draws one smooth blob and cannot resolve the scattered sub-patches
of rough bark. This refiner recovers patches WITHOUT retraining.

Idea: the learned blob answers "WHERE on the trunk is disease" reliably. Inside
that region we then separate diseased from healthy tissue at full pixel
resolution with a classical, disease-specific rule:

  * rough bark  -> patches are darker, rougher islands scattered on lighter
                   bark. Use local-texture energy + intensity, thresholded
                   ONLY inside the blob (Otsu on the in-region pixels). This
                   fragments the blob into the real islands.
  * stripe canker -> lesion is already ~contiguous; the blob is a good fit, so
                   refinement is light (keep the region, mild cleanup).

The region constraint is what makes the classical step work — the same
thresholding failed as a standalone method earlier precisely because it had no
region to anchor to. This is a two-stage detector: learned region -> classical
patch, disease-conditioned.

Compares three masks against the 40 GT annotations:
  region  = the raw decoder blob (Stage 3)
  refined = region-constrained patch mask (this step)
  gradcam = baseline

Run:  python scripts/16_refine_patches.py \
          --lesion_ckpt outputs/lesion/lesion_film.pt \
          --stage2_ckpt outputs/cls/cls_mcse_allones.pt \
          --lesion_root D:/RESEARCH/Dataset/lesion_eval_coco
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402

from src.config import load_config, PROJECT_ROOT                      # noqa: E402
from src.coco_utils import find_annotation_file, load_coco, polygons_to_mask  # noqa: E402
from src.imaging import read_image_rgb, read_mask_png                 # noqa: E402
from src.dataset_cls import crop_to_bbox, clahe_lab                   # noqa: E402
from src.model_lesion import build_lesion_decoder                    # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def norm_stem(fn):
    m = re.match(r"^\s*([A-Za-z]+\s*\(\d+\))", fn)
    return (m.group(1) if m else Path(fn).stem).lower().replace(" ", "")


def iou_dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = inter / union if union else (1.0 if pred.sum() == gt.sum() == 0 else 0.0)
    dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0
    return float(iou), float(dice)


def prep(img_rgb, bark_mask, size, margin=0.05):
    img, m = crop_to_bbox(img_rgb, bark_mask, margin)
    disp = clahe_lab(img)
    disp_r = cv2.resize(disp, (size, size), interpolation=cv2.INTER_AREA)
    m_r = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    x = ((disp_r.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    return x, m_r, disp_r


def texture_energy(gray):
    """Local std-dev as a rough-texture measure (high on bumpy rough bark)."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (9, 9))
    sq = cv2.blur(g * g, (9, 9))
    var = np.clip(sq - mean * mean, 0, None)
    return np.sqrt(var)


def refine_rough_bark(disp_rgb, region, bark):
    """
    Split the region blob into rough-bark patches.
    Inside the region, diseased tissue is darker AND rougher than healthy bark.
    Otsu-threshold the combined cue on in-region pixels only.
    """
    gray = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray                                  # dark -> high
    tex = texture_energy(gray)
    tex = (tex / (tex.max() + 1e-6) * 255).astype(np.uint8)
    cue = cv2.addWeighted(inv, 0.5, tex, 0.5, 0)      # dark + rough

    reg = (region > 0) & (bark > 0)
    if reg.sum() < 30:
        return region.astype(np.uint8)

    vals = cue[reg]
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    patch = ((cue >= thr) & reg).astype(np.uint8)

    # morphological cleanup: drop specks, close tiny gaps
    patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    patch = cv2.morphologyEx(patch, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # remove components smaller than 0.05% of the crop
    n, lab, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=8)
    min_area = 0.0005 * patch.size
    keep = np.zeros_like(patch)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == c] = 1
    return keep


def refine_stripe(disp_rgb, region, bark):
    """
    Stripe canker is ~contiguous; the blob already fits. Light cleanup only:
    keep the region, remove specks, fill small holes.
    """
    reg = ((region > 0) & (bark > 0)).astype(np.uint8)
    reg = cv2.morphologyEx(reg, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesion_ckpt", required=True)
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--lesion_root", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--region_thr", type=float, default=0.3,
                    help="threshold on the decoder prob to define the region")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = PROJECT_ROOT / "outputs" / "lesion"
    out.mkdir(parents=True, exist_ok=True)

    st = torch.load(args.lesion_ckpt, map_location=device, weights_only=False)
    model = build_lesion_decoder(args.stage2_ckpt,
                                 class_conditioned=st.get("film", True)).to(device)
    model.load_state_dict(st["model"])
    model.eval()

    manifest = pd.read_csv(cfg.path("paths", "processed_root") / "manifest.csv")
    stem_row = {r.stem: r for _, r in manifest.iterrows()}
    name_stem = {norm_stem(r.file_name): r.stem for _, r in manifest.iterrows()}

    anns = []
    for sub in ["", "train", "valid", "test"]:
        f = Path(args.lesion_root) / sub if sub else Path(args.lesion_root)
        a = find_annotation_file(f)
        if a:
            anns.append((f, a))

    rows, gallery = [], []
    for folder, ann_path in anns:
        for rec in load_coco(ann_path):
            ns = norm_stem(rec.file_name)
            if ns not in name_stem:
                continue
            stem = name_stem[ns]
            row = stem_row[stem]
            if row.split != "test":
                continue

            img = read_image_rgb(PROJECT_ROOT / row.image_path)
            pm = cfg.path("paths", "processed_root") / "test" / "pred_masks" / f"{stem}.png"
            bark = read_mask_png(pm) if pm.exists() else read_mask_png(PROJECT_ROOT / row.mask_path)

            gt_full = polygons_to_mask(rec, h=img.shape[0], w=img.shape[1])
            x, m_r, disp = prep(img, bark, args.size)
            _, gt_c = crop_to_bbox(gt_full, bark, 0.05)
            gt = cv2.resize(gt_c, (args.size, args.size), interpolation=cv2.INTER_NEAREST)

            cls_idx = int(row.class_idx)
            with torch.no_grad():
                prob = torch.sigmoid(model(torch.tensor(x, device=device),
                                     torch.tensor([cls_idx], device=device)))[0, 0].cpu().numpy()
            region = (prob > args.region_thr).astype(np.uint8)

            if row.class_name == "Rough bark":
                refined = refine_rough_bark(disp, region, m_r)
            else:
                refined = refine_stripe(disp, region, m_r)

            ri, rd = iou_dice(region & (m_r > 0), gt)
            fi, fd = iou_dice(refined & (m_r > 0), gt)
            rows.append({"stem": stem, "class": row.class_name,
                         "region_iou": ri, "refined_iou": fi,
                         "region_dice": rd, "refined_dice": fd})
            gallery.append((disp, region, refined, gt, row.class_name))

    rdf = pd.DataFrame(rows)
    print("=" * 62)
    print("REGION (blob)  vs  REFINED (patches)")
    print("=" * 62)
    print(rdf.groupby("class")[["region_iou", "refined_iou",
                                "region_dice", "refined_dice"]].mean().round(4).to_string())
    print("-" * 62)
    print(f"OVERALL region  IoU {rdf.region_iou.mean():.4f}  "
          f"Dice {rdf.region_dice.mean():.4f}")
    print(f"        refined IoU {rdf.refined_iou.mean():.4f}  "
          f"Dice {rdf.refined_dice.mean():.4f}")
    d = rdf.refined_iou.mean() - rdf.region_iou.mean()
    print(f"        refined - region IoU: {d:+.4f}")

    # per-class verdict — refinement is meant to help ROUGH BARK
    rb = rdf[rdf["class"] == "Rough bark"]
    if len(rb):
        drb = rb.refined_iou.mean() - rb.region_iou.mean()
        print(f"\nrough bark: region {rb.region_iou.mean():.4f} -> "
              f"refined {rb.refined_iou.mean():.4f}  ({drb:+.4f})")
        print("  -> refinement helps rough bark" if drb > 0.01
              else "  -> refinement does NOT help; keep region-level for rough bark too")

    rdf.to_csv(out / "refine_per_image.csv", index=False)

    # gallery: image | region | refined | GT, a few rough-bark first
    gallery.sort(key=lambda g: 0 if g[4] == "Rough bark" else 1)
    k = min(6, len(gallery))
    fig, axes = plt.subplots(k, 4, figsize=(16, 4 * k))
    axes = np.atleast_2d(axes)
    titles = ["image", "region (blob)", "refined (patches)", "GT"]
    for r in range(k):
        disp, region, refined, gt, cls = gallery[r]
        panels = [disp, None, None, None]
        axes[r, 0].imshow(disp); axes[r, 0].set_ylabel(cls, fontsize=9)
        axes[r, 1].imshow(disp); axes[r, 1].imshow(region, alpha=0.45, cmap="autumn")
        axes[r, 2].imshow(disp); axes[r, 2].imshow(refined, alpha=0.45, cmap="autumn")
        axes[r, 3].imshow(disp); axes[r, 3].imshow(gt, alpha=0.45, cmap="autumn")
        for c in range(4):
            axes[r, c].set_title(titles[c], fontsize=9)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "refine_overlays.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\noverlays -> {out / 'refine_overlays.png'}")

    with open(out / "refine_metrics.json", "w") as fh:
        json.dump({"region_iou": float(rdf.region_iou.mean()),
                   "refined_iou": float(rdf.refined_iou.mean()),
                   "per_class": rdf.groupby("class")[["region_iou", "refined_iou"]]
                   .mean().round(4).to_dict("index")}, fh, indent=2)
    print(f"metrics -> {out / 'refine_metrics.json'}")


if __name__ == "__main__":
    main()
