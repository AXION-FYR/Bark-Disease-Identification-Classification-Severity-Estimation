"""
Step 15 — evaluate the lesion decoder.  This produces your Stage 3 numbers.

Three things, in order of importance:

  1. LESION IoU / Dice against the 40 hand-annotated masks, versus a
     thresholded Grad-CAM baseline. This is the quantitative claim: does the
     weakly-supervised decoder localise lesions better than the obvious
     cheap alternative? Reported per class and overall.

  2. HEALTHY-ANCHOR sanity check (needs no lesion labels): mean lesion
     probability inside bark on healthy TEST trunks. Should be near zero. This
     is measurable on every healthy image and is strong evidence the method
     learned "healthy = no lesion" rather than lighting up on texture.

  3. QUALITATIVE maps: overlays of predicted lesion heatmap on the trunk, for
     a few images per class, plus the GT lesion polygons where available.

Grad-CAM baseline uses the Stage-2 classifier's final conv features w.r.t. the
predicted class — the standard weakly-supervised localisation method your work
must beat to claim novelty.

Run:  python scripts/15_eval_lesion.py \
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
import torch.nn.functional as F                                       # noqa: E402

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


def preprocess(img_rgb, bark_mask, size, margin=0.05):
    img, m = crop_to_bbox(img_rgb, bark_mask, margin)
    img = clahe_lab(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    x = ((img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    return x, m, img  # tensor-ready, mask@size, display image


def best_threshold(probs, gts, masks):
    """Pick the single global threshold maximising mean IoU over the eval set."""
    best_t, best_i = 0.5, -1
    for t in np.linspace(0.1, 0.9, 17):
        ious = []
        for p, g, mk in zip(probs, gts, masks):
            ious.append(iou_dice((p > t) & (mk > 0), g)[0])
        mi = float(np.mean(ious))
        if mi > best_i:
            best_i, best_t = mi, t
    return best_t


def gradcam(stage2_model, x, cls_idx, device):
    """Grad-CAM on the Stage-2 encoder's last feature map w.r.t. class cls_idx."""
    feats, grads = {}, {}

    # The classifier's encoder is a timm features_only model returning a list;
    # hook the whole encoder module and grab the last feature map it outputs.
    enc = stage2_model.encoder

    def fwd_hook(m, i, o):
        fm = o[-1] if isinstance(o, (list, tuple)) else o
        fm.retain_grad()
        feats["v"] = fm

    h1 = enc.register_forward_hook(fwd_hook)

    stage2_model.zero_grad()
    xt = torch.tensor(x, device=device)
    mask_ones = torch.ones(1, 1, x.shape[2], x.shape[3], device=device)
    logits = stage2_model(xt, mask_ones)
    logits[0, cls_idx].backward()

    f = feats["v"]
    g = f.grad
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    h1.remove()
    return cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesion_ckpt", required=True)
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--lesion_root", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--mask_source", default="pred", choices=["pred", "gt"])
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determinism: the eval must give the SAME numbers every run. Fix seeds and
    # disable nondeterministic cuDNN kernels; classifier/decoder go to eval mode
    # so dropout and BN are frozen.
    import random as _random
    _random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = PROJECT_ROOT / "outputs" / "lesion"
    out.mkdir(parents=True, exist_ok=True)

    # ---- load lesion decoder
    st = torch.load(args.lesion_ckpt, map_location=device, weights_only=False)
    model = build_lesion_decoder(args.stage2_ckpt,
                                 class_conditioned=st.get("film", True)).to(device)
    model.load_state_dict(st["model"])
    model.eval()

    # ---- load Stage-2 classifier for Grad-CAM baseline
    from src.model_cls import build_classifier
    s2 = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
    clf = build_classifier(variant=s2.get("variant", "mcse_allones"), pretrained=False).to(device)
    clf.load_state_dict(s2["model"])
    clf.eval()

    manifest = pd.read_csv(cfg.path("paths", "processed_root") / "manifest.csv")
    stem_to_row = {r.stem: r for _, r in manifest.iterrows()}
    name_to_stem = {norm_stem(r.file_name): r.stem for _, r in manifest.iterrows()}

    # ---- gather lesion GT
    anns = []
    for sub in ["", "train", "valid", "test"]:
        f = Path(args.lesion_root) / sub if sub else Path(args.lesion_root)
        a = find_annotation_file(f)
        if a:
            anns.append((f, a))
    if not anns:
        sys.exit(f"no lesion COCO under {args.lesion_root}")

    ours_p, cam_p, gts, masks, metas = [], [], [], [], []

    for folder, ann_path in anns:
        coco = load_coco(ann_path)
        for rec in coco:
            ns = norm_stem(rec.file_name)
            if ns not in name_to_stem:
                continue
            stem = name_to_stem[ns]
            row = stem_to_row[stem]
            if row.split != "test":
                continue

            # original image + bark mask (predicted, honest)
            img = read_image_rgb(PROJECT_ROOT / row.image_path)
            if args.mask_source == "pred":
                pm = cfg.path("paths", "processed_root") / "test" / "pred_masks" / f"{stem}.png"
                bark = read_mask_png(pm) if pm.exists() else read_mask_png(PROJECT_ROOT / row.mask_path)
            else:
                bark = read_mask_png(PROJECT_ROOT / row.mask_path)

            # lesion GT rasterised at the processed (letterboxed) resolution,
            # then cropped/resized the SAME way as the image
            gt_full = polygons_to_mask(rec, h=img.shape[0], w=img.shape[1])

            x, m_at_size, disp = preprocess(img, bark, args.size)
            # transform GT identically: crop to bark bbox, resize
            _, gt_c = crop_to_bbox(gt_full, bark, 0.05)
            gt_at = cv2.resize(gt_c, (args.size, args.size), interpolation=cv2.INTER_NEAREST)

            cls_idx = int(row.class_idx)
            with torch.no_grad():
                logit = model(torch.tensor(x, device=device),
                              torch.tensor([cls_idx], device=device))
                prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
            cam = gradcam(clf, x, cls_idx, device)

            ours_p.append(prob)
            cam_p.append(cam)
            gts.append(gt_at)
            masks.append(m_at_size)
            metas.append({"stem": stem, "class": row.class_name, "disp": disp})

    if not ours_p:
        sys.exit("no matched test-split lesion annotations found")

    n = len(ours_p)
    print(f"evaluating on {n} annotated test image(s)")

    # PRIMARY metric: fixed thresholds. Deterministic and reviewer-preferred.
    # The tuned threshold is reported too, but the headline number a thesis
    # quotes should not depend on a search that can flip between near-tied
    # values (which made stripe-canker IoU unstable run-to-run).
    print("\nfixed-threshold IoU (deterministic, primary):")
    for tf in [0.3, 0.5]:
        oi = np.mean([iou_dice((ours_p[i] > tf) & (masks[i] > 0), gts[i])[0]
                      for i in range(n)])
        ci = np.mean([iou_dice((cam_p[i] > tf) & (masks[i] > 0), gts[i])[0]
                      for i in range(n)])
        print(f"  @ {tf}:  ours {oi:.4f}   gradcam {ci:.4f}")

    # tuned threshold (secondary, may vary slightly at the search floor)
    t_ours = best_threshold(ours_p, gts, masks)
    t_cam = best_threshold(cam_p, gts, masks)
    print(f"\ntuned threshold (secondary)  ours {t_ours:.2f}  gradcam {t_cam:.2f}")

    # per-image table at the FIXED 0.5 threshold (deterministic). This is what
    # the metrics json and summary report, so numbers do not move run-to-run.
    T = 0.5
    rows = []
    for i, meta in enumerate(metas):
        oi, od = iou_dice((ours_p[i] > T) & (masks[i] > 0), gts[i])
        ci, cd = iou_dice((cam_p[i] > T) & (masks[i] > 0), gts[i])
        rows.append({"stem": meta["stem"], "class": meta["class"],
                     "ours_iou": oi, "ours_dice": od,
                     "cam_iou": ci, "cam_dice": cd})
    rdf = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("LESION LOCALISATION — ours vs Grad-CAM")
    print("=" * 60)
    summary = rdf.groupby("class")[["ours_iou", "cam_iou", "ours_dice", "cam_dice"]].mean()
    print(summary.round(4).to_string())
    print("-" * 60)
    print(f"OVERALL  ours IoU {rdf.ours_iou.mean():.4f}  Dice {rdf.ours_dice.mean():.4f}")
    print(f"         cam  IoU {rdf.cam_iou.mean():.4f}  Dice {rdf.cam_dice.mean():.4f}")
    win = rdf.ours_iou.mean() - rdf.cam_iou.mean()
    print(f"         ours - cam IoU: {win:+.4f}  "
          f"({'ours wins' if win > 0 else 'Grad-CAM wins — report honestly'})")

    rdf.to_csv(out / "lesion_eval_per_image.csv", index=False)

    # qualitative overlays — choose the most ILLUSTRATIVE examples, not the
    # first ones. An example shows localisation well when the GT lesion is a
    # MODERATE fraction of the bark (not ~0%, not ~100%): if the lesion covers
    # the whole trunk, "ours" and "GT" both span everything and the viewer
    # can't see that the model localises. We also prefer a mix of both diseases.
    def gt_frac(i):
        m = masks[i] > 0
        return float((gts[i][m] > 0).mean()) if m.sum() else 0.0

    scored = []
    for i in range(n):
        f = gt_frac(i)
        # ideal coverage ~0.15-0.6 of bark; score by closeness to that band
        if f < 0.03:
            s = -1.0                      # basically no lesion — skip
        elif f > 0.85:
            s = 0.2                       # whole-trunk severe — deprioritise
        else:
            s = 1.0 - abs(f - 0.35)       # peak around 35% coverage
        scored.append((s, i))

    # pick up to 6, balancing the two diseases
    order = [i for _, i in sorted(scored, key=lambda t: -t[0])]
    rb_idx = [i for i in order if metas[i]["class"] == "Rough bark"]
    sc_idx = [i for i in order if metas[i]["class"] != "Rough bark"]
    pick = []
    while len(pick) < min(6, n) and (rb_idx or sc_idx):
        if rb_idx:
            pick.append(rb_idx.pop(0))
        if len(pick) < min(6, n) and sc_idx:
            pick.append(sc_idx.pop(0))

    k = len(pick)
    # 4 columns: original | ours | Grad-CAM | GT — a direct visual comparison
    fig, axes = plt.subplots(k, 4, figsize=(16, 4 * k))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(pick):
        disp = metas[i]["disp"]
        m = masks[i] > 0
        axes[row, 0].imshow(disp)
        axes[row, 0].set_title(f"{metas[i]['class']}", fontsize=9)
        axes[row, 1].imshow(disp)
        axes[row, 1].imshow(ours_p[i] * m, alpha=0.5, cmap="jet", vmin=0, vmax=1)
        oi = iou_dice((ours_p[i] > 0.5) & m, gts[i])[0]
        axes[row, 1].set_title(f"ours  (IoU {oi:.2f})", fontsize=9)
        axes[row, 2].imshow(disp)
        axes[row, 2].imshow(cam_p[i] * m, alpha=0.5, cmap="jet", vmin=0, vmax=1)
        ci = iou_dice((cam_p[i] > 0.5) & m, gts[i])[0]
        axes[row, 2].set_title(f"Grad-CAM  (IoU {ci:.2f})", fontsize=9)
        axes[row, 3].imshow(disp)
        axes[row, 3].imshow(gts[i], alpha=0.4, cmap="Reds")
        axes[row, 3].set_title("GT lesion", fontsize=9)
        for c in range(4):
            axes[row, c].axis("off")
    fig.tight_layout()
    fig.savefig(out / "lesion_overlays.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\noverlays (ours vs Grad-CAM vs GT) -> {out / 'lesion_overlays.png'} "
          f"({k} illustrative examples)")

    # healthy-anchor sanity on ALL healthy test images (no lesion labels needed)
    healthy = manifest[(manifest.split == "test") &
                       (manifest.class_name == "healthy bark")]
    hp = []
    for _, row in healthy.iterrows():
        img = read_image_rgb(PROJECT_ROOT / row.image_path)
        pm = cfg.path("paths", "processed_root") / "test" / "pred_masks" / f"{row.stem}.png"
        bark = read_mask_png(pm) if pm.exists() else read_mask_png(PROJECT_ROOT / row.mask_path)
        x, m_at, _ = preprocess(img, bark, args.size)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(x, device=device),
                                 torch.tensor([int(row.class_idx)], device=device)))[0, 0].cpu().numpy()
        if m_at.sum():
            hp.append(float(prob[m_at > 0].mean()))
    if hp:
        print(f"\nHEALTHY-ANCHOR check: mean lesion prob inside bark on "
              f"{len(hp)} healthy test trunks = {np.mean(hp):.4f} "
              f"(want near 0; this needs no lesion labels)")

    with open(out / "lesion_metrics.json", "w") as fh:
        json.dump({"n": n, "threshold": T, "threshold_tuned_ours": t_ours,
                   "ours_iou": float(rdf.ours_iou.mean()),
                   "ours_dice": float(rdf.ours_dice.mean()),
                   "cam_iou": float(rdf.cam_iou.mean()),
                   "cam_dice": float(rdf.cam_dice.mean()),
                   "healthy_anchor_mean": float(np.mean(hp)) if hp else None,
                   "per_class": summary.round(4).to_dict("index")}, fh, indent=2)
    print(f"metrics -> {out / 'lesion_metrics.json'}")


if __name__ == "__main__":
    main()
