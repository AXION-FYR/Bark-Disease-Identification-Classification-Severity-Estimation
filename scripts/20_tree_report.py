"""
Step 20 — per-tree visual report.

Builds one figure for a single tree: every photo of that tree with its lesion
heatmap overlaid, its predicted disease, per-photo QSI and view weight, plus a
header summarising the tree's aggregated per-disease severity.

This is the figure that shows the whole pipeline at a glance for one tree —
segmentation region, disease call, lesion localisation, and how the ~15 views
combine into a tree severity.

Run:  python scripts/20_tree_report.py --tree 5 \
          --tree_root D:/RESEARCH/Dataset/multiview_trees \
          --seg_ckpt outputs/seg/best.pt \
          --stage2_ckpt outputs/cls/cls_mcse_allones.pt \
          --lesion_ckpt outputs/lesion/lesion_film.pt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402

from src.config import load_config, PROJECT_ROOT                      # noqa: E402
from src.imaging import letterbox_image, read_image_rgb              # noqa: E402
from src.dataset_cls import crop_to_bbox, clahe_lab                   # noqa: E402
from src.dataset_cls import IMAGENET_MEAN, IMAGENET_STD               # noqa: E402
from src.model_cls import build_classifier                           # noqa: E402
from src.model_lesion import build_lesion_decoder                    # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _texture_energy(gray):
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (9, 9))
    sq = cv2.blur(g * g, (9, 9))
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


def _damage_intensity(disp, bark, ref):
    gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(disp, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    tex = _texture_energy(gray)
    d_tex = np.clip((tex - ref["tex_mean"]) / (ref["tex_std"] + 1e-6), 0, None)
    d_dark = np.clip((ref["L_mean"] - L) / (ref["L_std"] + 1e-6), 0, None)
    return (np.clip((0.5 * d_tex + 0.5 * d_dark) / 3.0, 0, 1) * (bark > 0)).astype(np.float32)


def parse_tp(fn):
    m = re.match(r"^t(\d+)_p(\d+)", fn, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=int, required=True, help="tree number, e.g. 5")
    ap.add_argument("--tree_root", required=True)
    ap.add_argument("--seg_ckpt", default="outputs/seg/best.pt")
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--lesion_ckpt", required=True)
    ap.add_argument("--seg_size", type=int, default=512)
    ap.add_argument("--cls_size", type=int, default=224)
    ap.add_argument("--healthy_idx", type=int, default=1)
    ap.add_argument("--lesion_thr", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = PROJECT_ROOT / "outputs" / "tree"
    out.mkdir(parents=True, exist_ok=True)

    classes = json.load(open(cfg.path("paths", "processed_root") / "classes.json"))
    idx_to_name = {v: k for k, v in classes["class_to_idx"].items()}

    # locate this tree's photos
    root = Path(args.tree_root)
    photos = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            tp = parse_tp(f.name) if f.suffix.lower() in IMG_EXTS else None
            if tp and tp[0] == args.tree:
                photos.append((tp[1], f))
    photos.sort()
    if not photos:
        sys.exit(f"no photos found for tree {args.tree}")
    print(f"tree {args.tree}: {len(photos)} photos")

    # models
    import segmentation_models_pytorch as smp
    seg_st = torch.load(PROJECT_ROOT / args.seg_ckpt, map_location=device, weights_only=False)
    seg = smp.Unet(encoder_name=seg_st.get("encoder", "efficientnet-b0"),
                   encoder_weights=None, in_channels=3, classes=1).to(device)
    seg.load_state_dict(seg_st["model"]); seg.eval()

    s2 = torch.load(PROJECT_ROOT / args.stage2_ckpt, map_location=device, weights_only=False)
    clf = build_classifier(variant=s2.get("variant", "mcse_allones"), pretrained=False).to(device)
    clf.load_state_dict(s2["model"]); clf.eval()

    les_st = torch.load(PROJECT_ROOT / args.lesion_ckpt, map_location=device, weights_only=False)
    lesion = build_lesion_decoder(str(PROJECT_ROOT / args.stage2_ckpt),
                                  class_conditioned=les_st.get("film", True)).to(device)
    lesion.load_state_dict(les_st["model"]); lesion.eval()

    ref = json.load(open(PROJECT_ROOT / "outputs" / "qsi" / "healthy_reference.json"))

    results = []
    for pnum, path in photos:
        img = read_image_rgb(path)
        seg_in, params = letterbox_image(img, args.seg_size)
        x = ((seg_in.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
        x = torch.tensor(np.ascontiguousarray(x.transpose(2, 0, 1))[None], device=device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                bark_lb = (torch.sigmoid(seg(x))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        inner = bark_lb[params.pad_top:args.seg_size - params.pad_bottom,
                        params.pad_left:args.seg_size - params.pad_right]
        bark = cv2.resize(inner, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if bark.sum() < 100:
            continue

        crop, m = crop_to_bbox(img, bark, 0.05)
        disp = clahe_lab(crop)
        disp_r = cv2.resize(disp, (args.cls_size, args.cls_size), interpolation=cv2.INTER_AREA)
        m_r = cv2.resize(m, (args.cls_size, args.cls_size), interpolation=cv2.INTER_NEAREST)
        cx = ((disp_r.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
        cx = torch.tensor(np.ascontiguousarray(cx.transpose(2, 0, 1))[None], device=device)
        mm = torch.tensor(m_r.astype(np.float32)[None, None], device=device)

        with torch.no_grad():
            probs = torch.softmax(clf(cx * mm, mm), 1)[0].cpu().numpy()
        cls_idx = int(probs.argmax()); conf = float(probs[cls_idx])
        disease = idx_to_name[cls_idx]

        if cls_idx == args.healthy_idx:
            lp = np.zeros((args.cls_size, args.cls_size), np.float32)
            pct = qsi = 0.0
        else:
            with torch.no_grad():
                lp = torch.sigmoid(lesion(cx, torch.tensor([cls_idx], device=device)))[0, 0].cpu().numpy() * (m_r > 0)
            d = _damage_intensity(disp_r, m_r, ref)
            bark_px = float((m_r > 0).sum())
            pct = float((lp > args.lesion_thr).sum() / bark_px)
            qsi = float((lp * d).sum() / bark_px)

        results.append({"photo": pnum, "disp": disp_r, "lp": lp, "mask": m_r,
                        "disease": disease, "conf": conf, "pct": pct, "qsi": qsi})

    # aggregate for the header
    agg = {}
    for r in results:
        if r["disease"] == idx_to_name[args.healthy_idx]:
            continue
        agg.setdefault(r["disease"], []).append(r)
    header = f"Tree {args.tree:02d} — {len(results)} views\n"
    for dis, rs in agg.items():
        w = np.array([max(x["conf"], 0.01) for x in rs])
        mq = float((np.array([x["qsi"] for x in rs]) * w).sum() / w.sum())
        mp = float((np.array([x["pct"] for x in rs]) * w).sum() / w.sum())
        header += f"{dis}: {mp*100:.1f}% bark, QSI {mq:.3f} ({len(rs)} views)   "

    # figure: one row per photo — left = image+disease, right = lesion overlay
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4 * n))
    axes = np.atleast_2d(axes)

    for i, r in enumerate(results):
        axes[i, 0].imshow(r["disp"])
        axes[i, 0].set_title(f"p{r['photo']:02d}  {r['disease']}  (conf {r['conf']:.2f})",
                             fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(r["disp"])
        axes[i, 1].imshow(r["lp"], alpha=0.5, cmap="jet", vmin=0, vmax=1)
        axes[i, 1].set_title(f"lesion — QSI {r['qsi']:.3f},  {r['pct']*100:.0f}% bark",
                             fontsize=9)
        axes[i, 1].axis("off")

    fig.suptitle(header, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out / f"tree_{args.tree:02d}_report.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
