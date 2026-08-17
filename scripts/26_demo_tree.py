"""
Step 26 — single-tree demo: point at ONE tree folder, get the full pipeline
result as one visualisation.

Give it a folder of photos of a single tree (any images; names don't matter).
It runs every stage — segment bark, classify disease, localise lesion, compute
QSI — on each photo, aggregates to a per-disease tree severity + treatment
stage, and saves ONE summary figure:

  * top: a headline card per disease (severity, stage, #views)
  * below: each photo with its lesion overlay, disease and per-photo QSI

Use this for a demo or viva: "here is a tree, here is what the system says."

Run:  python scripts/26_demo_tree.py --folder D:/RESEARCH/Dataset/multiview_trees/tree5 \
          --seg_ckpt outputs/seg/best.pt \
          --stage2_ckpt outputs/cls/cls_mcse_allones.pt \
          --lesion_ckpt outputs/lesion/lesion_film.pt
"""
from __future__ import annotations

import argparse
import json
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
from matplotlib.patches import Patch                                  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# same disease-conditioned damage weights as the QSI / tree scripts
DISEASE_WEIGHTS = {"Rough bark": (0.70, 0.30),
                   "stripecanker": (0.30, 0.70), "_default": (0.50, 0.50)}
STAGE_COLOR = {"Preventive": "#2ca25f", "Early control": "#fed976",
               "Active management": "#fd8d3c", "Severe outbreak": "#e31a1c"}


def texture_energy(gray):
    g = gray.astype(np.float32)
    m = cv2.blur(g, (9, 9))
    sq = cv2.blur(g * g, (9, 9))
    return np.sqrt(np.clip(sq - m * m, 0, None))


def damage_intensity(disp, bark, ref, disease=None):
    gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(disp, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    tex = texture_energy(gray)
    d_tex = np.clip((tex - ref["tex_mean"]) / (ref["tex_std"] + 1e-6), 0, None)
    d_dark = np.clip((ref["L_mean"] - L) / (ref["L_std"] + 1e-6), 0, None)
    wt, wd = DISEASE_WEIGHTS.get(disease, DISEASE_WEIGHTS["_default"])
    return np.clip((wt * d_tex + wd * d_dark) / 3.0, 0, 1) * (bark > 0)


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


DISEASE_BANDS = {
    "Rough bark":   {"prev": 0.30, "early": 0.50, "active": 0.80},
    "stripecanker": {"prev": 0.20, "early": 0.40, "active": 0.70},
    "_default":     {"prev": 0.05, "early": 0.15, "active": 0.50},
}


def treatment_stage(pct: float, disease: str) -> str:
    """% diseased area -> treatment stage via disease-specific bands."""
    b = DISEASE_BANDS.get(disease, DISEASE_BANDS["_default"])
    if pct < b["prev"]:   return "Preventive"
    if pct < b["early"]:  return "Early control"
    if pct < b["active"]: return "Active management"
    return "Severe outbreak"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="one tree's photo folder")
    ap.add_argument("--seg_ckpt", default="outputs/seg/best.pt")
    ap.add_argument("--stage2_ckpt", default="outputs/cls/cls_mcse_allones.pt")
    ap.add_argument("--lesion_ckpt", default="outputs/lesion/lesion_film.pt")
    ap.add_argument("--seg_size", type=int, default=512)
    ap.add_argument("--cls_size", type=int, default=224)
    ap.add_argument("--healthy_idx", type=int, default=1)
    ap.add_argument("--lesion_thr", type=float, default=0.5)
    ap.add_argument("--min_views", type=int, default=2)
    ap.add_argument("--girdle_spread", type=float, default=0.6)
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"folder not found: {folder}")

    photos = sorted([f for f in folder.iterdir()
                     if f.suffix.lower() in IMG_EXTS])
    if not photos:
        sys.exit(f"no images in {folder}")
    tree_name = folder.name
    print(f"tree '{tree_name}': {len(photos)} photos")

    classes = json.load(open(cfg.path("paths", "processed_root") / "classes.json"))
    idx_to_name = {v: k for k, v in classes["class_to_idx"].items()}

    # ---- load models
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

    ref_path = PROJECT_ROOT / "outputs" / "qsi" / "healthy_reference.json"
    if not ref_path.exists():
        sys.exit("outputs/qsi/healthy_reference.json missing — run 17_qsi.py once.")
    ref = json.load(open(ref_path))

    # ---- run pipeline per photo
    results = []
    for path in photos:
        img = read_image_rgb(path)
        seg_in, p = letterbox_image(img, args.seg_size)
        x = ((seg_in.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
        x = torch.tensor(np.ascontiguousarray(x.transpose(2, 0, 1))[None], device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=(device.type == "cuda")):
            bark_lb = (torch.sigmoid(seg(x))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        inner = bark_lb[p.pad_top:args.seg_size - p.pad_bottom,
                        p.pad_left:args.seg_size - p.pad_right]
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
            d = damage_intensity(disp_r, m_r, ref, disease)
            bpx = float((m_r > 0).sum())
            pct = float((lp > args.lesion_thr).sum() / bpx)
            qsi = float((lp * d).sum() / bpx)

        gray = cv2.cvtColor(disp_r, cv2.COLOR_RGB2GRAY)
        w = float((m_r > 0).mean()) * conf * sharpness(gray)
        results.append({"disp": disp_r, "lp": lp, "disease": disease,
                        "conf": conf, "qsi": qsi, "pct": pct, "weight": w})

    if not results:
        sys.exit("no photo produced a bark mask — check the images")

    # ---- aggregate per disease (weighted)
    summary = {}
    for dis in set(r["disease"] for r in results):
        if dis == idx_to_name[args.healthy_idx]:
            continue
        g = [r for r in results if r["disease"] == dis]
        if len(g) < args.min_views:
            continue
        w = np.array([max(r["weight"], 1e-6) for r in g])
        q = float((np.array([r["qsi"] for r in g]) * w).sum() / w.sum())
        pc = float((np.array([r["pct"] for r in g]) * w).sum() / w.sum())
        spread = len(g) / max(len(results), 1)
        sev_stage = treatment_stage(pc, dis)
        stage_order = ["Preventive", "Early control",
                       "Active management", "Severe outbreak"]
        idx = stage_order.index(sev_stage)
        girdling = spread >= args.girdle_spread
        action = stage_order[min(idx + 1, 3)] if girdling and idx < 3 else sev_stage
        summary[dis] = {"qsi": q, "pct": pc, "n": len(g),
                        "spread": spread, "stage": action,
                        "escalated": action != sev_stage}

    print(f"\n=== {tree_name} ===")
    if not summary:
        print("  No significant disease detected -> Preventive")
    for dis, s in summary.items():
        esc = " (girdling escalated)" if s["escalated"] else ""
        print(f"  {dis}: {s['pct']*100:.0f}% area  "
              f"spread {s['spread']*100:.0f}%  [{s['stage']}]{esc}  "
              f"(QSI {s['qsi']:.3f}, {s['n']} views)")

    # ---- one summary figure
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    header_h = 1.6
    fig = plt.figure(figsize=(16, header_h + 3 * nrows))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[header_h] + [3] * nrows)

    # header: one card per disease
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
    title = f"Tree: {tree_name}    ({n} views)"
    ax0.text(0.5, 0.92, title, ha="center", va="top", fontsize=15, fontweight="bold",
             transform=ax0.transAxes)
    if summary:
        cards = list(summary.items())
        for i, (dis, s) in enumerate(cards):
            x0 = 0.5 - len(cards) * 0.23 + i * 0.46
            col = STAGE_COLOR.get(s["stage"], "#cccccc")
            ax0.add_patch(plt.Rectangle((x0, 0.05), 0.42, 0.6, transform=ax0.transAxes,
                                        facecolor=col, alpha=0.35, edgecolor=col, lw=2))
            ax0.text(x0 + 0.21, 0.50, dis, ha="center", va="center", fontsize=13,
                     fontweight="bold", transform=ax0.transAxes)
            ax0.text(x0 + 0.21, 0.32, f"{s['stage']}", ha="center", va="center",
                     fontsize=12, transform=ax0.transAxes)
            ax0.text(x0 + 0.21, 0.16, f"QSI {s['qsi']:.3f} · {s['pct']*100:.0f}% bark · {s['n']} views",
                     ha="center", va="center", fontsize=9, transform=ax0.transAxes)
    else:
        ax0.text(0.5, 0.35, "No significant disease  →  Preventive",
                 ha="center", va="center", fontsize=13, transform=ax0.transAxes,
                 color="#2ca25f", fontweight="bold")

    # per-photo panels
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        ax.imshow(r["disp"])
        if r["qsi"] > 0:
            ax.imshow(r["lp"], alpha=0.45, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"{r['disease']}  ({r['conf']:.2f})\nQSI {r['qsi']:.3f}", fontsize=8)
        ax.axis("off")

    legend = [Patch(facecolor=STAGE_COLOR[s], label=s) for s in STAGE_COLOR]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    out_path = Path(args.out) if args.out else (PROJECT_ROOT / "outputs" / "tree" / f"demo_{tree_name}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsummary figure -> {out_path}")


if __name__ == "__main__":
    main()
