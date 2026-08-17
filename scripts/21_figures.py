"""
Step 21 — publication-ready results figures.

Renders the visual results panel from files the pipeline already produced.
Nothing is recomputed except (optionally) the Stage-2 confusion matrix if it
isn't already saved.

Figures written to outputs/figures/:
  fig_stage1_iou.png         segmentation IoU/Dice per class (bars)
  fig_stage2_confusion.png   classification confusion matrix (best model)
  fig_stage2_ablation.png    accuracy per variant with error bars over seeds
  fig_stage3_lesion.png      lesion IoU: ours vs Grad-CAM, per class
  fig_stage4_qsi.png         QSI distribution by class (box) — the separation
  fig_overview.png           one combined panel of all of the above

Run:  python scripts/21_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, PROJECT_ROOT                      # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

C1, C2, C3 = "#2c7fb8", "#7fcdbb", "#edf8b1"     # a calm 3-colour palette
ACCENT = "#d95f02"


def _load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def fig_stage1(out, cfg):
    m = _load_json(PROJECT_ROOT / "outputs" / "seg" / "test_metrics.json")
    if not m or "test" not in m or "per_class" not in m["test"]:
        return None
    pc = m["test"]["per_class"]
    classes = list(pc.keys())
    ious = [pc[c]["iou"] for c in classes]
    dices = [pc[c]["dice"] for c in classes]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(classes))
    ax.bar(x - 0.2, ious, 0.4, label="IoU", color=C1)
    ax.bar(x + 0.2, dices, 0.4, label="Dice", color=C2)
    ax.axhline(m["test"]["iou_mean"], ls="--", c=ACCENT,
               label=f"mean IoU {m['test']['iou_mean']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("Stage 1 — Bark segmentation (test)")
    ax.legend(fontsize=8)
    p = out / "fig_stage1_iou.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_confusion(out, cfg):
    full = _load_json(PROJECT_ROOT / "outputs" / "cls" / "ablation_full.json")
    cm = None
    if full:
        # pick the best variant's confusion matrix (highest test_acc)
        best = max(full, key=lambda r: r.get("test_acc", 0))
        cm = np.array(best.get("cm", []))
    if cm is None or cm.size == 0:
        return None

    classes = ["Rough bark", "healthy", "stripe canker"]
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]*100:.0f}%",
                    ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=8); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"Stage 2 — Confusion matrix (acc {acc:.3f})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    p = out / "fig_stage2_confusion.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_ablation(out, cfg):
    df = None
    try:
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "cls" / "ablation.csv")
    except Exception:
        return None
    df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
    g = df.groupby("variant").test_acc.agg(["mean", "std"]).fillna(0)
    order = [v for v in ["plain", "masked", "concat", "se", "mcse", "mcse_allones"]
             if v in g.index]
    g = g.reindex(order)
    pretty = {"mcse_allones": "dual+SE", "mcse": "mask-SE", "se": "SE",
              "concat": "+texture", "masked": "masked", "plain": "baseline"}

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["std"], capsize=4, color=C1)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(v, v) for v in g.index], fontsize=8, rotation=20)
    ax.set_ylim(0.7, 1.0); ax.set_ylabel("test accuracy")
    ax.axhline(0.408, ls=":", c="grey", label="majority baseline")
    ax.set_title("Stage 2 — Ablation (mean ± std over seeds)")
    ax.legend(fontsize=8)
    p = out / "fig_stage2_ablation.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_lesion(out, cfg):
    m = _load_json(PROJECT_ROOT / "outputs" / "lesion" / "lesion_metrics_film.json")
    if not m:
        m = _load_json(PROJECT_ROOT / "outputs" / "lesion" / "lesion_metrics.json")
    if not m or "per_class" not in m:
        return None
    pc = m["per_class"]
    classes = list(pc.keys())
    ours = [pc[c]["ours_iou"] for c in classes]
    cam = [pc[c]["cam_iou"] for c in classes]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(classes))
    ax.bar(x - 0.2, ours, 0.4, label="ours (weakly supervised)", color=C1)
    ax.bar(x + 0.2, cam, 0.4, label="Grad-CAM", color=C2)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("lesion IoU")
    ax.set_title(f"Stage 3 — Lesion localisation (overall ours "
                 f"{m['ours_iou']:.3f} vs cam {m['cam_iou']:.3f})")
    ax.legend(fontsize=8)
    p = out / "fig_stage3_lesion.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_qsi(out, cfg):
    try:
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "qsi" / "qsi_test.csv")
    except Exception:
        return None
    classes = ["healthy bark", "Rough bark", "stripecanker"]
    data = [df[df.class_name == c].qsi.values for c in classes]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, tick_labels=["healthy", "rough bark", "stripe canker"],
                    patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], [C3, C1, C2]):
        patch.set_facecolor(col)
    ax.set_ylabel("QSI severity")
    hm = df[df.class_name == "healthy bark"].qsi.mean()
    dm = df[df.class_name != "healthy bark"].qsi.mean()
    ax.set_title(f"Stage 4 — QSI by class (healthy {hm:.4f} vs diseased {dm:.3f})")
    p = out / "fig_stage4_qsi.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    cfg = load_config()
    out = PROJECT_ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    made = []
    for name, fn in [("Stage 1 IoU", fig_stage1),
                     ("Stage 2 confusion", fig_confusion),
                     ("Stage 2 ablation", fig_ablation),
                     ("Stage 3 lesion", fig_lesion),
                     ("Stage 4 QSI", fig_qsi)]:
        try:
            p = fn(out, cfg)
        except Exception as e:
            p = None
            print(f"  {name}: failed ({e})")
        if p:
            made.append(p)
            print(f"  {name}: {p.name}")
        else:
            print(f"  {name}: skipped (source data not found)")

    print(f"\n{len(made)} figure(s) in {out}")


if __name__ == "__main__":
    main()
