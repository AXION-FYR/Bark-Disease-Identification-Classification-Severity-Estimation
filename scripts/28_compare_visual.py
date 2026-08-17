"""
Step 28 — visualise the model comparison across MULTIPLE metrics.

Reads outputs/cls/baseline_compare.csv (the baselines) and adds your proposed
dual-branch ensemble, then draws two figures:

  1. fig_metrics_grouped.png — grouped bars: each model shown across several
     metrics (accuracy, macro-F1, balanced accuracy, kappa, rough-bark recall),
     so you can see it doesn't just win on accuracy.

  2. fig_error_rate.png — a focused error-rate comparison (lower = better),
     the variable your supervisor asked for.

Your ensemble's metrics are passed in via --ours_* flags (defaults filled from
your known ensemble result); edit them if your numbers change.

Run:  python scripts/28_compare_visual.py \
          --ours_acc 0.883 --ours_macro_f1 0.87 --ours_bal_acc 0.86 \
          --ours_kappa 0.82 --ours_rb_recall 0.755
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT                                   # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

PRETTY = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0",
          "mobilenetv3_large_100": "MobileNetV3", "vit_small_patch16_224": "ViT-S"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/cls/baseline_compare.csv")
    ap.add_argument("--ours_acc", type=float, default=0.883)
    ap.add_argument("--ours_macro_f1", type=float, default=0.87)
    ap.add_argument("--ours_bal_acc", type=float, default=0.86)
    ap.add_argument("--ours_kappa", type=float, default=0.82)
    ap.add_argument("--ours_rb_recall", type=float, default=0.755)
    args = ap.parse_args()

    csv = PROJECT_ROOT / args.csv
    if not csv.exists():
        sys.exit(f"{csv} not found — run 27_baseline_compare.py first")
    df = pd.read_csv(csv)

    # average across seeds per model
    g = df.groupby("model").agg(
        test_acc=("test_acc", "mean"),
        error_rate=("error_rate", "mean"),
        macro_f1=("macro_f1", "mean"),
        balanced_acc=("balanced_acc", "mean"),
        kappa=("kappa", "mean"),
        rb_recall=("rb_recall", "mean"),
    ).reset_index()
    g["name"] = g.model.map(lambda m: PRETTY.get(m, m))

    # append the proposed method
    ours = {"name": "Dual-branch\nensemble (ours)",
            "test_acc": args.ours_acc,
            "error_rate": round(1 - args.ours_acc, 4),
            "macro_f1": args.ours_macro_f1,
            "balanced_acc": args.ours_bal_acc,
            "kappa": args.ours_kappa,
            "rb_recall": args.ours_rb_recall}
    rows = g.to_dict("records") + [ours]
    # order baselines by accuracy, ours last
    base = sorted([r for r in rows if "ours" not in r["name"]],
                  key=lambda r: r["test_acc"])
    ordered = base + [r for r in rows if "ours" in r["name"]]
    names = [r["name"] for r in ordered]

    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Figure 1 — grouped bars across metrics (higher = better metrics)
    # ---------------------------------------------------------------
    metrics = [("test_acc", "Accuracy"), ("macro_f1", "Macro-F1"),
               ("balanced_acc", "Balanced acc"), ("kappa", "Cohen's κ"),
               ("rb_recall", "Rough-bark recall")]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ordered)))

    x = np.arange(len(metrics))
    w = 0.8 / len(ordered)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(ordered):
        vals = [r[m] for m, _ in metrics]
        offset = (i - len(ordered) / 2 + 0.5) * w
        is_ours = "ours" in r["name"]
        ax.bar(x + offset, vals, w, label=r["name"].replace("\n", " "),
               color=colors[i],
               edgecolor="black" if is_ours else "none",
               linewidth=1.5 if is_ours else 0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Model comparison across metrics")
    ax.legend(fontsize=8, ncol=len(ordered), loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = fig_dir / "fig_metrics_grouped.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p1}")

    # ---------------------------------------------------------------
    # Figure 2 — error rate (lower = better), the requested variable
    # ---------------------------------------------------------------
    errs = [r["error_rate"] for r in ordered]
    bar_colors = ["#7fcdbb"] * (len(ordered) - 1) + ["#2c7fb8"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(names)), errs, color=bar_colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("\n", " ") for n in names], fontsize=8, rotation=15)
    ax.set_ylabel("error rate (lower is better)")
    ax.set_title("Classification error rate")
    for i, e in enumerate(errs):
        ax.text(i, e + 0.005, f"{e:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = fig_dir / "fig_error_rate.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p2}")

    # ---------------------------------------------------------------
    # text table for the thesis
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'model':22s}{'acc':>8}{'error':>8}{'macroF1':>9}"
          f"{'bal_acc':>9}{'kappa':>8}{'rb_rec':>8}")
    print("-" * 78)
    for r in ordered:
        nm = r["name"].replace("\n", " ")
        print(f"{nm:22s}{r['test_acc']:>8.3f}{r['error_rate']:>8.3f}"
              f"{r['macro_f1']:>9.3f}{r['balanced_acc']:>9.3f}"
              f"{r['kappa']:>8.3f}{r['rb_recall']:>8.3f}")
    print("=" * 78)

    pd.DataFrame(ordered).to_csv(fig_dir.parent / "cls" / "comparison_full.csv",
                                 index=False)
    print(f"table -> {fig_dir.parent / 'cls' / 'comparison_full.csv'}")


if __name__ == "__main__":
    main()
