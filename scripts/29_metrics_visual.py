"""
Step 29 — visualise the proposed model's OWN performance across metrics.

Not a comparison — this shows your dual-branch ensemble on its own, across the
several evaluation variables (accuracy, error rate, macro-F1, balanced accuracy,
Cohen's kappa) plus the per-class recall breakdown. All numbers are computed
exactly from the ensemble's confusion matrix, so nothing is estimated.

Default confusion matrix is your ensemble result [[37,6,6],[1,35,0],[1,0,34]]
(rows = true [Rough bark, healthy, stripe canker]). Override with --cm if needed.

Produces:
  fig_ours_metrics.png       overall metrics as a bar chart (with error rate)
  fig_ours_perclass.png      per-class precision / recall / F1

Run:  python scripts/29_metrics_visual.py
      python scripts/29_metrics_visual.py --cm 37,6,6,1,35,0,1,0,34
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT                                   # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

CLASSES = ["Rough bark", "healthy", "stripe canker"]


def metrics_from_cm(cm):
    k = cm.shape[0]
    n = cm.sum()
    acc = np.trace(cm) / n

    prec, rec, f1 = [], [], []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        prec.append(p); rec.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)

    macro_f1 = float(np.mean(f1))
    balanced_acc = float(np.mean(rec))
    po = acc
    pe = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(k)) / (n * n)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0

    return {
        "accuracy": acc, "error_rate": 1 - acc, "macro_f1": macro_f1,
        "balanced_acc": balanced_acc, "kappa": float(kappa),
        "precision": prec, "recall": rec, "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm", default="37,6,6,1,35,0,1,0,34",
                    help="9 comma-separated confusion-matrix values, row-major "
                         "(rows=true [rb, healthy, sc])")
    ap.add_argument("--title", default="Dual-branch ensemble (proposed)")
    args = ap.parse_args()

    vals = [int(v) for v in args.cm.split(",")]
    if len(vals) != 9:
        sys.exit("--cm needs exactly 9 values")
    cm = np.array(vals).reshape(3, 3)
    m = metrics_from_cm(cm)

    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- print
    print(f"\n{args.title}")
    print("=" * 46)
    print(f"  accuracy         {m['accuracy']:.3f}")
    print(f"  error rate       {m['error_rate']:.3f}")
    print(f"  macro-F1         {m['macro_f1']:.3f}")
    print(f"  balanced acc     {m['balanced_acc']:.3f}")
    print(f"  Cohen's kappa    {m['kappa']:.3f}")
    print("  per-class recall: " +
          "  ".join(f"{c} {r:.2f}" for c, r in zip(CLASSES, m["recall"])))

    # ---------------------------------------------------------------
    # Figure 1 — overall metrics (accuracy/macro-F1/balanced/kappa up,
    #            error rate shown separately in red since lower = better)
    # ---------------------------------------------------------------
    good = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
            ("balanced_acc", "Balanced acc"), ("kappa", "Cohen's κ")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [lbl for _, lbl in good] + ["Error rate"]
    vals_plot = [m[k] for k, _ in good] + [m["error_rate"]]
    colors = ["#2c7fb8"] * len(good) + ["#d95f02"]
    bars = ax.bar(range(len(names)), vals_plot, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(f"{args.title} — performance metrics")
    for i, v in enumerate(vals_plot):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    # note which direction is better
    ax.text(len(names) - 1, 0.02, "lower\nis better", ha="center",
            fontsize=7, color="#d95f02")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = fig_dir / "fig_ours_metrics.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p1}")

    # ---------------------------------------------------------------
    # Figure 2 — per-class precision / recall / F1
    # ---------------------------------------------------------------
    x = np.arange(len(CLASSES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, m["precision"], w, label="precision", color="#2c7fb8")
    ax.bar(x, m["recall"], w, label="recall", color="#7fcdbb")
    ax.bar(x + w, m["f1"], w, label="F1", color="#fdae61")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"{args.title} — per-class breakdown")
    for i in range(len(CLASSES)):
        for off, key in [(-w, "precision"), (0, "recall"), (w, "f1")]:
            ax.text(x[i] + off, m[key][i] + 0.01, f"{m[key][i]:.2f}",
                    ha="center", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = fig_dir / "fig_ours_perclass.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p2}")


if __name__ == "__main__":
    main()
