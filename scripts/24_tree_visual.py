"""
Step 24 — visualise per-tree severity and treatment stage.

Reads outputs/tree/tree_severity.csv and produces two clear figures:

  1. tree_severity_chart.png — a horizontal bar per tree/disease, length = QSI,
     colour = treatment stage. All trees shown; multi-disease trees get two
     bars. This is the "which trees need attention" figure.

  2. tree_stage_matrix.png — a compact grid: rows = trees, columns = the two
     diseases, each cell coloured by its treatment stage (or grey if that
     disease is absent). The at-a-glance orchard overview.

Run:  python scripts/24_tree_visual.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT                                   # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.patches import Patch                                  # noqa: E402

# stage -> colour (green = fine, red = severe), and an order for sorting
STAGE_COLOR = {
    "Preventive":        "#2ca25f",   # green
    "Early control":     "#fed976",   # yellow
    "Active management":  "#fd8d3c",  # orange
    "Severe outbreak":   "#e31a1c",   # red
}
STAGE_ORDER = ["Preventive", "Early control", "Active management", "Severe outbreak"]


def main():
    out = PROJECT_ROOT / "outputs" / "tree"
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv = out / "tree_severity.csv"
    if not csv.exists():
        sys.exit(f"{csv} not found — run 19_tree_severity.py first")
    df = pd.read_csv(csv)

    # ---------------------------------------------------------------
    # Figure 1 — horizontal bars, one row per tree/disease, coloured by stage
    # ---------------------------------------------------------------
    # sort trees numerically, and within a tree put higher QSI first
    df["tnum"] = df.tree.str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["tnum", "qsi"], ascending=[True, False]).reset_index(drop=True)

    labels, values, colors = [], [], []
    for _, r in df.iterrows():
        dis = "" if r.disease == "No significant disease" else f" · {r.disease}"
        labels.append(f"{r.tree}{dis}")
        values.append(r.pct_bark)
        colors.append(STAGE_COLOR.get(r.action_stage, "#cccccc"))

    h = max(4, 0.42 * len(labels))
    fig, ax = plt.subplots(figsize=(10, h))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()                      # t01 at top
    ax.set_xlabel("% diseased bark area")
    ax.set_title("Per-tree disease severity and treatment stage", fontsize=13)

    for i, (_, r) in enumerate(df.iterrows()):
        if r.disease != "No significant disease":
            spread_txt = f"  spread {r.spread*100:.0f}%"
            girdle_txt = " ⚠ girdling" if r.girdling_risk == "yes" else ""
            ax.text(r.pct_bark + 0.005, i,
                    f"{r.pct_bark*100:.0f}%  QSI {r.qsi:.3f}{spread_txt}{girdle_txt}",
                    va="center", fontsize=7, color="#333")
        else:
            ax.text(0.005, i, "no significant disease",
                    va="center", fontsize=7, color="#999", style="italic")

    legend = [Patch(facecolor=STAGE_COLOR[s], label=s) for s in STAGE_ORDER]
    ax.legend(handles=legend, title="Treatment stage", fontsize=8,
              title_fontsize=9, loc="lower right")
    ax.set_xlim(0, max(values) * 1.45 if max(values) > 0 else 1.0)
    fig.tight_layout()
    p1 = fig_dir / "tree_severity_chart.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p1}")

    # ---------------------------------------------------------------
    # Figure 2 — tree x disease stage matrix (orchard overview)
    # ---------------------------------------------------------------
    diseases = ["Rough bark", "stripecanker"]
    trees = sorted(df.tnum.unique())
    stage_idx = {s: i for i, s in enumerate(STAGE_ORDER)}

    grid = np.full((len(trees), len(diseases)), -1)     # -1 = absent
    for _, r in df.iterrows():
        if r.disease in diseases:
            ti = trees.index(r.tnum)
            di = diseases.index(r.disease)
            grid[ti, di] = stage_idx.get(r.action_stage, -1)

    cmap_colors = ["#eeeeee"] + [STAGE_COLOR[s] for s in STAGE_ORDER]  # -1 grey
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(4.5, max(5, 0.4 * len(trees))))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

    # mark fully-healthy trees (both diseases absent) so they read as "healthy",
    # not as missing data — tint the row and label it.
    healthy_trees = set(df[df.disease == "No significant disease"].tnum)
    for ti, t in enumerate(trees):
        if t in healthy_trees:
            ax.add_patch(plt.Rectangle((-0.5, ti - 0.5), len(diseases), 1,
                                       facecolor="#2ca25f", alpha=0.35, zorder=3))
            ax.text(len(diseases) / 2 - 0.5, ti, "healthy",
                    ha="center", va="center", fontsize=8, color="#146c43",
                    fontweight="bold", zorder=4)

    ax.set_xticks(range(len(diseases)))
    ax.set_xticklabels(["Rough bark", "Stripe canker"], fontsize=9)
    ax.set_yticks(range(len(trees)))
    ax.set_yticklabels([f"t{t:02d}" for t in trees], fontsize=8)
    ax.set_title("Treatment stage by tree and disease", fontsize=12)
    # grid lines
    ax.set_xticks(np.arange(-.5, len(diseases), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(trees), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    legend = ([Patch(facecolor=STAGE_COLOR[s], label=s) for s in STAGE_ORDER]
              + [Patch(facecolor="#eeeeee", label="disease not present")])
    ax.legend(handles=legend, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    p2 = fig_dir / "tree_stage_matrix.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p2}")

    # ---------------------------------------------------------------
    # text summary
    # ---------------------------------------------------------------
    print("\nstage distribution (per tree/disease row):")
    dis = df[df.disease != "No significant disease"]
    for s in STAGE_ORDER:
        n = int((dis.action_stage == s).sum())
        print(f"  {s:20s} {n}")
    healthy = int((df.disease == "No significant disease").sum())
    print(f"  {'No disease':20s} {healthy}")


if __name__ == "__main__":
    main()