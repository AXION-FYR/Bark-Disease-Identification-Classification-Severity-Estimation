"""
Step 25 — validate QSI treatment stages against expert grading.

Takes the expert's filled grading sheet (stem, stage in P/E/A/S) and compares
it to the pipeline's QSI-derived stage for the same trunks. Reports:

  * exact agreement %          (QSI stage == expert stage)
  * within-one-stage agreement (off by at most one adjacent stage)
  * Cohen's weighted kappa     (proper ordinal agreement metric)
  * a 4x4 confusion matrix figure (expert vs QSI)

Optionally compares plain vs disease-conditioned QSI agreement, if you saved a
plain-QSI CSV separately — this validates the disease-conditioning novelty.

Run:  python scripts/25_validate_stages.py \
          --grades outputs/grading_pack/grading_sheet.csv
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

STAGES = ["Preventive", "Early control", "Active management", "Severe outbreak"]
LETTER = {"P": 0, "E": 1, "A": 2, "S": 3}
SHORT = ["P", "E", "A", "S"]

# same disease-specific bands as 17_qsi.py and 19_tree_severity.py
DISEASE_BANDS = {
    "Rough bark":   {"prev": 0.30, "early": 0.50, "active": 0.80},
    "stripecanker": {"prev": 0.20, "early": 0.40, "active": 0.70},
    "_default":     {"prev": 0.05, "early": 0.15, "active": 0.50},
}


def area_to_stage_idx(pct: float, disease: str) -> int:
    """% diseased area -> stage index via disease-conditioned bands."""
    b = DISEASE_BANDS.get(disease, DISEASE_BANDS["_default"])
    if pct < b["prev"]:   return 0
    if pct < b["early"]:  return 1
    if pct < b["active"]: return 2
    return 3


def weighted_kappa(expert, pred, k=4):
    """Cohen's quadratic weighted kappa for ordinal categories."""
    O = np.zeros((k, k))
    for e, p in zip(expert, pred):
        O[e, p] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    row = O.sum(1)
    col = O.sum(0)
    E = np.outer(row, col) / n
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i, j] = (i - j) ** 2 / (k - 1) ** 2
    num = (W * O).sum()
    den = (W * E).sum()
    return 1 - num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", required=True,
                    help="expert sheet: columns stem,stage (P/E/A/S)")
    ap.add_argument("--qsi_csv", default="outputs/qsi/qsi_test.csv")
    args = ap.parse_args()

    exp = pd.read_csv(args.grades)
    exp = exp[exp.stage.notna() & (exp.stage.astype(str).str.strip() != "")]
    exp["stage"] = exp.stage.astype(str).str.strip().str.upper().str[0]
    exp = exp[exp.stage.isin(LETTER)]
    if len(exp) < 5:
        sys.exit("fewer than 5 graded rows — need more expert grades")

    qsi = pd.read_csv(PROJECT_ROOT / args.qsi_csv)
    m = exp.merge(qsi[["stem", "qsi", "area_pct", "class_name"]], on="stem", how="inner")
    if len(m) < 5:
        sys.exit(f"only {len(m)} images matched between sheet and QSI output — "
                 "check the stem column matches")

    m["expert_idx"] = m.stage.map(LETTER)
    m["qsi_idx"] = m.apply(
        lambda r: area_to_stage_idx(r.area_pct, r.class_name), axis=1)

    exact = float((m.expert_idx == m.qsi_idx).mean())
    within1 = float((np.abs(m.expert_idx - m.qsi_idx) <= 1).mean())
    kappa = weighted_kappa(m.expert_idx.values, m.qsi_idx.values)

    print("=" * 56)
    print(f"QSI STAGE vs EXPERT STAGE  ({len(m)} images)")
    print("=" * 56)
    print(f"exact agreement:        {exact*100:.1f}%")
    print(f"within one stage:       {within1*100:.1f}%")
    print(f"weighted kappa:         {kappa:.3f}", end="  ")
    print("(>0.6 substantial, >0.8 near-perfect)")

    # confusion matrix (rows = expert, cols = QSI)
    cm = np.zeros((4, 4), int)
    for e, p in zip(m.expert_idx, m.qsi_idx):
        cm[e, p] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(SHORT); ax.set_yticklabels(SHORT)
    ax.set_xlabel("QSI-predicted stage"); ax.set_ylabel("expert stage")
    ax.set_title(f"Stage agreement (exact {exact*100:.0f}%, κ={kappa:.2f})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / "stage_validation.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nconfusion matrix -> {p}")

    # per-disease breakdown, if useful
    if "class_name" in m.columns and m.class_name.nunique() > 1:
        print("\nagreement by disease:")
        for c, g in m.groupby("class_name"):
            ex = float((g.expert_idx == g.qsi_idx).mean())
            print(f"  {c:14s} {ex*100:.0f}% exact ({len(g)} images)")

    m.to_csv(PROJECT_ROOT / "outputs" / "qsi" / "stage_validation_detail.csv",
             index=False)
    print("\nInterpretation: most disagreements should be ADJACENT stages "
          "(near the diagonal). A high within-one-stage number with substantial "
          "kappa is a strong validation for a 4-category ordinal task.")


if __name__ == "__main__":
    main()