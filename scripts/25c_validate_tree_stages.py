"""
Step 25c — validate TREE-LEVEL severity against expert tree grading.

Reads the expert's filled-in tree_grading_sheet.csv (from 25b) and compares it
against the pipeline's own action_stage per tree per disease (from
tree_severity.csv). Computes the same agreement metrics used at image level
(25_validate_stages.py): exact agreement, within-one-stage, weighted kappa --
but now validating the FULL Stage 5 aggregation logic (weighting, spread,
girdling escalation), not just per-image QSI.

Run:  python scripts/25c_validate_tree_stages.py \
          --grades outputs/tree_grading_pack/tree_grading_sheet.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT                                    # noqa: E402

STAGES = ["Preventive", "Early control", "Active management", "Severe outbreak"]
LETTER = {"P": 0, "E": 1, "A": 2, "S": 3}
SHORT = ["P", "E", "A", "S"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", required=True,
                    help="filled-in tree_grading_sheet.csv from 25b")
    ap.add_argument("--tree_csv", default="outputs/tree/tree_severity.csv")
    args = ap.parse_args()

    grades = pd.read_csv(PROJECT_ROOT / args.grades)
    grades = grades[grades.expert_stage.notna() &
                    (grades.expert_stage.astype(str).str.strip() != "")]
    if len(grades) == 0:
        sys.exit("No filled-in expert_stage values found -- has the expert "
                 "completed the grading sheet yet?")

    grades["expert_idx"] = (grades.expert_stage.astype(str).str.strip()
                            .str.upper().str[0].map(LETTER))

    tree_df = pd.read_csv(PROJECT_ROOT / args.tree_csv)
    # normalise tree id format (t05 vs tree5 vs 05) by keeping digits only
    def norm_tree(x):
        return "".join(ch for ch in str(x) if ch.isdigit()).lstrip("0") or "0"
    grades["tree_key"] = grades.tree.apply(norm_tree)
    tree_df["tree_key"] = tree_df.tree.apply(norm_tree)

    m = grades.merge(tree_df[["tree_key", "disease", "action_stage"]],
                     on=["tree_key", "disease"], how="left")
    unmatched = m[m.action_stage.isna()]
    if len(unmatched):
        print(f"WARNING: {len(unmatched)} graded row(s) had no matching "
              f"pipeline prediction (tree/disease not found in {args.tree_csv}):")
        print(unmatched[["tree", "disease"]].to_string(index=False))
        m = m[m.action_stage.notna()]

    if len(m) == 0:
        sys.exit("No matched rows between expert grades and pipeline output "
                 "-- check that tree IDs and disease names align.")

    stage_idx = {s: i for i, s in enumerate(STAGES)}
    m["pred_idx"] = m.action_stage.map(stage_idx)
    m["stage_diff"] = m.pred_idx - m.expert_idx
    m["exact"] = m.stage_diff == 0
    m["within_one"] = m["stage_diff"].abs() <= 1

    exact_pct = m.exact.mean() * 100
    within_pct = m.within_one.mean() * 100

    # quadratic weighted kappa
    n_cls = 4
    conf = np.zeros((n_cls, n_cls), int)
    for _, r in m.iterrows():
        conf[int(r.expert_idx), int(r.pred_idx)] += 1
    n = conf.sum()
    row_marg = conf.sum(axis=1)
    col_marg = conf.sum(axis=0)
    w = np.zeros((n_cls, n_cls))
    for i in range(n_cls):
        for j in range(n_cls):
            w[i, j] = ((i - j) ** 2) / ((n_cls - 1) ** 2)
    expected = np.outer(row_marg, col_marg) / n
    num = (w * conf).sum()
    den = (w * expected).sum()
    kappa = 1 - num / den if den > 0 else float("nan")

    print("\n" + "=" * 66)
    print(f"TREE-LEVEL STAGE VALIDATION  ({len(m)} tree-disease gradings)")
    print("=" * 66)
    print(f"exact agreement:        {exact_pct:.1f}%")
    print(f"within one stage:       {within_pct:.1f}%")
    print(f"weighted kappa:         {kappa:.3f}  "
          f"(>0.6 substantial, >0.8 near-perfect)")

    print("\nagreement by disease:")
    for dis, g in m.groupby("disease"):
        print(f"  {dis:14s} {g.exact.mean()*100:.0f}% exact  "
              f"({len(g)} trees)")

    print("\nper-tree detail:")
    print(f"{'tree':8s} {'disease':14s} {'expert':20s} {'predicted':20s} match")
    print("-" * 78)
    for _, r in m.sort_values(["disease", "tree"]).iterrows():
        ok = "OK" if r.exact else ("~1" if r.within_one else "XX")
        print(f"{r.tree:8s} {r.disease:14s} {STAGES[int(r.expert_idx)]:20s} "
              f"{r.action_stage:20s} {ok}")

    # confusion matrix figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(conf, cmap="Blues", vmin=0)
        ax.set_xticks(range(4)); ax.set_xticklabels(SHORT)
        ax.set_yticks(range(4)); ax.set_yticklabels(SHORT)
        ax.set_xlabel("pipeline action_stage"); ax.set_ylabel("expert tree grade")
        ax.set_title(f"Tree-level: exact {exact_pct:.0f}%, "
                     f"within-1 {within_pct:.0f}%, kappa {kappa:.2f}")
        for i in range(4):
            for j in range(4):
                if conf[i, j] > 0:
                    ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                            color="white" if conf[i, j] > conf.max() / 2 else "black")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fp = PROJECT_ROOT / "outputs" / "figures" / "tree_stage_validation.png"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fp, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\nconfusion matrix -> {fp}")
    except Exception as e:
        print(f"(figure skipped: {e})")

    print("\nInterpretation: this validates the FULL Stage 5 aggregation "
          "logic (quality weighting, spread, girdling escalation) against "
          "an expert's holistic judgement of each tree -- not just the "
          "per-image QSI validated in Stage 4.")


if __name__ == "__main__":
    main()
