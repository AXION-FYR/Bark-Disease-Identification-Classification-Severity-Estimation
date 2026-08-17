"""
Step 12 — aggregate the ablation across seeds into mean +/- std.

Run scripts/11_train_cls.py with --seed 42, 1, 2 for your keeper variants
(plain, concat, mcse_allones), then run this. It reads outputs/cls/ablation.csv
and prints the table you put in the thesis: per-variant mean and standard
deviation of test accuracy and macro-F1 across seeds.

A claim survives if the winning variant's mean stays clearly above the
baseline's across seeds. If the spreads overlap, soften the wording to a
"modest, consistent improvement" — the numbers, not hope, decide which.

Run:  python scripts/12_aggregate_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                                    # noqa: E402

ORDER = ["plain", "masked", "concat", "se", "mcse", "mcse_allones"]
PRETTY = {"mcse_allones": "dualbranch_se (ours)", "mcse": "mask-conditioned SE",
          "se": "unmasked SE", "concat": "appearance+texture",
          "masked": "appearance (masked)", "plain": "appearance (baseline)"}


def main() -> None:
    cfg = load_config()
    csv_path = (cfg.path("paths", "processed_root").parent.parent
                / "outputs" / "cls" / "ablation.csv")
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found — run scripts/11_train_cls.py first")

    df = pd.read_csv(csv_path)
    if "seed" not in df.columns:
        df["seed"] = 42

    for col in ["test_acc", "test_f1", "val_f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("variant").agg(
        n_seeds=("seed", "nunique"),
        acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
        f1_mean=("test_f1", "mean"), f1_std=("test_f1", "std"),
    ).reindex([v for v in ORDER if v in df.variant.unique()])
    g = g.fillna(0.0)

    print("=" * 74)
    print("ABLATION — mean +/- std across seeds")
    print("=" * 74)
    print(f"{'variant':26s} {'seeds':>5s}  {'test acc':>16s}  {'macro-f1':>16s}")
    print("-" * 74)
    for v, r in g.iterrows():
        name = PRETTY.get(v, v)
        acc = f"{r.acc_mean:.3f} +/- {r.acc_std:.3f}"
        f1 = f"{r.f1_mean:.3f} +/- {r.f1_std:.3f}"
        print(f"{name:26s} {int(r.n_seeds):>5d}  {acc:>16s}  {f1:>16s}")

    # verdict on the mask-conditioning question
    if {"mcse", "mcse_allones"} <= set(g.index):
        d = g.loc["mcse_allones", "acc_mean"] - g.loc["mcse", "acc_mean"]
        print("\n" + "-" * 74)
        print(f"mask conditioning effect (mcse_allones - mcse): {d:+.3f} acc")
        if d > 0.02:
            print("  -> unmasked SE is better. Report that bark-masked pooling")
            print("     hurt, and use the unmasked dual-branch SE model.")
        elif d < -0.02:
            print("  -> mask conditioning helped after all; keep mcse.")
        else:
            print("  -> the two are within noise; either is defensible.")

    if {"plain", "mcse_allones"} <= set(g.index):
        d = g.loc["mcse_allones", "acc_mean"] - g.loc["plain", "acc_mean"]
        sep = "clear" if abs(d) > (g.loc["plain", "acc_std"]
                                  + g.loc["mcse_allones", "acc_std"]) else "overlapping"
        print(f"\ndual-branch vs baseline: {d:+.3f} acc  ({sep} across seeds)")
        if sep == "clear" and d > 0:
            print("  -> Claim 1 holds: the texture branch + SE fusion give a")
            print("     real improvement. State it plainly.")
        else:
            print("  -> spreads overlap. Soften to 'modest, consistent gain'.")

    out = csv_path.parent / "ablation_summary.csv"
    g.round(4).to_csv(out)
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
