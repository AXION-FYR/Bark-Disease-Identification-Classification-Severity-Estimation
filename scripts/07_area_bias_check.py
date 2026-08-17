"""
Step 7 — is bark mask area a shortcut for the class?

If mask area differs systematically by class, a network that sees the mask can
score well by reading mask geometry instead of bark texture. That would make
the dual-branch and conditioned-decoder results a measurement of a capture
artifact rather than of the architecture.

This separates the two possible causes:

  (a) FRAMING       — some classes were photographed closer. A real dataset
                      bias that has to be reported and controlled for.
  (b) ASPECT RATIO  — letterboxing a 16:9 image wastes 44% of the canvas as
                      padding versus 25% for 4:3. If aspect ratio correlates
                      with class, letterboxing manufactured the gap and the
                      raw masks are fine.

It then runs the decisive test: train a classifier on the SINGLE feature
`bark_frac` and report its accuracy. Chance is ~33%. Whatever this scores is
the floor your real model has to beat before any of its accuracy can be
attributed to the architecture.

Run:  python scripts/07_area_bias_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                                    # noqa: E402


def main() -> None:
    cfg = load_config()
    proc = cfg.path("paths", "processed_root")
    manifest = proc / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found — run scripts/02_build_masks.py first")

    df = pd.read_csv(manifest)
    df = df[df.class_name != "__none__"].copy()

    size = float(cfg["preprocess"]["size"])

    # fraction of the padded canvas that is real content (not letterbox bars)
    content_w = size - df.lb_pad_left - df.lb_pad_right
    content_h = size - df.lb_pad_top - df.lb_pad_bottom
    df["content_frac"] = (content_w * content_h) / (size * size)

    # undo the letterbox dilution -> bark fraction of the ORIGINAL photo
    df["bark_frac_raw"] = df.bark_frac / df.content_frac
    df["aspect"] = df.orig_w / df.orig_h
    df["aspect_long"] = np.maximum(df.orig_w, df.orig_h) / np.minimum(df.orig_w, df.orig_h)

    print("=" * 70)
    print("POST-LETTERBOX vs RAW bark fraction, by class")
    print("=" * 70)
    tbl = df.groupby("class_name").agg(
        n=("bark_frac", "size"),
        post_mean=("bark_frac", "mean"),
        raw_mean=("bark_frac_raw", "mean"),
        raw_median=("bark_frac_raw", "median"),
        content=("content_frac", "mean"),
        aspect=("aspect_long", "mean"),
    ).round(3)
    print(tbl.to_string())

    post_ratio = tbl.post_mean.max() / tbl.post_mean.min()
    raw_ratio = tbl.raw_mean.max() / tbl.raw_mean.min()
    print(f"\nmax/min class ratio  post-letterbox: {post_ratio:.2f}x"
          f"   raw: {raw_ratio:.2f}x")

    if raw_ratio < post_ratio * 0.7:
        print("=> Letterboxing is inflating the gap. The underlying framing is "
              "more balanced than the processed masks suggest.")
    else:
        print("=> The gap is in the RAW photos, not the letterbox. This is a "
              "genuine framing bias in how the classes were captured.")

    print("\n" + "=" * 70)
    print("ASPECT RATIO by class (long side / short side)")
    print("=" * 70)
    print(df.groupby("class_name").aspect_long
            .describe()[["mean", "min", "50%", "max"]].round(3).to_string())
    shapes = df.groupby(["class_name"]).apply(
        lambda g: g.aspect_long.round(2).value_counts().head(3).to_dict(),
        include_groups=False)
    print("\nmost common aspect ratios per class:")
    for k, v in shapes.items():
        print(f"  {k:14s} {v}")

    # ---------------------------------------------------------------
    # the decisive test: can one number predict the class?
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SHORTCUT TEST — classify from bark area ALONE")
    print("=" * 70)

    train = df[df.split == "train"]
    test = df[df.split == "test"]
    if len(train) == 0 or len(test) == 0:
        print("need both train and test rows; skipping")
        return

    classes = sorted(df.class_name.unique())
    chance = test.class_name.value_counts(normalize=True).max()

    for feat in ["bark_frac", "bark_frac_raw"]:
        # 1-D Gaussian naive Bayes, written out so there is no sklearn dependency
        stats = {}
        for c in classes:
            v = train.loc[train.class_name == c, feat].values
            stats[c] = (v.mean(), v.std() + 1e-6, len(v) / len(train))

        def predict(x):
            best, best_ll = None, -np.inf
            for c, (mu, sd, pri) in stats.items():
                ll = -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) + np.log(pri)
                if ll > best_ll:
                    best, best_ll = c, ll
            return best

        pred = [predict(x) for x in test[feat].values]
        acc = float(np.mean(np.array(pred) == test.class_name.values))
        print(f"  {feat:16s} test accuracy = {acc:.1%}")

    print(f"  {'majority class':16s} test accuracy = {chance:.1%}  (baseline)")

    print("\nHow to read this:")
    print("  near the majority baseline -> mask area carries little class")
    print("     information. Nothing to fix; say so in one methods sentence.")
    print("  well above it              -> this number is the FLOOR your model")
    print("     must beat before any accuracy can be credited to the")
    print("     architecture. Report it in your ablation table as a row.")
    print("     It is a strong thesis to state this openly and clear it.")

    out = cfg.path("paths", "qc_root") / "area_bias.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["split", "class_name", "file_name", "bark_frac", "bark_frac_raw",
        "content_frac", "aspect_long"]].to_csv(out, index=False)
    print(f"\nper-image values: {out}")


if __name__ == "__main__":
    main()
