"""
Step 8 — did the bark-bbox crop actually close the area shortcut?

Script 07 measured the shortcut on whole images: 85.0% from bark area alone,
against a 40.8% majority baseline. This measures the same shortcut on what the
Stage 2 model will actually see, so you get a before/after pair rather than an
assertion.

Two features are tested after cropping:

  fill_ratio  = bark pixels / bbox area
      How completely the trunk fills its own bounding box. Scale-free by
      construction, so shooting distance cannot influence it. This is the
      number that should collapse toward baseline.

  post_crop_frac = bark pixels / final square canvas
      What the network sees after the crop is letterboxed to square. Retains
      the trunk's aspect ratio, which is a real shape property and may legitimately
      differ by class.

Also reports crop pixel area per class — the residual scale confound that
RandomResizedCrop is there to randomise away.

Run:  python scripts/08_crop_bias_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, PROJECT_ROOT                      # noqa: E402
from src.imaging import read_mask_png                                 # noqa: E402
from src.dataset_cls import crop_to_bbox                              # noqa: E402


def gaussian_nb_accuracy(train: pd.DataFrame, test: pd.DataFrame,
                         feat: str) -> float:
    """1-D Gaussian naive Bayes. Same estimator script 07 uses, for comparability."""
    stats = {}
    for c in sorted(train.class_name.unique()):
        v = train.loc[train.class_name == c, feat].values
        stats[c] = (v.mean(), v.std() + 1e-6, len(v) / len(train))

    def predict(x):
        best, best_ll = None, -np.inf
        for c, (mu, sd, pri) in stats.items():
            ll = -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) + np.log(pri)
            if ll > best_ll:
                best, best_ll = c, ll
        return best

    pred = np.array([predict(x) for x in test[feat].values])
    return float((pred == test.class_name.values).mean())


def main() -> None:
    cfg = load_config()
    proc = cfg.path("paths", "processed_root")
    df = pd.read_csv(proc / "manifest.csv")
    df = df[(df.class_name != "__none__") & (df.bark_frac > 0)].copy()

    fmt = str(cfg["preprocess"]["save_format"]).lower()
    margin = 0.05

    fill, post, crop_px, aspect = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="cropping", unit="img"):
        mp = PROJECT_ROOT / row.mask_path
        mask = read_mask_png(mp) if fmt == "png" else np.load(mp).astype(np.uint8)

        _, mc = crop_to_bbox(mask, mask, margin)
        h, w = mc.shape[:2]
        area = float(h * w)
        bark = float(mc.sum())

        fill.append(bark / max(area, 1.0))
        crop_px.append(area)
        aspect.append(max(h, w) / max(min(h, w), 1))
        # bark fraction after letterboxing the crop to a square canvas
        post.append(bark / float(max(h, w) ** 2))

    df["fill_ratio"] = fill
    df["post_crop_frac"] = post
    df["crop_px"] = crop_px
    df["crop_aspect"] = aspect

    print("\n" + "=" * 74)
    print("AFTER BBOX CROP, by class")
    print("=" * 74)
    tbl = df.groupby("class_name").agg(
        n=("fill_ratio", "size"),
        before_raw=("bark_frac", "mean"),
        fill_ratio=("fill_ratio", "mean"),
        post_crop=("post_crop_frac", "mean"),
        crop_px=("crop_px", "mean"),
        aspect=("crop_aspect", "mean"),
    ).round(3)
    print(tbl.to_string())

    print(f"\nmax/min class ratio")
    print(f"  before (whole-image bark_frac): "
          f"{tbl.before_raw.max() / tbl.before_raw.min():.2f}x")
    print(f"  after  (fill_ratio)           : "
          f"{tbl.fill_ratio.max() / tbl.fill_ratio.min():.2f}x")
    print(f"  residual crop-size ratio      : "
          f"{tbl.crop_px.max() / tbl.crop_px.min():.2f}x   "
          f"<- what RandomResizedCrop randomises")

    train, test = df[df.split == "train"], df[df.split == "test"]
    if len(train) == 0 or len(test) == 0:
        sys.exit("need both train and test rows")

    baseline = test.class_name.value_counts(normalize=True).max()

    print("\n" + "=" * 74)
    print("SHORTCUT TEST — before vs after")
    print("=" * 74)
    rows = [
        ("bark_frac (whole image, BEFORE)", "bark_frac"),
        ("fill_ratio (after crop)", "fill_ratio"),
        ("post_crop_frac (after crop+letterbox)", "post_crop_frac"),
        ("crop_px (residual scale cue)", "crop_px"),
    ]
    for label, feat in rows:
        acc = gaussian_nb_accuracy(train, test, feat)
        flag = ""
        if acc > baseline + 0.15:
            flag = "  <- still leaking"
        elif acc <= baseline + 0.07:
            flag = "  <- effectively closed"
        print(f"  {label:40s} {acc:6.1%}{flag}")
    print(f"  {'majority class (baseline)':40s} {baseline:6.1%}")

    print("\nWhat to do with this:")
    print("  fill_ratio near baseline -> the crop closed the area shortcut.")
    print("     Put the before/after pair in your thesis; it is direct evidence,")
    print("     not an assertion.")
    print("  crop_px still high       -> expected, and it is why scale jitter")
    print("     is mandatory rather than optional. RandomResizedCrop with")
    print("     scale=(0.4, 1.0) removes the model's access to absolute scale.")
    print("  fill_ratio still high    -> trunk shape itself differs by class.")
    print("     That is a real property, not an artifact, but say so explicitly.")

    out = cfg.path("paths", "qc_root") / "crop_bias.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["split", "class_name", "file_name", "bark_frac", "fill_ratio",
        "post_crop_frac", "crop_px", "crop_aspect"]].to_csv(out, index=False)
    print(f"\nper-image values: {out}")


if __name__ == "__main__":
    main()
