"""
Step 3 — LOOK at the masks. Do not skip this.

Polygon decode bugs (coordinate flips, auto-orient mismatch, RLE handled wrong)
are invisible in every metric and instantly obvious in a picture.

Writes overlay grids to outputs/qc/ and prints the worst-case images by bark
fraction, which is where bad masks usually hide.

Run:  python scripts/03_verify_masks.py
      python scripts/03_verify_masks.py --n 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, PROJECT_ROOT                      # noqa: E402
from src.imaging import (read_image_rgb, read_mask_png,               # noqa: E402
                         write_image_rgb, overlay_mask)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402


def load_pair(row, fmt: str):
    ip, mp = PROJECT_ROOT / row.image_path, PROJECT_ROOT / row.mask_path
    if fmt == "png":
        return read_image_rgb(ip), read_mask_png(mp)
    return np.load(ip), np.load(mp).astype(np.uint8)


def grid(rows, fmt, out_path: Path, title: str) -> None:
    n = len(rows)
    if n == 0:
        return
    cols = 4
    r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(4 * cols, 4 * r))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, rows.iterrows()):
        img, mask = load_pair(row, fmt)
        ax.imshow(overlay_mask(img, mask))
        ax.set_title(f"{row.class_name}\n{row.stem[:26]}  bark={row.bark_frac:.2f}",
                     fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    proc = cfg.path("paths", "processed_root")
    qc = cfg.path("paths", "qc_root")
    qc.mkdir(parents=True, exist_ok=True)

    manifest = proc / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found — run scripts/02_build_masks.py first")

    df = pd.read_csv(manifest)
    fmt = str(cfg["preprocess"]["save_format"]).lower()
    n = args.n or int(cfg["qc"]["n_overlay_samples"])
    seed = int(cfg["qc"]["seed"])

    # 1. random sample, stratified by class
    print("random overlays:")
    for split in df.split.unique():
        sub = df[df.split == split]
        per_class = max(1, n // max(1, sub.class_name.nunique()))
        picks = pd.concat(
            [g.sample(min(len(g), per_class), random_state=seed)
             for _, g in sub.groupby("class_name")]
        ).reset_index(drop=True)
        grid(picks, fmt, qc / f"overlay_{split}_random.png",
             f"{split} — random sample (yellow contour = bark mask)")

    # 2. the extremes: smallest and largest bark fraction overall.
    #    A near-zero fraction usually means a decode failure; a near-one
    #    fraction usually means background was included in the polygon.
    print("extreme overlays:")
    low = df.nsmallest(8, "bark_frac")
    high = df.nlargest(8, "bark_frac")
    grid(low, fmt, qc / "overlay_lowest_bark_frac.png",
         "SMALLEST bark fraction — check for decode failures")
    grid(high, fmt, qc / "overlay_highest_bark_frac.png",
         "LARGEST bark fraction — check background was excluded")

    # 3. numeric summary
    print("\nbark fraction by class:")
    print(df.groupby("class_name").bark_frac
            .agg(["count", "mean", "min", "max"]).round(3).to_string())

    empt = df[df.bark_frac < 0.01]
    if len(empt):
        print(f"\n!! {len(empt)} image(s) with bark fraction < 1% — almost "
              f"certainly broken masks:")
        print(empt[["split", "file_name", "bark_frac"]].to_string(index=False))

    print(f"\nOpen {qc} and look at every image before training Stage 1.")


if __name__ == "__main__":
    main()
