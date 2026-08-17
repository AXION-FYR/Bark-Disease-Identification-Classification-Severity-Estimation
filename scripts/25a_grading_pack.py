"""
Step 25a — export a grading pack for the expert.

Copies the test-split trunk images into one folder and writes a blank grading
sheet the expert fills in with the treatment stage (by eye, WITHOUT seeing the
model's output — grading must be blind for the validation to be meaningful).

Stages: P = Preventive, E = Early control, A = Active management, S = Severe.

The sheet has a 'stem' column already filled (so it matches the QSI output) and
an empty 'stage' column for the expert. Images are given simple numbers so the
expert isn't biased by filenames that hint at the disease.

Run:  python scripts/25a_grading_pack.py --n 40
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, PROJECT_ROOT                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="how many images to grade")
    ap.add_argument("--split", default="test")
    ap.add_argument("--diseased_only", action="store_true",
                    help="only include diseased trunks (recommended — healthy "
                         "are trivially Preventive)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    manifest = pd.read_csv(cfg.path("paths", "processed_root") / "manifest.csv")
    sub = manifest[(manifest.split == args.split) & (manifest.class_idx >= 0)]
    if args.diseased_only:
        sub = sub[sub.class_name != "healthy bark"]

    # sample up to n, stratified across diseases for a balanced pack
    n = min(args.n, len(sub))
    picks = (sub.groupby("class_name", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), max(1, n // sub.class_name.nunique())),
                                       random_state=args.seed)))
    picks = picks.sample(min(n, len(picks)), random_state=args.seed).reset_index(drop=True)

    pack = PROJECT_ROOT / "outputs" / "grading_pack"
    img_dir = pack / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (_, r) in enumerate(picks.iterrows(), start=1):
        src = PROJECT_ROOT / r.image_path
        # neutral numbered name so the expert isn't cued by the filename
        dst = img_dir / f"img_{i:03d}{Path(r.image_path).suffix}"
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"  could not copy {src}: {e}")
            continue
        rows.append({"image_file": dst.name, "stem": r.stem, "stage": ""})

    sheet = pd.DataFrame(rows)
    sheet_path = pack / "grading_sheet.csv"
    sheet.to_csv(sheet_path, index=False)

    # instructions file
    (pack / "INSTRUCTIONS.txt").write_text(
        "EXPERT GRADING — cinnamon bark disease treatment stage\n"
        "=======================================================\n\n"
        f"Please assign each of the {len(rows)} trunk images in the 'images' "
        "folder to ONE treatment stage, based on how the bark looks:\n\n"
        "  P = Preventive        (healthy / trace — monitor only)\n"
        "  E = Early control     (limited disease — intervene early)\n"
        "  A = Active management (significant spread — active treatment)\n"
        "  S = Severe outbreak   (majority of bark affected — aggressive)\n\n"
        "Open grading_sheet.csv, and in the 'stage' column write P, E, A, or S\n"
        "for each image (match by the image_file name).\n\n"
        "IMPORTANT: please grade from the images ALONE. Do not look at any\n"
        "model output — the comparison is only valid if grading is independent.\n",
        encoding="utf-8")

    print(f"grading pack -> {pack}")
    print(f"  {len(rows)} images in {img_dir}")
    print(f"  blank sheet: {sheet_path}")
    print(f"  instructions: {pack / 'INSTRUCTIONS.txt'}")
    print("\nGive the whole 'grading_pack' folder to the expert. When they return\n"
          "the filled grading_sheet.csv, run 25_validate_stages.py.")


if __name__ == "__main__":
    main()
