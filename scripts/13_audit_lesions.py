"""
Step 13 — audit the lesion evaluation set (the 40 hand-annotated images).

This is a DIFFERENT annotation from the bark masks: each polygon is a LESION
inside the trunk (a stripe-canker streak, a rough-bark patch), NOT the trunk
outline. Used ONLY to evaluate Stage 3 — never to train it.

Checks:
  * lesion area fraction is SMALL (a few % of the image). If it is large, you
    annotated the whole trunk again by mistake.
  * every annotated image is a TEST-split image (no train/valid leakage)
  * every annotated image matches a stem in the processed manifest
  * class balance (~20 stripe canker + ~20 rough bark, no healthy)

Run:  python scripts/13_audit_lesions.py --lesion_root D:/RESEARCH/Dataset/lesion_eval_coco
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                                    # noqa: E402
from src.coco_utils import find_annotation_file, load_coco            # noqa: E402


def norm_stem(file_name: str) -> str:
    """'sc (12)_JPG.rf.<hash>.JPG' -> 'sc(12)'. Roboflow re-hashes on export,
    so match on the original bracketed stem, not the full filename."""
    m = re.match(r"^\s*([A-Za-z]+\s*\(\d+\))", file_name)
    return (m.group(1) if m else Path(file_name).stem).lower().replace(" ", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesion_root", required=True)
    args = ap.parse_args()

    cfg = load_config()
    root = Path(args.lesion_root)
    if not root.exists():
        sys.exit(f"not found: {root}")

    anns = []
    for sub in ["", "train", "valid", "test"]:
        f = root / sub if sub else root
        a = find_annotation_file(f)
        if a:
            anns.append((f, a))
    if not anns:
        sys.exit(f"no COCO json under {root}")

    manifest = pd.read_csv(cfg.path("paths", "processed_root") / "manifest.csv")
    all_stems = {norm_stem(fn): (sp, st) for fn, sp, st in
                 zip(manifest.file_name, manifest.split, manifest.stem)}

    total = 0
    fracs, classes, unmatched, nontest, healthy_hit, rows = [], Counter(), [], [], [], []
    for folder, ann_path in anns:
        coco = load_coco(ann_path)
        for rec in coco:
            total += 1
            img_area = float(rec.width * rec.height)
            frac = (sum(rec.areas) / img_area) if (rec.areas and img_area) else 0.0
            fracs.append(frac)

            ns = norm_stem(rec.file_name)
            pfx = ns[:2]
            cls = {"sc": "stripecanker", "rb": "Rough bark",
                   "h(": "healthy", "he": "healthy"}.get(pfx, "?")
            classes[cls] += 1
            if cls == "healthy":
                healthy_hit.append(rec.file_name)

            if ns in all_stems:
                sp, st = all_stems[ns]
                if sp != "test":
                    nontest.append((rec.file_name, sp))
                rows.append({"lesion_file": rec.file_name, "stem": st,
                             "split": sp, "class": cls,
                             "lesion_frac": round(frac, 4), "n_polys": rec.n_ann})
            else:
                unmatched.append(rec.file_name)

    print("=" * 68)
    print(f"lesion eval set: {total} annotated image(s)")
    print("=" * 68)
    print(f"class balance (filename prefix): {dict(classes)}")

    if fracs:
        q = np.percentile(fracs, [5, 50, 95])
        print(f"lesion area fraction  p5/median/p95: {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}")
        if q[1] > 0.25:
            print("  !! median > 0.25 — that is trunk-sized. You may have "
                  "annotated the whole bark again. Lesions should be a few %.")
        else:
            print("  -> small fractions, consistent with lesion-level annotation.")

    if healthy_hit:
        print(f"\n!! {len(healthy_hit)} HEALTHY image(s) annotated — remove them, "
              f"healthy trunks have no lesions: {healthy_hit[:5]}")
    if nontest:
        print(f"\n!! {len(nontest)} annotated image(s) NOT in test split (leakage):")
        for fn, sp in nontest[:5]:
            print(f"   {fn}  (in {sp})")
    if unmatched:
        print(f"\n!! {len(unmatched)} not found in manifest (name mismatch?): "
              f"{unmatched[:5]}")

    matched_test = [r for r in rows if r["split"] == "test"]
    print(f"\nusable (test-split, matched): {len(matched_test)} image(s)")
    if len(matched_test) < 20:
        print("  !! fewer than 20 usable — annotate more before trusting Stage 3 numbers.")

    out = cfg.path("paths", "qc_root") / "lesion_eval_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
