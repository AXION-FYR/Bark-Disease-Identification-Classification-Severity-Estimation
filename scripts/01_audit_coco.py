"""
Step 1 — audit the Roboflow export BEFORE processing anything.

Answers, in one run:
  * Are these whole-trunk masks or lesion masks? (median area fraction)
  * Do the class counts match what you expect (277 / 374 / 302)?
  * Are there images with polygons of more than one class?
  * Are there images with no annotation at all?
  * Do the JSON width/height agree with the actual image files? (auto-orient check)

Run:  python scripts/01_audit_coco.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                                    # noqa: E402
from src.coco_utils import (find_annotation_file, load_coco,          # noqa: E402
                            image_class, resolve_class_order)

try:
    from PIL import Image
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False


def audit_split(folder: Path, split: str, cfg) -> dict | None:
    ann = find_annotation_file(folder)
    if ann is None:
        print(f"[{split}] no _annotations.coco.json found — skipping")
        return None

    coco = load_coco(ann, min_area_frac=0.0)
    print(f"\n{'=' * 68}\n[{split}]  {ann.relative_to(cfg.path('paths', 'raw_root').parent.parent)}")
    print("=" * 68)

    n_imgs = len(coco)
    n_ann = sum(r.n_ann for r in coco)
    print(f"images: {n_imgs}    annotations: {n_ann}")
    print(f"categories in use: {list(coco.categories.values())}")

    # --- polygons per image
    per_img = Counter(r.n_ann for r in coco)
    print(f"polygons/image: {dict(sorted(per_img.items()))}")

    empty = [r.file_name for r in coco if r.n_ann == 0]
    if empty:
        print(f"!! {len(empty)} image(s) with NO annotation "
              f"(these break every downstream stage): {empty[:5]}")

    # --- class distribution, per annotation and per image
    ann_classes = Counter()
    for r in coco:
        ann_classes.update(r.class_names)
    img_classes, mixed = Counter(), []
    for r in coco:
        name, is_mixed = image_class(r)
        img_classes[name] += 1
        if is_mixed:
            mixed.append(r.file_name)

    print(f"class counts (per annotation): {dict(ann_classes)}")
    print(f"class counts (per image)     : {dict(img_classes)}")
    if sum(ann_classes.values()) != sum(img_classes.values()):
        print("   -> counts differ: some images carry multiple polygons. "
              "Stratify the split on the PER-IMAGE class.")

    if mixed:
        print(f"!! {len(mixed)} MIXED-CLASS image(s) — a trunk annotated with two "
              f"diseases. Decide now: exclude, or take majority area.")
        print(f"   examples: {mixed[:5]}")
    else:
        print("mixed-class images: 0")

    # --- area fraction: whole-trunk masks should sit around 0.15-0.60
    fracs = []
    for r in coco:
        if not r.areas:
            continue
        img_area = float(r.width * r.height)
        if img_area > 0:
            fracs.append(sum(r.areas) / img_area)
    if fracs:
        q = np.percentile(fracs, [5, 50, 95])
        print(f"mask area fraction  p5/median/p95: "
              f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
        if q[1] < 0.05:
            print("   !! median < 0.05 — these look like LESION polygons, not "
                  "whole-trunk masks. That changes the plan: Stage 3 becomes "
                  "supervised. Verify visually before continuing.")
        elif q[1] > 0.90:
            print("   !! median > 0.90 — masks cover nearly the whole frame. "
                  "Check that background is genuinely excluded.")
        else:
            print("   -> consistent with whole-trunk bark masks.")

    # --- auto-orient / dimension agreement
    if _HAVE_PIL:
        bad_dims, missing = [], []
        for r in coco:
            fp = folder / r.file_name
            if not fp.exists():
                missing.append(r.file_name)
                continue
            with Image.open(fp) as im:
                w, h = im.size
            if (w, h) != (r.width, r.height):
                bad_dims.append((r.file_name, (w, h), (r.width, r.height)))
        if missing:
            print(f"!! {len(missing)} file(s) listed in JSON but missing on disk: "
                  f"{missing[:5]}")
        if bad_dims:
            print(f"!! {len(bad_dims)} image(s) whose real size differs from the JSON. "
                  f"This is the classic auto-orient bug: masks will be rotated "
                  f"relative to the image. Re-export with auto-orient ON.")
            for name, real, js in bad_dims[:5]:
                print(f"   {name}: file={real}  json={js}")
        if not missing and not bad_dims:
            print("file/JSON dimensions: all consistent")

    return {"split": split, "n_images": n_imgs, "img_classes": img_classes}


def main() -> None:
    cfg = load_config()
    raw_root = cfg.path("paths", "raw_root")
    if not raw_root.exists():
        sys.exit(f"raw_root does not exist: {raw_root}\n"
                 f"Put the Roboflow COCO export there first.")

    results = []
    for split in cfg["splits"]:
        folder = raw_root / split
        if folder.exists():
            r = audit_split(folder, split, cfg)
            if r:
                results.append(r)

    if not results:                       # flat layout fallback
        r = audit_split(raw_root, "all", cfg)
        if r:
            results.append(r)

    if not results:
        sys.exit("\nNo COCO annotation file found anywhere under raw_root.")

    print(f"\n{'=' * 68}\nTOTAL\n{'=' * 68}")
    total = Counter()
    for r in results:
        total.update(r["img_classes"])
    print(f"images: {sum(r['n_images'] for r in results)}")
    print(f"per-image class totals: {dict(total)}")

    order = resolve_class_order(list(total.keys()), load_config()["classes"]["order"])
    print(f"resolved class order (index 0,1,2...): {order}")
    print("\nIf the numbers above look right, run scripts/02_build_masks.py")


if __name__ == "__main__":
    main()
