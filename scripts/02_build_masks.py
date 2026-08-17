"""
Step 2 — generate binary bark masks and the preprocessed cache.

For every image:
  1. decode all COCO polygons and take their UNION  -> binary bark mask {0,1}
     (all three classes collapse to one foreground label: Stage 1 is
      bark-vs-background; the disease class is carried in the manifest)
  2. letterbox-pad to square, then resize to `size`
     - image with INTER_AREA / INTER_LINEAR
     - mask  with INTER_NEAREST  (never interpolate a binary mask)
  3. write image + mask to data/processed/<split>/{images,masks}/
  4. record one manifest row per image

Run:  python scripts/02_build_masks.py
      python scripts/02_build_masks.py --limit 20      # quick trial
      python scripts/02_build_masks.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                                    # noqa: E402
from src.coco_utils import (find_annotation_file, load_coco,          # noqa: E402
                            polygons_to_mask, image_class,
                            resolve_class_order)
from src.imaging import (read_image_rgb, write_image_rgb,             # noqa: E402
                         write_mask_png, letterbox_image, letterbox_mask,
                         bbox_from_mask)


def process_split(folder: Path, split: str, cfg, class_to_idx: dict[str, int],
                  overwrite: bool, limit: int | None) -> list[dict]:
    ann_path = find_annotation_file(folder)
    if ann_path is None:
        print(f"[{split}] no annotation file — skipping")
        return []

    coco = load_coco(ann_path, min_area_frac=cfg["preprocess"]["min_area_frac"])

    out_root = cfg.path("paths", "processed_root") / split
    img_dir, mask_dir = out_root / "images", out_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    size = int(cfg["preprocess"]["size"])
    fmt = str(cfg["preprocess"]["save_format"]).lower()
    pad_value = int(cfg["preprocess"]["pad_value"])
    use_letterbox = bool(cfg["preprocess"]["letterbox"])

    rows: list[dict] = []
    rotate_dir = str(cfg["preprocess"].get("rotate_dir", "cw")).lower()
    n_rotated = n_resized = 0
    # --- honour preprocess.exclude (matched on the stem, so the Roboflow
    #     ".rf.<hash>" suffix and the extension do not have to be typed out)
    excl = [str(e).strip().lower() for e in (cfg["preprocess"].get("exclude") or [])]
    records = list(coco)
    if excl:
        before = len(records)
        records = [r for r in records
                   if not any(e in r.file_name.lower() for e in excl)]
        if before != len(records):
            print(f"  [{split}] excluded {before - len(records)} image(s) "
                  f"via preprocess.exclude")

    if limit:
        records = records[:limit]

    skipped_empty = 0
    for rec in tqdm(records, desc=f"[{split}]", unit="img"):
        src = folder / rec.file_name
        if not src.exists():
            print(f"  missing file, skipped: {rec.file_name}")
            continue

        stem = Path(rec.file_name).stem
        img_out = img_dir / (stem + (".png" if fmt == "png" else ".npy"))
        mask_out = mask_dir / (stem + (".png" if fmt == "png" else ".npy"))

        img = read_image_rgb(src)
        h, w = img.shape[:2]

        # Rasterise polygons in the JSON's coordinate frame.
        mask = polygons_to_mask(rec, h=rec.height, w=rec.width)

        # If the loaded image is TRANSPOSED relative to the JSON, EXIF was
        # stripped on export and read_image_rgb could not fix it. Rotating the
        # image is correct here; resizing would squash a portrait mask into a
        # landscape frame and silently ruin the annotation. Direction comes
        # from config (verify it with scripts/05_check_orientation.py).
        if (h, w) == (rec.width, rec.height) and (h, w) != (rec.height, rec.width):
            import cv2
            code = (cv2.ROTATE_90_CLOCKWISE if rotate_dir == "cw"
                    else cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.rotate(img, code)
            h, w = img.shape[:2]
            n_rotated += 1
        elif (rec.height, rec.width) != (h, w):
            # genuine scale difference, not a transpose -> resize the mask
            import cv2
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            n_resized += 1

        if mask.sum() == 0:
            skipped_empty += 1

        if use_letterbox:
            img_p, params = letterbox_image(img, size, pad_value=pad_value)
            mask_p = letterbox_mask(mask, params)
        else:
            import cv2
            img_p = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            mask_p = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            from src.imaging import LetterboxParams
            params = LetterboxParams(w, h, size / max(h, w), 0, 0, 0, 0, size)

        if overwrite or not img_out.exists():
            if fmt == "png":
                write_image_rgb(img_out, img_p)
                write_mask_png(mask_out, mask_p)
            else:
                np.save(img_out, img_p)
                np.save(mask_out, mask_p.astype(np.uint8))

        cls_name, is_mixed = image_class(rec)
        bbox = bbox_from_mask(mask_p)
        bark_frac = float(mask_p.mean())

        rows.append({
            "split": split,
            "file_name": rec.file_name,
            "stem": stem,
            "image_path": str(img_out.relative_to(cfg.path("paths", "processed_root").parent.parent)),
            "mask_path": str(mask_out.relative_to(cfg.path("paths", "processed_root").parent.parent)),
            "class_name": cls_name,
            "class_idx": class_to_idx.get(cls_name, -1),
            "mixed_class": int(is_mixed),
            "n_polygons": rec.n_ann,
            "orig_w": w,
            "orig_h": h,
            "bark_frac": round(bark_frac, 5),
            "bbox_x0": bbox[0] if bbox else -1,
            "bbox_y0": bbox[1] if bbox else -1,
            "bbox_x1": bbox[2] if bbox else -1,
            "bbox_y1": bbox[3] if bbox else -1,
            **{f"lb_{k}": v for k, v in params.to_dict().items()},
        })

    if n_rotated:
        print(f"  [{split}] rotated {n_rotated} image(s) to match the annotation "
              f"frame (rotate_dir={rotate_dir}) — verify with 05_check_orientation.py")
    if n_resized:
        print(f"  [{split}] resized {n_resized} mask(s) for a scale mismatch")
    if skipped_empty:
        print(f"  !! {skipped_empty} image(s) produced an EMPTY mask in [{split}]")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N images per split (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_root = cfg.path("paths", "raw_root")
    proc_root = cfg.path("paths", "processed_root")
    proc_root.mkdir(parents=True, exist_ok=True)

    # ---- discover the class vocabulary across every split first
    found: set[str] = set()
    split_folders: list[tuple[Path, str]] = []
    for split in cfg["splits"]:
        f = raw_root / split
        if f.exists() and find_annotation_file(f):
            split_folders.append((f, split))
    if not split_folders and find_annotation_file(raw_root):
        split_folders = [(raw_root, "all")]
    if not split_folders:
        sys.exit(f"no COCO annotations found under {raw_root}")

    for folder, _ in split_folders:
        coco = load_coco(find_annotation_file(folder))
        found.update(coco.categories.values())

    order = resolve_class_order(sorted(found), cfg["classes"]["order"])
    class_to_idx = {name: i for i, name in enumerate(order)}
    print(f"class order: {order}")
    print(f"class_to_idx: {class_to_idx}\n")

    all_rows: list[dict] = []
    for folder, split in split_folders:
        all_rows += process_split(folder, split, cfg, class_to_idx,
                                  args.overwrite, args.limit)

    if not all_rows:
        sys.exit("nothing processed")

    df = pd.DataFrame(all_rows)
    manifest = proc_root / "manifest.csv"
    df.to_csv(manifest, index=False)

    with open(proc_root / "classes.json", "w", encoding="utf-8") as fh:
        json.dump({"order": order, "class_to_idx": class_to_idx}, fh, indent=2)

    print(f"\nwrote {manifest}  ({len(df)} rows)")
    print(df.groupby(["split", "class_name"]).size().to_string())
    print(f"\nbark area fraction (after letterbox): "
          f"mean={df.bark_frac.mean():.3f}  min={df.bark_frac.min():.3f}  "
          f"max={df.bark_frac.max():.3f}")

    empties = df[df.bark_frac == 0]
    if len(empties):
        print(f"!! {len(empties)} EMPTY masks — inspect before training:")
        print(empties[["split", "file_name"]].head(10).to_string(index=False))

    print("\nNext: python scripts/03_verify_masks.py   (look at the overlays)")


if __name__ == "__main__":
    main()
