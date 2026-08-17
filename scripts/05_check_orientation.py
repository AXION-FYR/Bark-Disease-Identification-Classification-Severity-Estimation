"""
Step 5 — diagnose the transposed-dimension images, and prove the fix.

Two questions, answered in one run:

  Q1. Did the EXIF orientation tag survive the Roboflow export?
      If yes, read_image_rgb's exif_transpose already fixes these images and
      nothing further is needed.
      If no, the image must be rotated manually and we need the direction.

  Q2. Which rotation direction is correct?
      The script renders BOTH candidates side by side with the mask overlaid.
      You look at four pictures and pick the one where the mask sits on the
      trunk. Thirty seconds, and it is definitive in a way no heuristic is.

Run:  python scripts/05_check_orientation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config                                    # noqa: E402
from src.coco_utils import find_annotation_file, load_coco            # noqa: E402
from src.coco_utils import polygons_to_mask                           # noqa: E402
from src.imaging import overlay_mask                                  # noqa: E402

from PIL import Image, ImageOps                                       # noqa: E402
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

EXIF_ORIENTATION_TAG = 274


def raw_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size                       # (w, h), EXIF NOT applied


def exif_size_and_tag(path: Path) -> tuple[tuple[int, int], int | None]:
    with Image.open(path) as im:
        tag = None
        try:
            exif = im.getexif()
            if exif:
                tag = exif.get(EXIF_ORIENTATION_TAG)
        except Exception:
            pass
        fixed = ImageOps.exif_transpose(im)
        return fixed.size, tag


def main() -> None:
    cfg = load_config()
    raw_root = cfg.path("paths", "raw_root")
    qc = cfg.path("paths", "qc_root")
    qc.mkdir(parents=True, exist_ok=True)

    affected = []          # (folder, record)
    for split in cfg["splits"] + ["."]:
        folder = raw_root / split if split != "." else raw_root
        ann = find_annotation_file(folder)
        if ann is None:
            continue
        for rec in load_coco(ann):
            fp = folder / rec.file_name
            if not fp.exists():
                continue
            w, h = raw_size(fp)
            if (w, h) != (rec.width, rec.height):
                affected.append((folder, rec, (w, h)))

    if not affected:
        print("No dimension mismatches. Nothing to fix.")
        return

    print(f"{len(affected)} image(s) whose stored size differs from the JSON.\n")

    # ---- Q1: does exif_transpose resolve it?
    fixed_by_exif, still_broken, tags = 0, [], {}
    for folder, rec, _ in affected:
        fp = folder / rec.file_name
        size, tag = exif_size_and_tag(fp)
        tags[tag] = tags.get(tag, 0) + 1
        if size == (rec.width, rec.height):
            fixed_by_exif += 1
        else:
            still_broken.append((folder, rec))

    print(f"EXIF orientation tags found: {tags}")
    print(f"  (tag 1 = normal, 6 = rotate 90 CW, 8 = rotate 90 CCW, "
          f"None = tag stripped)")
    print(f"\nresolved by exif_transpose : {fixed_by_exif} / {len(affected)}")
    print(f"still mismatched           : {len(still_broken)}")

    if not still_broken:
        print("\n=> The EXIF tag survived the export. src/imaging.read_image_rgb "
              "already applies it, so every mask will align. No re-export, no "
              "config change. Run 02_build_masks.py, then check the overlays.")
    else:
        print("\n=> EXIF was stripped on export for some files. Those must be "
              "rotated manually. Pick the direction from the figure below and "
              "set preprocess.rotate_dir in config.yaml.")

    # ---- Q2: render both candidates for a few affected images
    sample = affected[:4]
    fig, axes = plt.subplots(len(sample), 3, figsize=(15, 5 * len(sample)))
    axes = np.atleast_2d(axes)

    for r, (folder, rec, stored) in enumerate(sample):
        fp = folder / rec.file_name
        with Image.open(fp) as im:
            base = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))

        mask = polygons_to_mask(rec, h=rec.height, w=rec.width)

        cands = [("exif_transpose (current default)", base)]
        with Image.open(fp) as im:
            rawim = np.asarray(im.convert("RGB"))
        cands.append(("manual rotate CW", cv2.rotate(rawim, cv2.ROTATE_90_CLOCKWISE)))
        cands.append(("manual rotate CCW",
                      cv2.rotate(rawim, cv2.ROTATE_90_COUNTERCLOCKWISE)))

        for c, (label, img) in enumerate(cands):
            ax = axes[r, c]
            if img.shape[:2] == mask.shape[:2]:
                ax.imshow(overlay_mask(img, mask))
                ax.set_title(f"{label}\n{img.shape[1]}x{img.shape[0]}  MATCHES",
                             fontsize=9)
            else:
                ax.imshow(img)
                ax.set_title(f"{label}\n{img.shape[1]}x{img.shape[0]}  "
                             f"(mask is {mask.shape[1]}x{mask.shape[0]})",
                             fontsize=9, color="red")
            ax.axis("off")
        axes[r, 0].set_ylabel(rec.file_name[:20], fontsize=7)

    fig.suptitle("Pick the panel where the mask sits ON the trunk", fontsize=14)
    fig.tight_layout()
    out = qc / "orientation_candidates.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out} — open it and choose.")

    with open(qc / "orientation_affected.txt", "w", encoding="utf-8") as fh:
        for folder, rec, stored in affected:
            fh.write(f"{folder.name}\t{rec.file_name}\tstored={stored}\t"
                     f"json=({rec.width},{rec.height})\n")
    print(f"full list: {qc / 'orientation_affected.txt'}")


if __name__ == "__main__":
    main()
