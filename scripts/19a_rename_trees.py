"""
Step 19a — standardise multi-view tree folder filenames.

Walks the parent folder (t1/, t2/, ... or T1/, tree1/, etc.), and renames every
image inside each tree folder to:

    t<TREE>_p<PHOTO>.JPG      e.g. t01_p01.JPG, t01_p02.JPG, ...

Tree number comes from the folder name (any trailing digits: 't1', 'tree_5',
'T12' all work). Photo numbers are assigned 1..N in sorted order within each
folder. No disease is put in the name — your classifier assigns disease per
photo downstream, which tests the full pipeline and handles trees with two
diseases automatically.

SAFETY:
  * DRY-RUN by default: prints what WOULD happen, changes nothing. Add --apply
    to actually rename.
  * renames via a temporary suffix first, so it cannot clobber a file that
    already has a target name.
  * writes a mapping CSV (old -> new) so every rename is reversible.

Run:  python scripts/19a_rename_trees.py --root D:/RESEARCH/Dataset/multiview_trees
      python scripts/19a_rename_trees.py --root D:/RESEARCH/Dataset/multiview_trees --apply
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def tree_number(folder_name: str) -> int | None:
    """'t1'->1, 'tree_05'->5, 'T12'->12. Returns None if no trailing digits."""
    m = re.search(r"(\d+)\s*$", folder_name)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="parent folder holding the tree subfolders")
    ap.add_argument("--apply", action="store_true",
                    help="actually rename (default is a dry run)")
    ap.add_argument("--ext", default=".JPG",
                    help="output extension (default .JPG)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"not found: {root}")

    folders = sorted([d for d in root.iterdir() if d.is_dir()],
                     key=lambda d: (tree_number(d.name) or 9999, d.name))
    if not folders:
        sys.exit(f"no subfolders under {root}")

    mapping = []          # (old_path, new_name)
    warnings = []
    tree_ids_seen = {}

    for folder in folders:
        tnum = tree_number(folder.name)
        if tnum is None:
            warnings.append(f"folder '{folder.name}' has no number — skipped")
            continue
        if tnum in tree_ids_seen:
            warnings.append(f"tree number {tnum} appears twice "
                            f"('{tree_ids_seen[tnum]}' and '{folder.name}') "
                            f"— photos will be merged in numbering, check this")
        tree_ids_seen[tnum] = folder.name

        imgs = sorted([f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in IMG_EXTS],
                      key=lambda f: f.name.lower())
        if not imgs:
            warnings.append(f"'{folder.name}' has no images")
            continue

        for i, img in enumerate(imgs, start=1):
            new_name = f"t{tnum:02d}_p{i:02d}{args.ext}"
            mapping.append((img, new_name))

    # ---- report
    print(f"parent: {root}")
    print(f"tree folders: {len(tree_ids_seen)}   images to rename: {len(mapping)}")
    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  ! {w}")

    print("\nsample of planned renames:")
    for old, new in mapping[:8]:
        print(f"  {old.parent.name}/{old.name}  ->  {new}")
    if len(mapping) > 8:
        print(f"  ... and {len(mapping) - 8} more")

    # ---- write mapping CSV (always, even in dry run, for the record)
    map_csv = root / "rename_mapping.csv"
    with open(map_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tree_folder", "old_name", "new_name"])
        for old, new in mapping:
            w.writerow([old.parent.name, old.name, new])
    print(f"\nmapping written: {map_csv}")

    if not args.apply:
        print("\nDRY RUN — nothing changed. Re-run with --apply to rename.")
        return

    # ---- apply, two-phase to avoid collisions
    print("\napplying renames (two-phase)...")
    tmp = []
    for old, new in mapping:
        tmp_path = old.with_name(old.name + ".__tmp_rename__")
        old.rename(tmp_path)
        tmp.append((tmp_path, old.with_name(new)))
    for tmp_path, final_path in tmp:
        tmp_path.rename(final_path)

    print(f"done — renamed {len(mapping)} files.")
    print(f"to undo, use {map_csv} (columns tree_folder, old_name, new_name).")


if __name__ == "__main__":
    main()
