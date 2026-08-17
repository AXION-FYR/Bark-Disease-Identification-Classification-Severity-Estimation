"""
Step 25b — export a TREE-LEVEL grading pack for the expert.

Unlike the image-level grading pack (25a), this asks the expert to look at ALL
~15 photos of ONE tree together and assign a single treatment stage per disease
for that whole tree -- the same judgement your pipeline's Stage 5 (aggregation)
is trying to automate. This validates the aggregation logic itself, not just
the per-image QSI.

For each tree, builds a contact-sheet image (a grid of all its raw photos, no
model output, no lesion overlays, no predictions) so grading stays blind. Also
writes a blank grading sheet with one row per (tree, disease-if-present) for
the expert to fill in.

Since a tree can show more than one disease, the expert is asked to grade EACH
disease they see on the tree separately (e.g. "this tree shows rough bark at
Active management AND early stripe canker at Preventive").

Run:  python scripts/25b_tree_grading_pack.py --tree_root D:/RESEARCH/Dataset/multiview_trees
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT                                    # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def make_contact_sheet(photos: list[Path], thumb=220, cols=5) -> np.ndarray:
    """Grid of raw photo thumbnails, no predictions, no overlays -- for blind grading."""
    n = len(photos)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * thumb, cols * thumb, 3), 255, np.uint8)
    for i, p in enumerate(photos):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = thumb / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        r, c = divmod(i, cols)
        y0 = r * thumb + (thumb - img.shape[0]) // 2
        x0 = c * thumb + (thumb - img.shape[1]) // 2
        sheet[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.putText(sheet, f"{i+1}", (c * thumb + 6, r * thumb + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree_root", required=True,
                    help="folder containing one subfolder of photos per tree")
    ap.add_argument("--thumb", type=int, default=220)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.tree_root)
    if not root.exists():
        sys.exit(f"folder not found: {root}")

    tree_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not tree_dirs:
        sys.exit(f"no tree subfolders found in {root}")

    pack = PROJECT_ROOT / "outputs" / "tree_grading_pack"
    sheets_dir = pack / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for td in tree_dirs:
        photos = sorted([f for f in td.iterdir() if f.suffix.lower() in IMG_EXTS])
        if not photos:
            continue
        sheet = make_contact_sheet(photos, args.thumb, args.cols)
        out_path = sheets_dir / f"{td.name}_contact_sheet.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        manifest_rows.append({"tree": td.name, "n_photos": len(photos),
                              "contact_sheet": out_path.name})
        print(f"  {td.name}: {len(photos)} photos -> {out_path.name}")

    # blank grading sheet: two rows per tree (rough bark, stripe canker) so the
    # expert can grade whichever disease(s) they actually see, leaving the
    # other blank/N/A if that disease isn't present on this tree.
    sheet_path = pack / "tree_grading_sheet.csv"
    with open(sheet_path, "w") as fh:
        fh.write("tree,disease,expert_stage,expert_notes\n")
        for r in manifest_rows:
            fh.write(f"{r['tree']},Rough bark,,\n")
            fh.write(f"{r['tree']},stripecanker,,\n")

    instructions = pack / "INSTRUCTIONS.txt"
    instructions.write_text(
        "TREE-LEVEL SEVERITY GRADING INSTRUCTIONS\n"
        "=========================================\n\n"
        "For each tree, open its contact sheet (contact_sheets/<tree>_contact_sheet.png).\n"
        "It shows ALL photos taken of that one tree, numbered, with NO model\n"
        "predictions or overlays -- please grade based on your own visual judgement only.\n\n"
        "For each disease you can see present anywhere on the tree (rough bark\n"
        "and/or stripe canker), assign ONE overall treatment stage for that\n"
        "disease considering the tree as a whole -- not per photo:\n\n"
        "  P = Preventive          (no/trace disease, monitor only)\n"
        "  E = Early control       (limited, localised disease)\n"
        "  A = Active management   (significant, spreading disease)\n"
        "  S = Severe outbreak     (majority affected / girdling risk)\n\n"
        "Fill in the 'expert_stage' column in tree_grading_sheet.csv with P/E/A/S.\n"
        "If a disease is NOT present on a tree at all, leave that row blank.\n"
        "Use 'expert_notes' for anything relevant (e.g. 'looks like it may be\n"
        "girdling on one side', 'hard to tell from photos, would want field visit').\n\n"
        "Please grade blind: do not look at any pipeline output before grading.\n",
        encoding="utf-8")

    print(f"\n{len(manifest_rows)} tree contact sheets -> {sheets_dir}")
    print(f"grading sheet -> {sheet_path}")
    print(f"instructions -> {instructions}")
    print("\nSend the whole outputs/tree_grading_pack/ folder to your expert.")


if __name__ == "__main__":
    main()
