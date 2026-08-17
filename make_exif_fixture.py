"""Reproduce the real bug: landscape file bytes + EXIF orientation 6,
JSON recording portrait dims, polygon drawn in the portrait frame."""
import json
from pathlib import Path
import numpy as np, cv2, piexif
from PIL import Image

ROOT = Path(__file__).parent / "data/raw"
CLASSES = ["Rough bark", "healthy bark", "stripecanker"]

for split, n in [("train", 6), ("valid", 4), ("test", 4)]:
    d = ROOT / split; d.mkdir(parents=True, exist_ok=True)
    images, anns, aid = [], [], 1
    for i in range(n):
        # PORTRAIT display frame: 300 wide x 500 tall. Trunk is a vertical band.
        DW, DH = 300, 500
        disp = np.random.randint(30, 70, (DH, DW, 3), np.uint8)
        x0, x1 = 90, 210
        cv2.rectangle(disp, (x0, 0), (x1, DH), (110, 85, 60), -1)
        poly = [x0, 0, x1, 0, x1, DH - 1, x0, DH - 1]

        exif_bug = (i % 2 == 0)   # half the images carry the bug
        if exif_bug:
            # store the LANDSCAPE bytes + orientation tag 6 (= rotate 90 CW to view)
            stored = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            Image.fromarray(stored).save(d / f"{split}_{i:03d}.jpg", quality=95,
                exif=piexif.dump({"0th": {piexif.ImageIFD.Orientation: 6}}))
        else:
            Image.fromarray(disp).save(d / f"{split}_{i:03d}.jpg", quality=95)

        images.append({"id": i, "file_name": f"{split}_{i:03d}.jpg",
                       "width": DW, "height": DH})     # JSON = DISPLAY frame
        anns.append({"id": aid, "image_id": i,
                     "category_id": (i % 3) + 1, "segmentation": [poly],
                     "area": float((x1 - x0) * DH), "iscrowd": 0,
                     "bbox": [x0, 0, x1 - x0, DH]}); aid += 1

    cats = [{"id": 0, "name": "cinnamon", "supercategory": "none"}] + \
           [{"id": k + 1, "name": c, "supercategory": "cinnamon"}
            for k, c in enumerate(CLASSES)]
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(d / "_annotations.coco.json", "w"))
print("EXIF-bug fixture written to", ROOT)
