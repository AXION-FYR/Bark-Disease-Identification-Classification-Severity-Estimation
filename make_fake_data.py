"""Fabricate a small Roboflow-style COCO export to smoke-test the pipeline."""
import json, random
from pathlib import Path
import numpy as np, cv2

random.seed(0); np.random.seed(0)
ROOT = Path(__file__).parent / "data/raw"
CLASSES = ["Rough bark", "healthy bark", "stripecanker"]

for split, n in [("train", 12), ("valid", 4), ("test", 4)]:
    d = ROOT / split; d.mkdir(parents=True, exist_ok=True)
    images, anns, aid = [], [], 1
    for i in range(n):
        W, H = random.choice([(480, 640), (600, 900), (900, 600)])  # incl. landscape
        img = np.random.randint(30, 70, (H, W, 3), np.uint8)
        cx, tw = W // 2, int(W * random.uniform(0.3, 0.5))
        x0, x1 = cx - tw // 2, cx + tw // 2
        cv2.rectangle(img, (x0, 0), (x1, H), (110, 85, 60), -1)
        cls = CLASSES[i % 3]
        poly = [x0, 0, x1, 0, x1, H - 1, x0, H - 1]
        area = float(tw * H)
        cv2.imwrite(str(d / f"{split}_{i:03d}.jpg"), img)
        images.append({"id": i, "file_name": f"{split}_{i:03d}.jpg",
                       "width": W, "height": H})
        anns.append({"id": aid, "image_id": i, "category_id": CLASSES.index(cls) + 1,
                     "segmentation": [poly], "area": area, "iscrowd": 0,
                     "bbox": [x0, 0, tw, H]}); aid += 1
        # one image gets a second polygon of a different class (mixed-class case)
        if split == "train" and i == 5:
            anns.append({"id": aid, "image_id": i, "category_id": 1,
                         "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]],
                         "area": 2500.0, "iscrowd": 0, "bbox": [10, 10, 50, 50]}); aid += 1
    cats = [{"id": 0, "name": "cinnamon", "supercategory": "none"}] + \
           [{"id": k + 1, "name": c, "supercategory": "cinnamon"} for k, c in enumerate(CLASSES)]
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(d / "_annotations.coco.json", "w"))
print("fake data written to", ROOT)
