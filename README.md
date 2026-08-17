# Cinnamon bark — Stage 1 preprocessing & bark mask generation

Turns a Roboflow COCO-segmentation export into binary bark masks plus a
preprocessed cache, with the audit and QC steps that catch the failures which
are invisible in metrics.

## Install

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

`pycocotools` is deliberately **not** required — polygons are rasterised with
OpenCV. It is imported lazily only if your export contains RLE segmentation,
in which case the error message tells you.

## Put the data in place

Export from Roboflow as **COCO Segmentation**, with **auto-orient ON**,
**resize OFF**, **augmentation OFF**. Unzip into `data/raw/`:

```
data/raw/
├── train/  _annotations.coco.json  +  *.jpg
├── valid/  _annotations.coco.json  +  *.jpg
└── test/   _annotations.coco.json  +  *.jpg
```

A flat single-folder export also works.

## Run, in order

```bash
python scripts/01_audit_coco.py       # read the output before continuing
python scripts/02_build_masks.py      # writes masks + manifest.csv
python scripts/03_verify_masks.py     # writes overlays to outputs/qc/ — LOOK at them
python scripts/04_phash_check.py      # split-leakage check; can run during training
```

Smoke-test on 20 images first: `python scripts/02_build_masks.py --limit 20`

## What to check in the step-01 output

| Output | Meaning |
|---|---|
| median area fraction 0.15–0.60 | whole-trunk bark masks, as expected |
| median area fraction < 0.05 | these are **lesion** polygons, not trunk masks — the plan changes, Stage 3 becomes supervised |
| mixed-class images > 0 | a trunk annotated with two diseases. Decide now: exclude, or take majority area (the default). Don't discover this on Day 5 |
| file/JSON dimension mismatch | the auto-orient bug — masks will be rotated relative to images. Re-export |
| images with 0 annotations | breaks every downstream stage silently. Drop or annotate |
| per-annotation ≠ per-image counts | some images carry several polygons; stratify the split on the **per-image** class |

## Output layout

```
data/processed/
├── train/images/*.png      512×512 letterboxed RGB
├── train/masks/*.png       512×512 binary 0/255
├── valid/… test/…
├── manifest.csv            one row per image
└── classes.json            {'Rough bark': 0, 'healthy bark': 1, 'stripecanker': 2}
```

`manifest.csv` columns worth knowing: `class_idx` (Stage 2 label),
`bark_frac` (sanity metric), `bbox_*` (the bark bounding box, used on Day 3 to
crop before the classifier), `lb_*` (letterbox parameters, so predictions can
be mapped back to original image coordinates).

## Two design choices that matter downstream

**Letterbox, not square-resize.** Stretching a tall trunk to a square changes
aspect ratio, and stripe canker's signature is vertical anisotropy — the exact
property the class-conditioned decoder depends on. Padding preserves it, and
the padded region is background that the bark mask excludes anyway.

**Masks are resized with INTER_NEAREST, images with INTER_AREA/LINEAR.**
Bilinear interpolation of a binary mask creates fractional labels along every
boundary. `src/imaging.letterbox_mask` enforces this.

All three disease classes collapse to a single foreground label in the mask —
Stage 1 is bark-vs-background. The disease class is carried in `manifest.csv`
for Stage 2.

## Day 2

```python
from src.dataset import BarkSegDataset
train_ds = BarkSegDataset("train", augment=True)
val_ds   = BarkSegDataset("valid", augment=False)
```

Needs `pip install albumentations` for augmentation. Rotation is capped at ±15°
and shear/perspective are excluded on purpose; flips on both axes are safe.

## Delete before submitting

`make_fake_data.py` is a synthetic-data generator used to test the pipeline.
It has no role in the research.
