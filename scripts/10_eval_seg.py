"""
Step 10 — evaluate Stage 1 on test, and cache predicted masks.

Produces:
  outputs/seg/test_metrics.json      IoU/Dice overall and per disease class
  outputs/seg/overlay_worst.png      the 8 worst predictions — look at these
  data/processed/<split>/pred_masks/ predicted bark masks for every image

Why per-class IoU matters here: if segmentation is systematically worse on one
disease, every downstream number for that class inherits the error, and you
want to know that before you interpret a confusion matrix.

Why cache predicted masks: Stage 2 and 3 train on ground-truth masks, but at
inference no ground truth exists. Reporting test accuracy with PREDICTED masks
as well as ground-truth masks is the honest end-to-end number, and the gap
between them tells you what Stage 1 costs the pipeline.

Run:  python scripts/10_eval_seg.py
      python scripts/10_eval_seg.py --ckpt outputs/seg/best.pt
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

import torch                                                          # noqa: E402
from torch.utils.data import DataLoader                               # noqa: E402

from src.config import load_config, PROJECT_ROOT                      # noqa: E402
from src.dataset import BarkSegDataset                                # noqa: E402
from src.imaging import read_image_rgb, write_mask_png, overlay_mask  # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config()
    proc = cfg.path("paths", "processed_root")
    out = PROJECT_ROOT / "outputs" / "seg"
    ckpt_path = Path(args.ckpt) if args.ckpt else (out / "best.pt")
    if not ckpt_path.exists():
        sys.exit(f"{ckpt_path} not found — run scripts/09_train_seg.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt_path, map_location=device, weights_only=False)

    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=st.get("encoder", "efficientnet-b0"),
                     encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(st["model"])
    model.eval()
    print(f"loaded {ckpt_path.name} (epoch {st.get('epoch', '?')}, "
          f"val IoU {st.get('val_iou', float('nan')):.4f})")

    manifest = pd.read_csv(proc / "manifest.csv")
    results = {}

    for split in ["train", "valid", "test"]:
        try:
            ds = BarkSegDataset(split, augment=False, return_meta=True)
        except (ValueError, FileNotFoundError):
            continue

        pred_dir = proc / split / "pred_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers,
                        collate_fn=lambda b: b)          # keep meta dicts intact

        for batch in tqdm(dl, desc=f"[{split}]", unit="batch"):
            xs = torch.stack([b[0] for b in batch]).to(device)
            ys = torch.stack([b[1] for b in batch]).to(device)
            metas = [b[2] for b in batch]

            with torch.no_grad():
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    logits = model(xs)
            p = (torch.sigmoid(logits.float()) > args.thr).float()

            inter = (p * ys).sum(dim=(1, 2, 3))
            union = p.sum(dim=(1, 2, 3)) + ys.sum(dim=(1, 2, 3)) - inter
            iou = ((inter + 1e-6) / (union + 1e-6)).cpu().numpy()
            dice = ((2 * inter + 1e-6) /
                    (p.sum(dim=(1, 2, 3)) + ys.sum(dim=(1, 2, 3)) + 1e-6)
                    ).cpu().numpy()

            pn = p.cpu().numpy().astype(np.uint8)
            for k, meta in enumerate(metas):
                write_mask_png(pred_dir / f"{meta['stem']}.png", pn[k, 0])
                rows.append({"split": split, "stem": meta["stem"],
                             "class_name": meta["class_name"],
                             "iou": float(iou[k]), "dice": float(dice[k])})

        rdf = pd.DataFrame(rows)
        results[split] = {
            "n": len(rdf),
            "iou_mean": float(rdf.iou.mean()),
            "dice_mean": float(rdf.dice.mean()),
            "per_class": rdf.groupby("class_name")[["iou", "dice"]]
                            .mean().round(4).to_dict("index"),
        }
        print(f"\n[{split}] IoU {rdf.iou.mean():.4f}   Dice {rdf.dice.mean():.4f}")
        print(rdf.groupby("class_name")[["iou", "dice"]]
                 .agg(["count", "mean"]).round(4).to_string())

        if split == "test":
            worst = rdf.nsmallest(8, "iou")
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for ax, (_, r) in zip(axes.ravel(), worst.iterrows()):
                m = manifest[manifest.stem == r.stem].iloc[0]
                img = read_image_rgb(PROJECT_ROOT / m.image_path)
                data = np.fromfile(str(pred_dir / f"{r.stem}.png"), np.uint8)
                import cv2
                pm = (cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
                ax.imshow(overlay_mask(img, pm))
                ax.set_title(f"{r.class_name}  IoU {r.iou:.2f}\n{r.stem[:24]}",
                             fontsize=8)
                ax.axis("off")
            fig.suptitle("Worst test predictions (yellow = PREDICTED bark)",
                         fontsize=13)
            fig.tight_layout()
            fig.savefig(out / "overlay_worst.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"\nwrote {out / 'overlay_worst.png'}")

        rdf.to_csv(out / f"per_image_{split}.csv", index=False)

    with open(out / "test_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nmetrics -> {out / 'test_metrics.json'}")
    print(f"predicted masks -> {proc}/<split>/pred_masks/")
    print("\nStage 1 is now CLOSED. Do not tune it further. Next: Stage 2.")


if __name__ == "__main__":
    main()
