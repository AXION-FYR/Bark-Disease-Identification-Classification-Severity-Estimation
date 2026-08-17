"""
Stage 1 — bark segmentation. U-Net + EfficientNet-B0 encoder.

Tuned for a 4 GB laptop GPU (RTX 3050 Ti): mixed precision, small batch,
gradient accumulation for a larger effective batch, and automatic batch-size
halving on CUDA OOM rather than a crash forty minutes in.

This stage is a standard module ADOPTED, not a novelty claim. Train it, report
IoU/Dice, freeze it, move on. Do not spend Day 2 tuning it.

Run:  python scripts/09_train_seg.py
      python scripts/09_train_seg.py --epochs 40 --batch 4
      python scripts/09_train_seg.py --smoke          # 20 steps, checks it runs
      python scripts/09_train_seg.py --resume         # continue from last.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
from torch.utils.data import DataLoader                               # noqa: E402

from src.config import load_config                                    # noqa: E402
from src.dataset import BarkSegDataset                                # noqa: E402


# --------------------------------------------------------------------------
# loss: BCE + soft Dice. BCE alone is unstable when foreground is ~18% of the
# frame; Dice alone gives weak gradients early. The sum behaves well on both.
# --------------------------------------------------------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, dice_w: float = 1.0, bce_w: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_w, self.bce_w = dice_w, bce_w

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        p = torch.sigmoid(logits)
        num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
        den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
        dice = 1 - (num / den).mean()
        return self.bce_w * bce + self.dice_w * dice


@torch.no_grad()
def iou_dice(logits, target, thr: float = 0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = ((inter + 1e-6) / (union + 1e-6))
    dice = ((2 * inter + 1e-6) / (p.sum(dim=(1, 2, 3))
                                  + target.sum(dim=(1, 2, 3)) + 1e-6))
    return iou.sum().item(), dice.sum().item(), target.size(0)


def build_model(encoder: str = "efficientnet-b0"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=1)


# --------------------------------------------------------------------------
def run_epoch(model, loader, crit, opt, scaler, device, accum: int = 1,
              train: bool = True, max_steps: int | None = None):
    model.train() if train else model.eval()
    tot_loss = tot_iou = tot_dice = n = 0
    t0 = time.time()

    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = crit(logits, y)

            if train:
                scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

        i, d, b = iou_dice(logits.float(), y)
        tot_loss += loss.item() * b
        tot_iou += i
        tot_dice += d
        n += b

        if train and step % 20 == 0:
            print(f"    step {step:4d}/{len(loader)}  loss {loss.item():.4f}",
                  end="\r", flush=True)

    return (tot_loss / max(n, 1), tot_iou / max(n, 1),
            tot_dice / max(n, 1), time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2,
                    help="gradient accumulation; effective batch = batch*accum")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2,
                    help="set 0 if Windows DataLoader misbehaves")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--encoder", default="efficientnet-b0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg.path("paths", "processed_root").parent.parent / "outputs" / "seg"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        print("  !! no CUDA. On CPU a 512px U-Net is ~20-30 min/epoch. "
              "Install the CUDA build of torch before training for real.")

    train_ds = BarkSegDataset("train", augment=True)
    val_ds = BarkSegDataset("valid", augment=False)
    print(f"train {len(train_ds)}   valid {len(val_ds)}")

    batch = args.batch
    while True:
        try:
            train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                                  num_workers=args.workers, pin_memory=True,
                                  drop_last=True, persistent_workers=args.workers > 0)
            val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True,
                                persistent_workers=args.workers > 0)

            model = build_model(args.encoder).to(device)
            crit = BCEDiceLoss()
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(args.epochs, 1))
            scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

            # one forward/backward to provoke OOM early rather than at epoch 3
            xb, yb = next(iter(train_dl))
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                loss = crit(model(xb.to(device)), yb.to(device))
            scaler.scale(loss).backward()
            opt.zero_grad(set_to_none=True)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 1:
                sys.exit("OOM even at batch size 1. Lower preprocess.size to "
                         "384 in config.yaml and rebuild the cache.")
            batch //= 2
            args.accum *= 2
            print(f"  OOM -> retrying with batch {batch}, accum {args.accum}")

    print(f"batch {batch} x accum {args.accum} = effective {batch * args.accum}")

    start_epoch, best_iou, bad = 0, -1.0, 0
    hist = []
    ckpt_last, ckpt_best = out / "last.pt", out / "best.pt"

    if args.resume and ckpt_last.exists():
        st = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_epoch, best_iou = st["epoch"] + 1, st["best_iou"]
        hist = st.get("hist", [])
        print(f"resumed from epoch {start_epoch}, best IoU {best_iou:.4f}")

    steps = 20 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs

    for ep in range(start_epoch, epochs):
        tl, ti, td, tt = run_epoch(model, train_dl, crit, opt, scaler, device,
                                   args.accum, True, steps)
        vl, vi, vd, vt = run_epoch(model, val_dl, crit, opt, scaler, device,
                                   1, False, steps)
        sched.step()

        print(f"epoch {ep + 1:3d}/{epochs}  "
              f"train loss {tl:.4f} IoU {ti:.4f}  |  "
              f"val loss {vl:.4f} IoU {vi:.4f} Dice {vd:.4f}  "
              f"({tt:.0f}s + {vt:.0f}s)")

        hist.append({"epoch": ep + 1, "train_loss": tl, "train_iou": ti,
                     "val_loss": vl, "val_iou": vi, "val_dice": vd})

        # checkpoint EVERY epoch — a crash at epoch 30 must not cost 30 epochs
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best_iou": best_iou, "hist": hist,
                    "encoder": args.encoder}, ckpt_last)

        if vi > best_iou:
            best_iou, bad = vi, 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_iou": vi, "val_dice": vd,
                        "encoder": args.encoder}, ckpt_best)
            print(f"    new best IoU {vi:.4f} -> {ckpt_best.name}")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: {args.patience} epochs without improvement")
                break

    with open(out / "history.json", "w", encoding="utf-8") as fh:
        json.dump(hist, fh, indent=2)

    print(f"\nbest val IoU: {best_iou:.4f}")
    print(f"checkpoints in {out}")
    if not args.smoke:
        print("Next: python scripts/10_eval_seg.py   (test IoU + predicted masks)")


if __name__ == "__main__":
    main()
