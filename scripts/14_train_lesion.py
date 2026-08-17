"""
Step 14 — train the weakly-supervised lesion decoder.  Novelty Claims 2 & 3.

Trains on ALL train images (healthy + diseased) using only image-level labels
and the Stage-1 bark masks. No lesion labels are used in training — they exist
only for evaluation (step 15).

Warm-start schedule (important): the first `warmup` epochs use only the
healthy-anchor + bark-containment losses. This teaches the map to be zero on
healthy bark and outside bark BEFORE the MIL term starts pushing diseased maps
up. Skipping the warm-start is the fastest way to a degenerate all-zero or
all-one map.

Encoder is frozen and warm-started from the Stage-2 dualbranch_se checkpoint.

Run:  python scripts/14_train_lesion.py --stage2_ckpt outputs/cls/cls_mcse_allones.pt
      python scripts/14_train_lesion.py --stage2_ckpt ... --no_film      # ablation
      python scripts/14_train_lesion.py --stage2_ckpt ... --smoke
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
from torch.utils.data import DataLoader                               # noqa: E402

from src.config import load_config                                    # noqa: E402
from src.dataset_cls import BarkClsDataset                            # noqa: E402
from src.model_lesion import build_lesion_decoder                    # noqa: E402
from src.losses_lesion import lesion_loss                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_ckpt", required=True,
                    help="path to the Stage-2 checkpoint, e.g. "
                         "outputs/cls/cls_mcse_allones.pt")
    ap.add_argument("--no_film", action="store_true",
                    help="disable class conditioning (ablation control)")
    ap.add_argument("--mask_source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5,
                    help="epochs of anchor+containment only before MIL")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--w_anchor", type=float, default=1.0)
    ap.add_argument("--w_contain", type=float, default=2.0)
    ap.add_argument("--w_mil", type=float, default=1.0)
    ap.add_argument("--w_tv", type=float, default=0.05)
    ap.add_argument("--w_sparse", type=float, default=0.3)
    ap.add_argument("--healthy_idx", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    film = not args.no_film
    tag = "film" if film else "nofilm"
    print(f"device: {device}   film: {film}   mask_source: {args.mask_source}")

    if not Path(args.stage2_ckpt).exists():
        sys.exit(f"Stage-2 checkpoint not found: {args.stage2_ckpt}")

    # NOTE: apply_mask=False — the decoder needs the FULL image (lesion texture
    # lives in the pixels); the bark mask is passed separately to the losses.
    train_ds = BarkClsDataset("train", size=args.size, train=True,
                              mask_source=args.mask_source, apply_mask=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=args.workers > 0)

    model = build_lesion_decoder(args.stage2_ckpt, class_conditioned=film).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    out = cfg.path("paths", "processed_root").parent.parent / "outputs" / "lesion"
    out.mkdir(parents=True, exist_ok=True)

    epochs = 1 if args.smoke else args.epochs
    max_steps = 15 if args.smoke else None
    hist = []

    for ep in range(epochs):
        model.train()          # decoder to train; frozen encoder stays eval
        enable_mil = ep >= args.warmup
        t0 = time.time()
        agg = {"anchor": 0.0, "contain": 0.0, "mil": 0.0, "tv": 0.0, "sparse": 0.0, "total": 0.0}
        nb = 0

        for step, (img, mask, y) in enumerate(train_dl):
            if max_steps and step >= max_steps:
                break
            img, mask, y = img.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logit = model(img, y)
                loss, parts = lesion_loss(
                    logit, mask, y, healthy_idx=args.healthy_idx,
                    w_anchor=args.w_anchor, w_contain=args.w_contain,
                    w_mil=args.w_mil, w_tv=args.w_tv, w_sparse=args.w_sparse,
                    enable_mil=enable_mil)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            for k in parts:
                agg[k] += parts[k]
            agg["total"] += loss.item()
            nb += 1

        sched.step()
        for k in agg:
            agg[k] /= max(nb, 1)
        phase = "warmup" if not enable_mil else "full"
        print(f"epoch {ep + 1:2d}/{epochs} [{phase:6s}] "
              f"total {agg['total']:.4f}  anchor {agg['anchor']:.4f}  "
              f"contain {agg['contain']:.4f}  mil {agg['mil']:.4f}  "
              f"sparse {agg['sparse']:.4f}  tv {agg['tv']:.4f}  "
              f"({time.time() - t0:.0f}s)")
        hist.append({"epoch": ep + 1, "phase": phase, **agg})

        # checkpoint every epoch
        torch.save({"model": model.state_dict(), "film": film,
                    "stage2_ckpt": args.stage2_ckpt, "epoch": ep,
                    "args": vars(args)}, out / f"lesion_{tag}_last.pt")

    torch.save({"model": model.state_dict(), "film": film,
                "stage2_ckpt": args.stage2_ckpt, "epoch": epochs - 1,
                "args": vars(args)}, out / f"lesion_{tag}.pt")
    with open(out / f"lesion_{tag}_history.json", "w") as fh:
        json.dump(hist, fh, indent=2)

    print(f"\nsaved {out / f'lesion_{tag}.pt'}")
    if not args.smoke:
        print("Next: python scripts/15_eval_lesion.py "
              f"--lesion_ckpt outputs/lesion/lesion_{tag}.pt "
              "--lesion_root <your annotation folder>")


if __name__ == "__main__":
    main()
