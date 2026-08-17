"""
Step 11 — train the Stage 2 classifier, with ablation variants.

Runs one variant per invocation and appends a row to
outputs/cls/ablation.csv, so after running all six you have the table that
IS your Novelty Claim 1 defence:

    variant        test_acc  macro_f1
    plain            ...       ...
    masked           ...       ...     <- does masking help?
    concat           ...       ...     <- does texture help?
    se               ...       ...     <- does fusion type matter?
    mcse (OURS)      ...       ...
    mcse_allones     ...       ...     <- isolates the mask conditioning

If mcse does not beat se and concat, you do not have Claim 1 — you have a
tuning result. Better to know that from this table than from a viva.

Masks are the Stage-1 PREDICTED masks (mask_source="pred"), so the numbers are
honest end-to-end. Pass --mask_source gt for the clean-mask upper bound.

Run:  python scripts/11_train_cls.py --variant mcse
      python scripts/11_train_cls.py --variant plain --epochs 30
      python scripts/11_train_cls.py --all           # every variant in sequence
      python scripts/11_train_cls.py --variant mcse --smoke
"""
from __future__ import annotations

import argparse
import csv
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
from src.dataset_cls import BarkClsDataset                            # noqa: E402
from src.model_cls import build_classifier                           # noqa: E402

class FocalLoss(nn.Module):
    """
    Focal loss: down-weights easy examples so training focuses on the hard,
    misclassified cases — here, mild rough bark that looks like healthy bark.
    gamma controls the focusing; alpha is the per-class weight (inverse-freq).
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing,
                                      reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)               # per-sample CE
        pt = torch.exp(-ce)                         # prob of the true class
        return ((1 - pt) ** self.gamma * ce).mean()


VARIANTS = ["plain", "masked", "concat", "se", "mcse", "mcse_allones"]


def macro_f1(y_true, y_pred, n_cls: int) -> float:
    f1s = []
    for c in range(n_cls):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate(model, loader, device, n_cls):
    model.eval()
    yt, yp = [], []
    for img, mask, y in loader:
        img, mask = img.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(img, mask)
        yp.append(logits.argmax(1).cpu().numpy())
        yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    acc = float((yt == yp).mean())
    f1 = macro_f1(yt, yp, n_cls)
    cm = np.zeros((n_cls, n_cls), int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return acc, f1, cm


SEED = 42


def train_one(variant: str, args, cfg, device) -> dict:
    global SEED
    SEED = args.seed
    print(f"\n{'=' * 60}\nvariant: {variant}\n{'=' * 60}")

    ms = args.mask_source
    train_ds = BarkClsDataset("train", size=args.size, train=True,
                              mask_source=ms)
    val_ds = BarkClsDataset("valid", size=args.size, train=False, mask_source=ms)
    test_ds = BarkClsDataset("test", size=args.size, train=False, mask_source=ms)

    # inverse-frequency class weights from the training split
    ys = train_ds.df.class_idx.values
    counts = np.bincount(ys, minlength=3).astype(np.float32)
    cls_w = torch.tensor(counts.sum() / (len(counts) * counts)).to(device)

    def mk(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=shuffle,
                          persistent_workers=args.workers > 0)

    train_dl, val_dl, test_dl = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    model = build_classifier(variant=variant, num_classes=3).to(device)
    if args.focal:
        crit = FocalLoss(alpha=cls_w, gamma=args.gamma, label_smoothing=0.05)
    else:
        crit = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    out = cfg.path("paths", "processed_root").parent.parent / "outputs" / "cls"
    out.mkdir(parents=True, exist_ok=True)

    best_f1, bad, best_state = -1.0, 0, None
    epochs = 1 if args.smoke else args.epochs
    max_steps = 15 if args.smoke else None

    for ep in range(epochs):
        model.train()
        t0, run = time.time(), 0.0
        for step, (img, mask, y) in enumerate(train_dl):
            if max_steps and step >= max_steps:
                break
            img, mask, y = img.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                loss = crit(model(img, mask), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
        sched.step()

        vacc, vf1, _ = evaluate(model, val_dl, device, 3)
        print(f"  epoch {ep + 1:2d}/{epochs}  loss {run / max(step, 1):.4f}  "
              f"val acc {vacc:.4f}  val f1 {vf1:.4f}  ({time.time() - t0:.0f}s)")

        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    tacc, tf1, cm = evaluate(model, test_dl, device, 3)
    print(f"  TEST  acc {tacc:.4f}  macro-f1 {tf1:.4f}")
    print(f"  confusion matrix (rows=true [rb, healthy, sc]):\n{cm}")

    vname = variant + ("_focal" if args.focal else "")

    torch.save({"model": model.state_dict(), "variant": variant,
                "test_acc": tacc, "test_f1": tf1,
                "mask_source": ms}, out / f"cls_{vname}.pt")

    return {"variant": vname, "mask_source": ms, "seed": SEED,
            "val_f1": round(best_f1, 4),
            "test_acc": round(tacc, 4), "test_f1": round(tf1, 4),
            "cm": cm.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mcse", choices=VARIANTS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mask_source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--focal", action="store_true",
                    help="use focal loss (targets hard rough-bark cases)")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="focal focusing parameter")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed; run 2-3 seeds and report mean +/- std")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}   mask_source: {args.mask_source}")

    variants = VARIANTS if args.all else [args.variant]
    out = cfg.path("paths", "processed_root").parent.parent / "outputs" / "cls"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ablation.csv"

    results = []
    for v in variants:
        # batch auto-halve on OOM
        b = args.batch
        while True:
            try:
                args.batch = b
                results.append(train_one(v, args, cfg, device))
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if b <= 2:
                    print(f"  OOM at batch 2 on {v}; skipping")
                    break
                b //= 2
                print(f"  OOM -> retrying {v} at batch {b}")

    # append/update the ablation table
    existing = {}
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                existing[(r["variant"], r["mask_source"], r.get("seed"))] = r
    for r in results:
        existing[(r["variant"], r["mask_source"], r.get("seed"))] = {
            k: v for k, v in r.items() if k != "cm"}

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "mask_source", "seed",
                                           "val_f1", "test_acc", "test_f1"])
        w.writeheader()
        order = {v: i for i, v in enumerate(VARIANTS)}
        def _seed_key(k):
            try:
                return int(k[2]) if k[2] not in (None, "") else 0
            except (ValueError, TypeError):
                return 0
        for key in sorted(existing, key=lambda k: (k[1], order.get(k[0], 99), _seed_key(k))):
            w.writerow(existing[key])

    with open(out / "ablation_full.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'=' * 60}\nABLATION TABLE -> {csv_path}\n{'=' * 60}")
    with open(csv_path) as fh:
        print(fh.read())
    print("Read it: mcse should beat se and concat. If it does not, that is a "
          "finding to report, not to hide.")


if __name__ == "__main__":
    main()
