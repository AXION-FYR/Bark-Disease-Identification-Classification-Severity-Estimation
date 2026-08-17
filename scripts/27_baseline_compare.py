"""
Step 27 — baseline comparison against EXISTING off-the-shelf models.

Trains standard, widely-used classifiers (ResNet-50, plain EfficientNet-B0, and
optionally ViT / MobileNet) on the SAME dataset and SAME train/val/test split as
your dual-branch model, using the SAME evaluation. This answers the supervisor's
request: "compare against an existing model on the same data."

These baselines use only the image (no bark mask, no texture branch, no fusion)
— they are the plain, published architectures. Your dual-branch ensemble is the
proposed method; this shows where it stands relative to standard models.

Results are appended to outputs/cls/baseline_compare.csv and a bar chart is
written to outputs/figures/fig_baseline_compare.png.

Run:  python scripts/27_baseline_compare.py --model resnet50 --seed 42 --workers 0
      python scripts/27_baseline_compare.py --model efficientnet_b0 --seed 42 --workers 0
      python scripts/27_baseline_compare.py --all --seed 42 --workers 0
"""
from __future__ import annotations

import argparse
import csv
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

# standard, recognised baselines (timm model names)
BASELINES = ["resnet50", "efficientnet_b0", "mobilenetv3_large_100", "vit_small_patch16_224"]


def macro_f1(yt, yp, k=3):
    f1s = []
    for c in range(k):
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    for img, _mask, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            logits = model(img.to(device))
        yp.append(logits.argmax(1).cpu().numpy())
        yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    acc = float((yt == yp).mean())
    cm = np.zeros((3, 3), int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return acc, macro_f1(yt, yp), cm


def train_baseline(model_name, args, cfg, device):
    import timm
    print(f"\n{'='*56}\nbaseline: {model_name}\n{'='*56}")

    # plain classifiers use only the image; no mask applied
    train_ds = BarkClsDataset("train", size=args.size, train=True,
                              mask_source=args.mask_source, apply_mask=False)
    val_ds = BarkClsDataset("valid", size=args.size, train=False,
                            mask_source=args.mask_source, apply_mask=False)
    test_ds = BarkClsDataset("test", size=args.size, train=False,
                             mask_source=args.mask_source, apply_mask=False)

    ys = train_ds.df.class_idx.values
    counts = np.bincount(ys, minlength=3).astype(np.float32)
    cls_w = torch.tensor(counts.sum() / (len(counts) * counts)).to(device)

    def mk(ds, sh):
        return DataLoader(ds, batch_size=args.batch, shuffle=sh,
                          num_workers=args.workers, pin_memory=True, drop_last=sh)
    train_dl, val_dl, test_dl = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    model = timm.create_model(model_name, pretrained=True, num_classes=3).to(device)
    crit = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1, bad, best_state = -1.0, 0, None
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        for img, _mask, y in train_dl:
            img, y = img.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                loss = crit(model(img), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        sched.step()
        vacc, vf1, _ = evaluate(model, val_dl, device)
        print(f"  epoch {ep+1:2d}/{args.epochs}  val acc {vacc:.4f}  "
              f"val f1 {vf1:.4f}  ({time.time()-t0:.0f}s)")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    tacc, tf1, cm = evaluate(model, test_dl, device)
    rb_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    print(f"  TEST  acc {tacc:.4f}  macro-f1 {tf1:.4f}  "
          f"rough-bark recall {rb_recall:.4f}")
    print(f"  confusion matrix [rb, healthy, sc]:\n{cm}")

    return {"model": model_name, "seed": args.seed, "test_acc": round(tacc, 4),
            "test_f1": round(tf1, 4), "rough_bark_recall": round(float(rb_recall), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=BASELINES)
    ap.add_argument("--all", action="store_true", help="train all baselines")
    ap.add_argument("--mask_source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = cfg.path("paths", "processed_root").parent.parent / "outputs" / "cls"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "baseline_compare.csv"

    models = BASELINES if args.all else [args.model]
    results = []
    for mname in models:
        b = args.batch
        while True:
            try:
                args.batch = b
                results.append(train_baseline(mname, args, cfg, device))
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if b <= 2:
                    print(f"  OOM at batch 2 on {mname}; skipping")
                    break
                b //= 2
                print(f"  OOM -> retry {mname} at batch {b}")

    # append to CSV (key by model+seed)
    existing = {}
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                existing[(r["model"], r.get("seed"))] = r
    for r in results:
        existing[(r["model"], str(r["seed"]))] = r
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "seed", "test_acc",
                                           "test_f1", "rough_bark_recall"])
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])

    print(f"\n{'='*56}\nBASELINE COMPARISON -> {csv_path}\n{'='*56}")
    with open(csv_path) as fh:
        print(fh.read())
    print("Add your dual-branch ensemble result (acc 0.883) as the proposed "
          "method when you tabulate this in the thesis.")

    # bar chart: baselines + your method
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.read_csv(csv_path)
        g = df.groupby("model").test_acc.mean().sort_values()
        names = list(g.index) + ["dual-branch\nensemble (ours)"]
        accs = list(g.values) + [0.883]
        colors = ["#7fcdbb"] * len(g) + ["#2c7fb8"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(names)), accs, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylim(0.6, 1.0); ax.set_ylabel("test accuracy")
        ax.set_title("Classification: existing baselines vs proposed")
        for i, a in enumerate(accs):
            ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        fp = out.parent / "figures" / "fig_baseline_compare.png"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fp, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"chart -> {fp}")
    except Exception as e:
        print(f"(chart skipped: {e})")


if __name__ == "__main__":
    main()
