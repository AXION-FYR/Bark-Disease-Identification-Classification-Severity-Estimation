"""
Step 22 — ensemble classifier for stable rough-bark accuracy.

Single classifiers on this data are high-variance: rough-bark recall bounced
between 57% and 76% across seeds of the SAME focal configuration. That is a
reliability problem, not a tuning one. An ensemble fixes it: train N models on
different seeds, average their softmax probabilities at test time. The averaged
prediction cancels per-model randomness, so rough-bark recall stabilises near
the top of the range and reproduces on every run.

Trains N focal models (saved individually so they can be reused), then reports:
  * each member's rough-bark recall (to show the spread)
  * the ENSEMBLE confusion matrix, accuracy, macro-F1, and rough-bark recall

Run:  python scripts/22_ensemble_cls.py --variant mcse_allones --focal --n 5 --workers 0
      python scripts/22_ensemble_cls.py --variant mcse_allones --focal --n 5 --reuse
          (--reuse loads existing member checkpoints instead of retraining)
"""
from __future__ import annotations

import argparse
import json
import sys
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
    """Focal loss — focuses training on hard, misclassified cases (rough bark)."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing,
                                      reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


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


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_member(variant, seed, args, cfg, device):
    set_seed(seed)
    ms = args.mask_source
    train_ds = BarkClsDataset("train", size=args.size, train=True, mask_source=ms)
    val_ds = BarkClsDataset("valid", size=args.size, train=False, mask_source=ms)

    ys = train_ds.df.class_idx.values
    counts = np.bincount(ys, minlength=3).astype(np.float32)
    cls_w = torch.tensor(counts.sum() / (len(counts) * counts)).to(device)

    def mk(ds, sh):
        return DataLoader(ds, batch_size=args.batch, shuffle=sh,
                          num_workers=args.workers, pin_memory=True, drop_last=sh)
    train_dl, val_dl = mk(train_ds, True), mk(val_ds, False)

    model = build_classifier(variant=variant, num_classes=3).to(device)
    crit = (FocalLoss(alpha=cls_w, gamma=args.gamma, label_smoothing=0.05)
            if args.focal else nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1, bad, best_state = -1.0, 0, None
    for ep in range(args.epochs):
        model.train()
        for step, (img, mask, y) in enumerate(train_dl):
            img, mask, y = img.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                loss = crit(model(img, mask), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        sched.step()
        # quick val f1 for early stopping
        model.eval(); yt, yp = [], []
        with torch.no_grad():
            for img, mask, y in val_dl:
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    lg = model(img.to(device), mask.to(device))
                yp.append(lg.argmax(1).cpu().numpy()); yt.append(y.numpy())
        vf1 = macro_f1(np.concatenate(yt), np.concatenate(yp), 3)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def member_probs(model, loader, device):
    """Return softmax probs (N,3) and labels (N,) for the test set."""
    model.eval(); P, Y = [], []
    for img, mask, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            lg = model(img.to(device), mask.to(device))
        P.append(torch.softmax(lg.float(), 1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def report(cm, tag):
    acc = np.trace(cm) / cm.sum()
    rb_recall = cm[0, 0] / cm[0].sum()
    print(f"  {tag}: acc {acc:.4f}   rough-bark recall {rb_recall:.4f}")
    return acc, rb_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mcse_allones")
    ap.add_argument("--n", type=int, default=5, help="ensemble members")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--mask_source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--reuse", action="store_true",
                    help="load saved member checkpoints instead of retraining")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = cfg.path("paths", "processed_root").parent.parent / "outputs" / "cls"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.variant + ("_focal" if args.focal else "")

    test_ds = BarkClsDataset("test", size=args.size, train=False,
                             mask_source=args.mask_source)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers)

    seeds = list(range(args.n))
    all_probs, labels = [], None
    print(f"ensemble: {args.n} members, variant {tag}\n")

    print("member rough-bark recall (the spread the ensemble fixes):")
    for i, s in enumerate(seeds):
        ckpt = out / f"ens_{tag}_m{i}.pt"
        if args.reuse and ckpt.exists():
            st = torch.load(ckpt, map_location=device, weights_only=False)
            model = build_classifier(variant=args.variant, num_classes=3).to(device)
            model.load_state_dict(st["model"])
        else:
            model = train_member(args.variant, s, args, cfg, device)
            torch.save({"model": model.state_dict(), "seed": s}, ckpt)

        P, Y = member_probs(model, test_dl, device)
        labels = Y
        all_probs.append(P)
        pred = P.argmax(1)
        cm = np.zeros((3, 3), int)
        for t, p in zip(Y, pred):
            cm[t, p] += 1
        report(cm, f"member {i} (seed {s})")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- ensemble: average probabilities ---------------------------------
    mean_p = np.mean(all_probs, axis=0)
    ens_pred = mean_p.argmax(1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(labels, ens_pred):
        cm[t, p] += 1

    print("\n" + "=" * 56)
    print("ENSEMBLE (averaged softmax)")
    print("=" * 56)
    acc = float((ens_pred == labels).mean())
    f1 = macro_f1(labels, ens_pred, 3)
    print(f"accuracy {acc:.4f}   macro-f1 {f1:.4f}")
    print("confusion matrix (rows=true [rb, healthy, sc]):")
    print(cm)
    rb = cm[0, 0] / cm[0].sum()
    print(f"rough-bark recall: {rb:.4f}")

    json.dump({"n": args.n, "variant": tag, "ensemble_acc": acc,
               "ensemble_f1": f1, "rough_bark_recall": float(rb),
               "cm": cm.tolist()}, open(out / f"ensemble_{tag}.json", "w"),
              indent=2)
    # also save a confusion matrix that 21_figures.py can render
    json.dump([{"variant": f"{tag}_ensemble", "test_acc": acc,
                "test_f1": f1, "cm": cm.tolist()}],
              open(out / "ablation_full.json", "w"), indent=2)
    print(f"\nsaved {out / f'ensemble_{tag}.json'}")
    print("confusion matrix written to ablation_full.json — run 21_figures.py "
          "to render the ensemble confusion figure.")


if __name__ == "__main__":
    main()
