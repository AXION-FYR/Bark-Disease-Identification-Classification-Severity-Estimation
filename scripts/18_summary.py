"""
Step 18 — consolidated results summary.

Reads every metrics file the pipeline produced and prints one table with every
headline number, ready to drop into the thesis. Also writes a machine-readable
summary.json and a results.md you can paste into the write-up.

Missing files are reported as "not run" rather than crashing, so you can run
this at any point and see what's still outstanding.

Run:  python scripts/18_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, PROJECT_ROOT                      # noqa: E402


def load_json(p: Path):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def load_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def main() -> None:
    cfg = load_config()
    proc = cfg.path("paths", "processed_root")
    out = PROJECT_ROOT / "outputs"
    lines = []      # markdown lines for results.md
    summary = {}

    def show(title):
        print("\n" + "=" * 66)
        print(title)
        print("=" * 66)
        lines.append(f"\n## {title}\n")

    # ---------------------------------------------------------------
    # dataset
    # ---------------------------------------------------------------
    show("DATASET")
    man = load_csv(proc / "manifest.csv")
    if man is not None:
        man = man[man.class_name != "__none__"]
        by = man.groupby(["split", "class_name"]).size().unstack(fill_value=0)
        print(by.to_string())
        print(f"total usable: {len(man)}")
        lines.append(f"Total images: {len(man)}\n")
        lines.append("```\n" + by.to_string() + "\n```\n")
        summary["dataset"] = {"total": int(len(man)),
                              "by_split_class": by.to_dict()}
    else:
        print("manifest.csv not found — run 02_build_masks.py")

    # ---------------------------------------------------------------
    # Stage 1 — segmentation
    # ---------------------------------------------------------------
    show("STAGE 1 — Bark segmentation (U-Net + EfficientNet-B0)")
    seg = load_json(out / "seg" / "test_metrics.json")
    if seg and "test" in seg:
        t = seg["test"]
        print(f"test IoU  {t['iou_mean']:.4f}    Dice {t['dice_mean']:.4f}")
        lines.append(f"Test IoU **{t['iou_mean']:.4f}**, Dice {t['dice_mean']:.4f}\n")
        if "per_class" in t:
            print("per class:")
            for c, v in t["per_class"].items():
                print(f"  {c:14s} IoU {v['iou']:.4f}  Dice {v['dice']:.4f}")
        summary["stage1_seg"] = {"iou": t["iou_mean"], "dice": t["dice_mean"],
                                 "per_class": t.get("per_class")}
    else:
        print("seg/test_metrics.json not found — run 10_eval_seg.py")

    # ---------------------------------------------------------------
    # Stage 2 — classification
    # ---------------------------------------------------------------
    show("STAGE 2 — Disease classification (ablation, mean ± std over seeds)")
    absum = load_csv(out / "cls" / "ablation.csv")
    if absum is not None:
        for col in ["test_acc", "test_f1"]:
            absum[col] = pd.to_numeric(absum[col], errors="coerce")
        g = absum.groupby("variant").agg(
            seeds=("test_acc", "size"),
            acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
            f1_mean=("test_f1", "mean"), f1_std=("test_f1", "std")).fillna(0)
        order = ["plain", "masked", "concat", "se", "mcse", "mcse_allones"]
        g = g.reindex([v for v in order if v in g.index])
        pretty = {"mcse_allones": "dualbranch_se (best)", "mcse": "mask-cond SE",
                  "se": "unmasked SE", "concat": "appear+texture",
                  "masked": "appear (masked)", "plain": "appear (baseline)"}
        for v, r in g.iterrows():
            print(f"  {pretty.get(v, v):22s} acc {r.acc_mean:.3f}±{r.acc_std:.3f}"
                  f"   f1 {r.f1_mean:.3f}±{r.f1_std:.3f}  ({int(r.seeds)} seeds)")
        summary["stage2_cls"] = g.round(4).to_dict("index")
    else:
        print("cls/ablation.csv not found — run 11_train_cls.py")

    # ---------------------------------------------------------------
    # Stage 3 — lesion localisation
    # ---------------------------------------------------------------
    show("STAGE 3 — Lesion localisation (weakly supervised vs Grad-CAM)")
    les = load_json(out / "lesion" / "lesion_metrics.json")
    if les:
        print(f"ours   IoU {les['ours_iou']:.4f}  Dice {les['ours_dice']:.4f}")
        print(f"GradCAM IoU {les['cam_iou']:.4f}  Dice {les['cam_dice']:.4f}")
        print(f"margin (ours - cam): {les['ours_iou'] - les['cam_iou']:+.4f}")
        if les.get("healthy_anchor_mean") is not None:
            print(f"healthy-anchor mean lesion prob: "
                  f"{les['healthy_anchor_mean']:.4f}  (near 0 = good)")
        if "per_class" in les:
            print("per class (ours vs cam IoU):")
            for c, v in les["per_class"].items():
                print(f"  {c:14s} ours {v['ours_iou']:.4f}  cam {v['cam_iou']:.4f}")
        summary["stage3_lesion"] = les
    else:
        print("lesion/lesion_metrics.json not found — run 15_eval_lesion.py")

    # FiLM ablation, if a nofilm eval was saved separately
    nofilm = load_json(out / "lesion" / "lesion_metrics_nofilm.json")
    if nofilm:
        print(f"\nFiLM ablation:  film IoU {les['ours_iou']:.4f}  "
              f"vs no-film IoU {nofilm['ours_iou']:.4f}  "
              f"({les['ours_iou'] - nofilm['ours_iou']:+.4f})")
        summary["stage3_film_ablation"] = {
            "film_iou": les["ours_iou"], "nofilm_iou": nofilm["ours_iou"]}

    # refiner
    ref = load_json(out / "lesion" / "refine_metrics.json")
    if ref:
        print(f"\npatch refiner (rough bark region vs refined IoU):")
        pc = ref.get("per_class", {})
        for c, v in pc.items():
            print(f"  {c:14s} region {v['region_iou']:.4f}  "
                  f"refined {v['refined_iou']:.4f}")
        summary["stage3_refiner"] = ref

    # ---------------------------------------------------------------
    # Stage 4 — QSI
    # ---------------------------------------------------------------
    show("STAGE 4 — QSI severity")
    qsi = load_csv(out / "qsi" / "qsi_test.csv")
    if qsi is not None:
        mean_by = qsi.groupby("class_name")[["qsi", "area_pct"]].mean()
        print("mean QSI vs naive area by class:")
        print(mean_by.round(4).to_string())
        h = qsi[qsi.class_name == "healthy bark"].qsi.mean()
        d = qsi[qsi.class_name != "healthy bark"].qsi.mean()
        print(f"\nhealthy {h:.4f}  vs  diseased {d:.4f}  "
              f"(separation {d / (h + 1e-9):.0f}x)")
        from scipy.stats import spearmanr
        rho, _ = spearmanr(qsi.qsi, qsi.area_pct)
        print(f"QSI vs area Spearman rho: {rho:.3f}")
        if "stage" in qsi.columns:
            print("stages: " + qsi["stage"].value_counts().to_string())
        elif "grade" in qsi.columns:
            gr = qsi["grade"].value_counts().sort_index()
            print("grades: " + ", ".join(f"{k} {v}" for k, v in gr.items()))
        summary["stage4_qsi"] = {
            "healthy_mean": float(h), "diseased_mean": float(d),
            "rho_vs_area": float(rho),
            "by_class": mean_by.round(4).to_dict("index")}
    else:
        print("qsi/qsi_test.csv not found — run 17_qsi.py")

    # ---------------------------------------------------------------
    # write outputs
    # ---------------------------------------------------------------
    (out).mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(out / "results.md", "w") as fh:
        fh.write("# Cinnamon bark disease pipeline — results\n")
        fh.write("\n".join(lines))

    print("\n" + "=" * 66)
    print(f"written: {out / 'summary.json'}")
    print(f"written: {out / 'results.md'}  (paste into your thesis)")
    print("=" * 66)


if __name__ == "__main__":
    main()
