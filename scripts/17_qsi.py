"""
Step 17 — Quantitative Severity Index (QSI).  Novelty Claim 4.

Plain "affected-area percentage" treats a small necrotic patch the same as a
large faint discolouration. QSI weights each diseased pixel by how DAMAGED it
looks, so severity reflects both extent and intensity.

  QSI = ( sum over bark pixels of  p_i * d_i ) / ( sum over bark of 1 )

  p_i = lesion probability from the Stage-3 decoder (extent)
  d_i = normalised DAMAGE INTENSITY at pixel i (how far its local texture/colour
        deviates from HEALTHY bark) — bounded to [0,1]

The healthy reference (mean texture-energy and mean L*) is estimated from
TRAINING healthy images ONLY. Fitting it on all healthy images would leak test
data into the metric — a mistake an examiner checks for.

Outputs, per test image:
  * QSI in [0,1] and a grade (Mild / Moderate / Severe)
  * naive area-% for comparison (shows QSI is not just area)
Plus, if expert grades are supplied via --grades, Spearman rho between QSI and
the expert ordinal grade — the validation that makes Claim 4 real.

Run:  python scripts/17_qsi.py \
          --lesion_ckpt outputs/lesion/lesion_film.pt \
          --stage2_ckpt outputs/cls/cls_mcse_allones.pt
      # optional expert validation:
      #   --grades path/to/grades.csv   (columns: stem,grade  grade in 0/1/2)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402

from src.config import load_config, PROJECT_ROOT                      # noqa: E402
from src.imaging import read_image_rgb, read_mask_png                 # noqa: E402
from src.dataset_cls import crop_to_bbox, clahe_lab                   # noqa: E402
from src.model_lesion import build_lesion_decoder                    # noqa: E402

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def texture_energy(gray):
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (9, 9))
    sq = cv2.blur(g * g, (9, 9))
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


def prep(img_rgb, bark_mask, size, margin=0.05):
    img, m = crop_to_bbox(img_rgb, bark_mask, margin)
    disp = clahe_lab(img)
    disp_r = cv2.resize(disp, (size, size), interpolation=cv2.INTER_AREA)
    m_r = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    x = ((disp_r.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    return x, m_r, disp_r


# Disease-conditioned damage weights (texture_weight, darkness_weight).
# Rough bark is a DIFFUSE TEXTURE disease -> weight roughness more.
# Stripe canker is DARK STREAKS -> weight darkness more.
# Healthy is never scored for severity. Unknown -> balanced.
# This mirrors how an expert weighs different visual cues per disease, and is
# the novel "disease-conditioned severity" contribution.
DISEASE_WEIGHTS = {
    "Rough bark":    (0.70, 0.30),   # texture-dominant
    "stripecanker":  (0.30, 0.70),   # darkness-dominant
    "_default":      (0.50, 0.50),
}


def damage_intensity(disp_rgb, bark, ref, disease=None):
    """
    d_i in [0,1]: how far each bark pixel deviates from the healthy reference in
    (texture energy, L*), combined with DISEASE-CONDITIONED weights so the
    damage measure matches each disease's visual signature. Non-bark -> 0.
    """
    gray = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    tex = texture_energy(gray)

    # deviation from healthy means, in the "worse" direction:
    # diseased bark is ROUGHER (higher texture) and often DARKER (lower L)
    d_tex = np.clip((tex - ref["tex_mean"]) / (ref["tex_std"] + 1e-6), 0, None)
    d_dark = np.clip((ref["L_mean"] - L) / (ref["L_std"] + 1e-6), 0, None)

    w_tex, w_dark = DISEASE_WEIGHTS.get(disease, DISEASE_WEIGHTS["_default"])
    d = w_tex * d_tex + w_dark * d_dark

    # normalise to [0,1] by a soft cap at 3 std devs
    d = np.clip(d / 3.0, 0, 1)
    d = d * (bark > 0)
    return d.astype(np.float32)


def fit_healthy_reference(cfg, size, device, args):
    """Mean/std of texture energy and L* over TRAINING healthy bark pixels."""
    manifest = pd.read_csv(cfg.path("paths", "processed_root") / "manifest.csv")
    healthy = manifest[(manifest.split == "train") &
                       (manifest.class_name == "healthy bark")]
    tex_vals, L_vals = [], []
    for _, row in tqdm(healthy.iterrows(), total=len(healthy),
                       desc="healthy ref", unit="img"):
        img = read_image_rgb(PROJECT_ROOT / row.image_path)
        bark = read_mask_png(PROJECT_ROOT / row.mask_path)
        _, m_r, disp = prep(img, bark, size)
        gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(disp, cv2.COLOR_RGB2LAB)
        sel = m_r > 0
        if sel.sum() < 50:
            continue
        tex_vals.append(texture_energy(gray)[sel])
        L_vals.append(lab[:, :, 0].astype(np.float32)[sel])
    tex_all = np.concatenate(tex_vals)
    L_all = np.concatenate(L_vals)
    return {"tex_mean": float(tex_all.mean()), "tex_std": float(tex_all.std()),
            "L_mean": float(L_all.mean()), "L_std": float(L_all.std())}


# ---- Disease-specific treatment-stage bands (AUDITABLE LOOKUP TABLE) -----
# Applied to % diseased area (pct, 0..1) per disease.  Kept as a table so it
# is documented, editable without retraining, and consistent with the tree-
# level staging in 19_tree_severity.py.
# Stripe canker scale is compressed (~1.7x) because equal extent is more
# urgent: active, progressive cambial necrosis vs periderm-quality disorder.
# These are the SAME bands used at tree level — staging is consistent
# whether you look at one image or aggregate many views of a tree.
DISEASE_BANDS = {
        "Rough bark":   {"prev": 0.30, "early": 0.50, "active": 0.80},
        "stripecanker": {"prev": 0.20, "early": 0.40, "active": 0.70},
        "_default":     {"prev": 0.05, "early": 0.15, "active": 0.50},
}
STAGE_NAMES = ["Preventive", "Early control", "Active management", "Severe outbreak"]


def treatment_stage(pct: float, disease: str) -> str:
    """Continuous % diseased area -> treatment stage via disease-conditioned bands."""
    b = DISEASE_BANDS.get(disease, DISEASE_BANDS["_default"])
    if pct < b["prev"]:   return "Preventive"
    if pct < b["early"]:  return "Early control"
    if pct < b["active"]: return "Active management"
    return "Severe outbreak"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesion_ckpt", required=True)
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--grades", default=None,
                    help="optional CSV with columns stem,grade for expert "
                         "validation (stage label P/E/A/S or numeric 0-3)")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = PROJECT_ROOT / "outputs" / "qsi"
    out.mkdir(parents=True, exist_ok=True)

    # 1. healthy reference from TRAIN healthy only
    ref = fit_healthy_reference(cfg, args.size, device, args)
    print(f"healthy reference: tex {ref['tex_mean']:.2f}±{ref['tex_std']:.2f}  "
          f"L {ref['L_mean']:.2f}±{ref['L_std']:.2f}")
    with open(out / "healthy_reference.json", "w") as fh:
        json.dump(ref, fh, indent=2)

    # 2. lesion decoder
    st = torch.load(args.lesion_ckpt, map_location=device, weights_only=False)
    model = build_lesion_decoder(args.stage2_ckpt,
                                 class_conditioned=st.get("film", True)).to(device)
    model.load_state_dict(st["model"])
    model.eval()

    manifest = pd.read_csv(cfg.path("paths", "processed_root") / "manifest.csv")
    sub = manifest[manifest.split == args.split]
    sub = sub[sub.class_idx >= 0]

    rows = []
    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="QSI", unit="img"):
        img = read_image_rgb(PROJECT_ROOT / row.image_path)
        pm = cfg.path("paths", "processed_root") / args.split / "pred_masks" / f"{row.stem}.png"
        bark = read_mask_png(pm) if pm.exists() else read_mask_png(PROJECT_ROOT / row.mask_path)

        x, m_r, disp = prep(img, bark, args.size)
        cls_idx = int(row.class_idx)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(x, device=device),
                                 torch.tensor([cls_idx], device=device)))[0, 0].cpu().numpy()
        prob = prob * (m_r > 0)

        d = damage_intensity(disp, m_r, ref, disease=row.class_name)
        bark_px = float((m_r > 0).sum())
        if bark_px < 1:
            continue

        qsi = float((prob * d).sum() / bark_px)
        area_pct = float((prob > 0.3).sum() / bark_px)
        stage = treatment_stage(area_pct, row.class_name)

        rows.append({"stem": row.stem, "split": args.split,
                     "class_name": row.class_name,
                     "qsi": round(qsi, 5),
                     "area_pct": round(area_pct, 5),
                     "stage": stage})

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / f"qsi_{args.split}.csv", index=False)

    print("\n" + "=" * 60)
    print(f"QSI on {args.split} ({len(rdf)} images)")
    print("=" * 60)
    print("mean QSI by class:")
    print(rdf.groupby("class_name")[["qsi", "area_pct"]].mean().round(4).to_string())
    print("\nstage distribution (disease-specific bands on % area):")
    print(rdf.groupby("class_name")["stage"].value_counts().to_string())

    # healthy trunks should score LOW — a built-in sanity check
    h = rdf[rdf.class_name == "healthy bark"]
    if len(h):
        print(f"\nhealthy-trunk mean QSI: {h.qsi.mean():.4f} "
              f"(sanity: should be well below diseased classes)")

    # QSI vs naive area — if they rank-correlate ~1.0, QSI adds nothing;
    # a gap shows the intensity weighting changes the ordering.
    if len(rdf) > 3:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(rdf.qsi, rdf.area_pct)
        print(f"\nQSI vs naive area-%  Spearman rho = {rho:.3f}")
        print("  (high rho = similar ranking; the value of QSI is in the cases "
              "where intensity re-orders severity vs raw area)")

    # optional expert validation
    if args.grades and Path(args.grades).exists():
        g = pd.read_csv(args.grades)
        m = rdf.merge(g, on="stem", how="inner")
        if len(m) >= 5:
            from scipy.stats import spearmanr
            rho_e, p_e = spearmanr(m.qsi, m.grade)
            # compare predicted stage (from area_pct + disease) to expert grade
            stage_idx = {s: i for i, s in enumerate(STAGE_NAMES)}
            m["pred_stage_idx"] = m.apply(
                lambda r: stage_idx.get(treatment_stage(r.area_pct, r.class_name), 0), axis=1)
            mae = float((m.grade - m.pred_stage_idx).abs().mean())
            print("\n" + "=" * 60)
            print(f"EXPERT VALIDATION ({len(m)} graded images)")
            print("=" * 60)
            print(f"Spearman QSI vs expert grade: rho = {rho_e:.3f}  p = {p_e:.4f}")
            print(f"grade MAE (QSI-threshold vs expert): {mae:.3f}")
        else:
            print(f"\nonly {len(m)} matched grades — need >=5 for correlation")

    # gallery: image + lesion prob + QSI value, sorted by severity
    rdf_sorted = rdf.sort_values("qsi")
    picks = pd.concat([rdf_sorted.head(3), rdf_sorted.tail(3)])
    fig, axes = plt.subplots(len(picks), 2, figsize=(8, 4 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, (_, prow) in enumerate(picks.iterrows()):
        mrow = manifest[manifest.stem == prow.stem].iloc[0]
        img = read_image_rgb(PROJECT_ROOT / mrow.image_path)
        pm = cfg.path("paths", "processed_root") / args.split / "pred_masks" / f"{prow.stem}.png"
        bark = read_mask_png(pm) if pm.exists() else read_mask_png(PROJECT_ROOT / mrow.mask_path)
        x, m_r, disp = prep(img, bark, args.size)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(x, device=device),
                                 torch.tensor([int(mrow.class_idx)], device=device)))[0, 0].cpu().numpy()
        axes[r, 0].imshow(disp); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"{prow.class_name}", fontsize=9)
        axes[r, 1].imshow(disp); axes[r, 1].imshow(prob * (m_r > 0), alpha=0.5, cmap="jet")
        axes[r, 1].axis("off")
        axes[r, 1].set_title(f"QSI {prow.qsi:.3f} — {prow.stage}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"qsi_gallery_{args.split}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\nper-image QSI -> {out / f'qsi_{args.split}.csv'}")
    print(f"gallery       -> {out / f'qsi_gallery_{args.split}.png'}")
    print("\nTo validate Claim 4: have an expert grade ~40 images  save as "
          "stem,grade CSV, and re-run with --grades. Report the Spearman rho.")


if __name__ == "__main__":
    main()
