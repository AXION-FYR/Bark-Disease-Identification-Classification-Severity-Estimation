"""
Step 19 — multi-view tree severity aggregation.

Runs the FULL pipeline on each raw tree photo, fresh (these images were never
seen by any model), then aggregates to one severity per disease per tree.

per photo:
  1. Stage 1  U-Net -> bark mask
  2. Stage 2  classifier -> disease + confidence
  3. Stage 3  lesion decoder -> lesion map
  4. Stage 4  -> % diseased bark  and  QSI (extent x damage intensity)
  5. image-quality weight (bark area x confidence x sharpness)

per tree:
  group the ~15 photos by predicted disease, take a WEIGHTED average of the
  per-photo severity within each disease. A tree with two diseases yields two
  rows. Views weighted by (bark area x classifier confidence x sharpness) so
  blurry / edge-on / low-confidence views count less -- this is what makes
  the tree score more than a plain mean of 15 runs.

STAGE POLICY:
  The stage sent downstream ("action_stage") is derived from % diseased area
  ALONE, via the disease-specific band table. Circumferential spread and the
  girdling flag are still computed and reported for every tree/disease, but
  they do NOT change action_stage unless --escalate_by_spread is explicitly
  passed. This is deliberate: Module 03 (the decision engine) computes its
  own stage from severity_percentage on its side. If this script also
  escalated the stage based on spread, two different components of the
  system could disagree about the SAME tree's stage for the same underlying
  percentage. Exactly one place (the decision engine) owns "percentage ->
  stage" by default. Spread/girdling remain in the CSV as decision-support
  context, not baked silently into the stage label.

Output: outputs/tree/tree_severity.csv  and a readable printout:

  tree  disease        pct_bark  qsi     n_views  weight_frac
  t01   StripeCanker   0.34      0.31    9        0.71
  t01   RoughBark      0.12      0.08    4        0.29

Run:  python scripts/19_tree_severity.py \
          --tree_root D:/RESEARCH/Dataset/multiview_trees \
          --seg_ckpt outputs/seg/best.pt \
          --stage2_ckpt outputs/cls/cls_mcse_allones.pt \
          --lesion_ckpt outputs/lesion/lesion_film.pt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                          # noqa: E402

from src.config import load_config, PROJECT_ROOT                      # noqa: E402
from src.imaging import letterbox_image, letterbox_mask, read_image_rgb  # noqa: E402
from src.dataset_cls import crop_to_bbox, clahe_lab                   # noqa: E402
from src.model_cls import build_classifier                           # noqa: E402
from src.model_lesion import build_lesion_decoder                    # noqa: E402
from src.dataset_cls import IMAGENET_MEAN, IMAGENET_STD               # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --- reuse QSI helpers without importing the whole 17 script's main() ---------
def _texture_energy(gray):
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (9, 9))
    sq = cv2.blur(g * g, (9, 9))
    return np.sqrt(np.clip(sq - mean * mean, 0, None))


DISEASE_WEIGHTS = {
    "Rough bark":   (0.70, 0.30),   # texture-dominant
    "stripecanker": (0.30, 0.70),   # darkness-dominant
    "_default":     (0.50, 0.50),
}


def _damage_intensity(disp_rgb, bark, ref, disease=None):
    gray = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    tex = _texture_energy(gray)
    d_tex = np.clip((tex - ref["tex_mean"]) / (ref["tex_std"] + 1e-6), 0, None)
    d_dark = np.clip((ref["L_mean"] - L) / (ref["L_std"] + 1e-6), 0, None)
    w_tex, w_dark = DISEASE_WEIGHTS.get(disease, DISEASE_WEIGHTS["_default"])
    d = np.clip((w_tex * d_tex + w_dark * d_dark) / 3.0, 0, 1)
    return (d * (bark > 0)).astype(np.float32)


def parse_tree_photo(fname: str):
    """'t01_p03.JPG' -> (1, 3). Returns None if it doesn't match."""
    m = re.match(r"^t(\d+)_p(\d+)", fname, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else None


def sharpness(gray):
    """Variance of Laplacian -- low on blurred / motion-blurred images."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree_root", required=True)
    ap.add_argument("--seg_ckpt", default="outputs/seg/best.pt")
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--lesion_ckpt", required=True)
    ap.add_argument("--seg_size", type=int, default=512)
    ap.add_argument("--cls_size", type=int, default=224)
    ap.add_argument("--healthy_idx", type=int, default=1)
    ap.add_argument("--min_views", type=int, default=2,
                    help="a disease is reported for a tree only if >= this many "
                         "views predict it (filters single-photo noise)")
    ap.add_argument("--girdle_spread", type=float, default=0.6,
                    help="spread (fraction of views showing disease) at/above "
                         "which the tree is flagged as girdling risk in the "
                         "report (informational -- see --escalate_by_spread)")
    ap.add_argument("--escalate_by_spread", action="store_true",
                    help="if set, bump action_stage by one level when "
                         "girdling risk is flagged. OFF by default so this "
                         "script's stage cannot disagree with a downstream "
                         "stage computed independently (e.g. by the decision "
                         "engine / Module 03) from the same severity_percentage.")
    ap.add_argument("--lesion_thr", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.tree_root)
    out = PROJECT_ROOT / "outputs" / "tree"
    out.mkdir(parents=True, exist_ok=True)

    classes = json.load(open(cfg.path("paths", "processed_root") / "classes.json"))
    idx_to_name = {v: k for k, v in classes["class_to_idx"].items()}

    # ---- load the three models -------------------------------------------
    import segmentation_models_pytorch as smp
    seg_st = torch.load(PROJECT_ROOT / args.seg_ckpt, map_location=device,
                        weights_only=False)
    seg = smp.Unet(encoder_name=seg_st.get("encoder", "efficientnet-b0"),
                   encoder_weights=None, in_channels=3, classes=1).to(device)
    seg.load_state_dict(seg_st["model"])
    seg.eval()

    s2 = torch.load(PROJECT_ROOT / args.stage2_ckpt, map_location=device,
                    weights_only=False)
    clf = build_classifier(variant=s2.get("variant", "mcse_allones"),
                           pretrained=False).to(device)
    clf.load_state_dict(s2["model"])
    clf.eval()

    les_st = torch.load(PROJECT_ROOT / args.lesion_ckpt, map_location=device,
                        weights_only=False)
    lesion = build_lesion_decoder(str(PROJECT_ROOT / args.stage2_ckpt),
                                  class_conditioned=les_st.get("film", True)).to(device)
    lesion.load_state_dict(les_st["model"])
    lesion.eval()

    # healthy reference for damage intensity (from QSI step, if present)
    ref_path = PROJECT_ROOT / "outputs" / "qsi" / "healthy_reference.json"
    if ref_path.exists():
        ref = json.load(open(ref_path))
    else:
        sys.exit("healthy_reference.json not found -- run scripts/17_qsi.py once "
                 "first so the damage-intensity reference exists.")

    # ---- collect tree photos ---------------------------------------------
    photos = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                tp = parse_tree_photo(f.name)
                if tp:
                    photos.append((tp[0], tp[1], f))
    if not photos:
        sys.exit(f"no t<N>_p<N> images under {root} -- run 19a_rename_trees.py")

    print(f"processing {len(photos)} photos across "
          f"{len({t for t, _, _ in photos})} trees")

    per_photo = []
    for tnum, pnum, path in tqdm(photos, unit="img"):
        img = read_image_rgb(path)

        # Stage 1: bark mask (letterboxed 512, then map back)
        seg_in, params = letterbox_image(img, args.seg_size)
        x = ((seg_in.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
        x = torch.tensor(np.ascontiguousarray(x.transpose(2, 0, 1))[None],
                         device=device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                bark_lb = (torch.sigmoid(seg(x))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        # un-letterbox back to original image size
        pl = params
        inner = bark_lb[pl.pad_top:args.seg_size - pl.pad_bottom,
                        pl.pad_left:args.seg_size - pl.pad_right]
        bark = cv2.resize(inner, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        if bark.sum() < 100:
            continue      # no bark found, skip

        # Stage 2 input: crop to bark bbox, clahe, mask, resize
        crop, m = crop_to_bbox(img, bark, 0.05)
        disp = clahe_lab(crop)
        disp_r = cv2.resize(disp, (args.cls_size, args.cls_size), interpolation=cv2.INTER_AREA)
        m_r = cv2.resize(m, (args.cls_size, args.cls_size), interpolation=cv2.INTER_NEAREST)
        cx = ((disp_r.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
        cx = torch.tensor(np.ascontiguousarray(cx.transpose(2, 0, 1))[None], device=device)
        mm = torch.tensor(m_r.astype(np.float32)[None, None], device=device)

        with torch.no_grad():
            logits = clf(cx * mm, mm)   # masked input, like training
            probs = torch.softmax(logits, 1)[0].cpu().numpy()
        cls_idx = int(probs.argmax())
        conf = float(probs[cls_idx])
        disease = idx_to_name[cls_idx]

        # Stage 3+4 on diseased photos only (healthy -> severity 0)
        if cls_idx == args.healthy_idx:
            pct, qsi = 0.0, 0.0
        else:
            with torch.no_grad():
                lg = lesion(cx, torch.tensor([cls_idx], device=device))
                lp = torch.sigmoid(lg)[0, 0].cpu().numpy() * (m_r > 0)
            d = _damage_intensity(disp_r, m_r, ref, disease=disease)
            bark_px = float((m_r > 0).sum())
            pct = float((lp > args.lesion_thr).sum() / bark_px)   # % diseased bark
            qsi = float((lp * d).sum() / bark_px)

        gray = cv2.cvtColor(disp_r, cv2.COLOR_RGB2GRAY)
        w_area = float((m_r > 0).mean())
        w_sharp = sharpness(gray)
        weight = w_area * conf * w_sharp

        per_photo.append({"tree": tnum, "photo": pnum, "disease": disease,
                          "conf": conf, "pct_bark": pct, "qsi": qsi,
                          "w_area": w_area, "w_sharp": w_sharp, "weight": weight})

    pdf = pd.DataFrame(per_photo)
    pdf.to_csv(out / "tree_per_photo.csv", index=False)

    # ---- disease-specific treatment-stage bands (AUDITABLE LOOKUP TABLE) --
    # These are ACTION categories, not measurements. The percentage boundaries
    # are disease-specific because equal extent carries different urgency:
    #   * Stripe canker is progressive/lethal (cambial necrosis beyond the
    #     visible margin, girdling kills the stem) -> compressed (~1.7x) scale.
    #   * Rough bark is a periderm-quality disorder (tree usually survives;
    #     loss is peelability/quill grade) -> wider scale.
    # Kept as a table (not baked into the network) so it is documented,
    # auditable, and editable without retraining. Values in fraction 0..1.
    # STARTING PROPOSAL — anchor each band to an actual DEA-recommended
    # intervention with a domain expert before final use.
    DISEASE_BANDS = {
        "Rough bark":   {"prev": 0.30, "early": 0.50, "active": 0.80},
        "stripecanker": {"prev": 0.20, "early": 0.40, "active": 0.70},
        "_default":     {"prev": 0.05, "early": 0.15, "active": 0.50},
    }

    def treatment_stage(pct, disease):
        """Continuous % -> stage via a disease-conditioned lookup on % area."""
        b = DISEASE_BANDS.get(disease, DISEASE_BANDS["_default"])
        if pct < b["prev"]:
            return "Preventive"
        if pct < b["early"]:
            return "Early control"
        if pct < b["active"]:
            return "Active management"
        return "Severe outbreak"

    # ---- aggregate per tree per disease (weighted) -----------------------
    # TWO SEPARATE FACTS ARE COMPUTED -- but only ONE decides the stage:
    #   1) SEVERITY = weighted % diseased area (extent) + QSI (intensity).
    #      -> this alone determines severity_stage / action_stage by default.
    #   2) SPREAD   = fraction of views showing the disease, a proxy for
    #                 circumferential coverage (girdling risk indicator).
    #      -> reported for every row, does NOT change the stage unless
    #         --escalate_by_spread is explicitly passed.
    # Rationale: a downstream component (the decision engine, Module 03)
    # independently derives a stage from severity_percentage. If this script
    # also silently escalated the stage based on spread, the same tree could
    # be reported at two different stages by two different parts of the
    # system for the same underlying percentage. Spread/girdling stay
    # visible as decision-support context; they just don't overwrite the
    # stage label by default.
    rows = []
    all_trees = sorted(pdf.tree.unique())
    trees_with_disease = set()
    for (tnum, disease), g in pdf.groupby(["tree", "disease"]):
        if disease == idx_to_name[args.healthy_idx]:
            continue
        if len(g) < args.min_views:
            continue
        w = g.weight.values
        wsum = w.sum() if w.sum() > 0 else 1.0
        pct = float((g.pct_bark.values * w).sum() / wsum)
        qsi = round(float((g.qsi.values * w).sum() / wsum), 4)

        total_views = int((pdf.tree == tnum).sum())
        spread = len(g) / max(total_views, 1)          # circumference proxy

        severity_stage = treatment_stage(pct, disease)
        girdling = spread >= args.girdle_spread         # informational flag

        stage_order = ["Preventive", "Early control", "Active management", "Severe outbreak"]
        if args.escalate_by_spread and girdling:
            idx = stage_order.index(severity_stage)
            action_stage = stage_order[min(idx + 1, 3)]
            escalated = action_stage != severity_stage
        else:
            # default: stage = severity alone, so it can never disagree with
            # a stage computed downstream from the same severity_percentage
            action_stage = severity_stage
            escalated = False

        rows.append({
            "tree": f"t{tnum:02d}", "disease": disease,
            "pct_bark": round(pct, 4),
            "qsi": qsi,
            "severity_stage": severity_stage,          # from extent alone
            "spread": round(spread, 3),
            "girdling_risk": "yes" if girdling else "no",   # informational only
            "action_stage": action_stage,              # = severity_stage unless --escalate_by_spread
            "escalated_by_spread": "yes" if escalated else "no",
            "n_views": int(len(g)),
            "total_views": total_views,
            "mean_conf": round(float(g.conf.mean()), 3),
        })
        trees_with_disease.add(tnum)

    for tnum in all_trees:
        if tnum not in trees_with_disease:
            nv = int((pdf.tree == tnum).sum())
            rows.append({
                "tree": f"t{tnum:02d}", "disease": "No significant disease",
                "pct_bark": 0.0, "qsi": 0.0, "severity_stage": "Preventive",
                "spread": 0.0, "girdling_risk": "no", "action_stage": "Preventive",
                "escalated_by_spread": "no", "n_views": nv, "total_views": nv,
                "mean_conf": None})

    tdf = pd.DataFrame(rows).sort_values(["tree", "qsi"], ascending=[True, False])
    tdf.to_csv(out / "tree_severity.csv", index=False)

    # ---- report -----------------------------------------------------------
    mode = ("escalation ON -- action_stage may exceed severity_stage on "
            "girdling risk" if args.escalate_by_spread else
            "escalation OFF (default) -- action_stage = severity_stage; "
            "spread/girdling shown for information only")
    print("\n" + "=" * 72)
    print("TREE SEVERITY -- % area (disease-specific bands) + spread")
    print(f"stage policy: {mode}")
    print("=" * 72)
    if len(tdf) == 0:
        print("no trees processed")
    else:
        for tree, g in tdf.groupby("tree"):
            print(f"\n{tree}:")
            for _, r in g.iterrows():
                if r.disease == "No significant disease":
                    print(f"   No significant disease  "
                          f"({r.n_views} views) -> Preventive")
                else:
                    esc = "  (escalated by spread)" if r.escalated_by_spread == "yes" else ""
                    print(f"   {r.disease:14s}  {r.pct_bark*100:4.0f}% area "
                          f"[{r.severity_stage}]  +  spread {r.spread*100:3.0f}% "
                          f"({r.n_views}/{r.total_views} views, girdling "
                          f"{r.girdling_risk})")
                    print(f"       -> STAGE SENT DOWNSTREAM: {r.action_stage}{esc}   "
                          f"(QSI {r.qsi:.3f}, conf {r.mean_conf:.2f})")

    # trees flagged with >1 disease
    dis = tdf[tdf.disease != "No significant disease"]
    multi = dis.groupby("tree").size()
    multi = multi[multi > 1]
    if len(multi):
        print(f"\n{len(multi)} tree(s) with more than one disease: "
              f"{', '.join(multi.index)}")

    print(f"\nper-photo detail -> {out / 'tree_per_photo.csv'}")
    print(f"tree severity    -> {out / 'tree_severity.csv'}")


if __name__ == "__main__":
    main()