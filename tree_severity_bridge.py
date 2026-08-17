"""
tree_severity_bridge.py
=========================
In-memory adaptation of scripts/26_demo_tree.py for live Streamlit use.

PLACE THIS FILE AT YOUR PROJECT ROOT -- the same level as scripts/ and src/
(i.e. next to 26_demo_tree.py's project root, D:/RESEARCH/.../cinnamon/).
That's what makes the `from src...` imports below resolve exactly like
they do in the original script.

Model loading, per-photo inference, and per-disease aggregation are
reused 1:1 from scripts/26_demo_tree.py. The only thing that changes is
the I/O layer: instead of reading a folder of files from disk and saving
a matplotlib figure, BarkSeverityPipeline.analyze() takes PIL Images
already in memory (straight from Streamlit's file_uploader) and returns
a JSON-serializable dict ready to forward to Module 03.

IMPORTANT: instantiate BarkSeverityPipeline ONCE and reuse it -- model
loading is the slow part. In app_ui.py this is done with
@st.cache_resource so Streamlit doesn't reload the models on every rerun.

STAGE POLICY (changed -- see below):
  per_disease[dis]["stage"] is now the SEVERITY-based stage ONLY (from %
  diseased area via the disease-specific band table). Circumferential
  spread and girdling risk are still computed and returned (as "spread"
  and "girdling_risk"), but they NO LONGER silently bump the stage that
  gets sent to Module 03. Reason: Module 03's decision engine derives its
  own stage from severity_percentage on its side. If this bridge ALSO
  escalated the stage before sending it, the same tree could be described
  at two different stages by two different parts of the system for the
  same underlying percentage -- confusing, and it undermines trust in the
  report. Exactly one place now owns "percentage -> stage": the decision
  engine. Spread/girdling remain available in the response as decision-
  support context, not as something baked silently into the stage label.
  Pass escalate_by_spread=True to the constructor to restore the old
  behaviour (e.g. for a standalone demo where nothing downstream also
  computes a stage).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, PROJECT_ROOT              # noqa: E402
from src.imaging import letterbox_image                        # noqa: E402
from src.dataset_cls import crop_to_bbox, clahe_lab             # noqa: E402
from src.dataset_cls import IMAGENET_MEAN, IMAGENET_STD         # noqa: E402
from src.model_cls import build_classifier                     # noqa: E402
from src.model_lesion import build_lesion_decoder               # noqa: E402

# ---- copied unchanged from scripts/26_demo_tree.py ----
DISEASE_WEIGHTS = {"Rough bark": (0.70, 0.30), "stripecanker": (0.30, 0.70), "_default": (0.50, 0.50)}
DISEASE_BANDS = {
    "Rough bark":   {"prev": 0.30, "early": 0.50, "active": 0.80},
    "stripecanker": {"prev": 0.20, "early": 0.40, "active": 0.70},
    "_default":     {"prev": 0.05, "early": 0.15, "active": 0.50},
}
STAGE_ORDER = ["Preventive", "Early control", "Active management", "Severe outbreak"]


def texture_energy(gray):
    g = gray.astype(np.float32)
    m = cv2.blur(g, (9, 9))
    sq = cv2.blur(g * g, (9, 9))
    return np.sqrt(np.clip(sq - m * m, 0, None))


def damage_intensity(disp, bark, ref, disease=None):
    gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(disp, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    tex = texture_energy(gray)
    d_tex = np.clip((tex - ref["tex_mean"]) / (ref["tex_std"] + 1e-6), 0, None)
    d_dark = np.clip((ref["L_mean"] - L) / (ref["L_std"] + 1e-6), 0, None)
    wt, wd = DISEASE_WEIGHTS.get(disease, DISEASE_WEIGHTS["_default"])
    return np.clip((wt * d_tex + wd * d_dark) / 3.0, 0, 1) * (bark > 0)


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def treatment_stage(pct: float, disease: str) -> str:
    b = DISEASE_BANDS.get(disease, DISEASE_BANDS["_default"])
    if pct < b["prev"]:
        return "Preventive"
    if pct < b["early"]:
        return "Early control"
    if pct < b["active"]:
        return "Active management"
    return "Severe outbreak"


class BarkSeverityPipeline:
    """
    Loads the 5-stage Bark AI models ONCE (segmentation, dual-branch
    classifier, lesion decoder, healthy reference stats), then serves
    cheap .analyze() calls. Same checkpoints/paths as 26_demo_tree.py.
    """

    def __init__(
        self,
        seg_ckpt: str = "outputs/seg/best.pt",
        stage2_ckpt: str = "outputs/cls/cls_mcse_allones.pt",
        lesion_ckpt: str = "outputs/lesion/lesion_film.pt",
        seg_size: int = 512,
        cls_size: int = 224,
        healthy_idx: int = 1,
        lesion_thr: float = 0.5,
        min_views: int = 2,
        girdle_spread: float = 0.6,
        escalate_by_spread: bool = False,
    ):
        self.seg_size = seg_size
        self.cls_size = cls_size
        self.healthy_idx = healthy_idx
        self.lesion_thr = lesion_thr
        self.min_views = min_views
        self.girdle_spread = girdle_spread
        # OFF by default -- see STAGE POLICY note at the top of this file.
        # Module 03 independently derives a stage from severity_percentage;
        # this bridge should not also silently move the stage, or the same
        # tree can end up reported at two disagreeing stages.
        self.escalate_by_spread = escalate_by_spread

        self.cfg = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classes = json.load(open(self.cfg.path("paths", "processed_root") / "classes.json"))
        self.idx_to_name = {v: k for k, v in classes["class_to_idx"].items()}

        seg_st = torch.load(PROJECT_ROOT / seg_ckpt, map_location=self.device, weights_only=False)
        self.seg = smp.Unet(
            encoder_name=seg_st.get("encoder", "efficientnet-b0"),
            encoder_weights=None, in_channels=3, classes=1,
        ).to(self.device)
        self.seg.load_state_dict(seg_st["model"])
        self.seg.eval()

        s2 = torch.load(PROJECT_ROOT / stage2_ckpt, map_location=self.device, weights_only=False)
        self.clf = build_classifier(variant=s2.get("variant", "mcse_allones"), pretrained=False).to(self.device)
        self.clf.load_state_dict(s2["model"])
        self.clf.eval()

        les_st = torch.load(PROJECT_ROOT / lesion_ckpt, map_location=self.device, weights_only=False)
        self.lesion = build_lesion_decoder(
            str(PROJECT_ROOT / stage2_ckpt), class_conditioned=les_st.get("film", True)
        ).to(self.device)
        self.lesion.load_state_dict(les_st["model"])
        self.lesion.eval()

        ref_path = PROJECT_ROOT / "outputs" / "qsi" / "healthy_reference.json"
        if not ref_path.exists():
            raise FileNotFoundError(
                "outputs/qsi/healthy_reference.json missing -- run scripts/17_qsi.py "
                "once before starting the app."
            )
        self.ref = json.load(open(ref_path))

    def _run_one_photo(self, img_rgb: np.ndarray, image_name: str = "") -> Optional[dict]:
        seg_in, p = letterbox_image(img_rgb, self.seg_size)
        x = ((seg_in.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
        x = torch.tensor(np.ascontiguousarray(x.transpose(2, 0, 1))[None], device=self.device)
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")
        ):
            bark_lb = (torch.sigmoid(self.seg(x))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        inner = bark_lb[p.pad_top:self.seg_size - p.pad_bottom, p.pad_left:self.seg_size - p.pad_right]
        bark = cv2.resize(inner, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        if bark.sum() < 100:
            return None  # same "skip -- no bark detected" rule as 26_demo_tree.py

        crop, m = crop_to_bbox(img_rgb, bark, 0.05)
        disp = clahe_lab(crop)
        disp_r = cv2.resize(disp, (self.cls_size, self.cls_size), interpolation=cv2.INTER_AREA)
        m_r = cv2.resize(m, (self.cls_size, self.cls_size), interpolation=cv2.INTER_NEAREST)
        cx = ((disp_r.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD)
        cx = torch.tensor(np.ascontiguousarray(cx.transpose(2, 0, 1))[None], device=self.device)
        mm = torch.tensor(m_r.astype(np.float32)[None, None], device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.clf(cx * mm, mm), 1)[0].cpu().numpy()
        cls_idx = int(probs.argmax())
        conf = float(probs[cls_idx])
        disease = self.idx_to_name[cls_idx]

        if cls_idx == self.healthy_idx:
            lp = np.zeros((self.cls_size, self.cls_size), np.float32)
            pct = qsi = 0.0
        else:
            with torch.no_grad():
                lp = torch.sigmoid(
                    self.lesion(cx, torch.tensor([cls_idx], device=self.device))
                )[0, 0].cpu().numpy() * (m_r > 0)
            d = damage_intensity(disp_r, m_r, self.ref, disease)
            bpx = float((m_r > 0).sum())
            pct = float((lp > self.lesion_thr).sum() / bpx)
            qsi = float((lp * d).sum() / bpx)

        gray = cv2.cvtColor(disp_r, cv2.COLOR_RGB2GRAY)
        w = float((m_r > 0).mean()) * conf * sharpness(gray)
        return {
                "image_name": image_name,
                "disease": disease,
                "conf": conf,
                "qsi": qsi,
                "pct": pct,
                "weight": w,
                "stage": treatment_stage(pct, disease)
            }

    def analyze(self, tree_id: str, images: List[Image.Image]) -> dict:
        """
        images: PIL Images already in memory (from
        load_images_from_streamlit_uploads()). Mirrors the aggregation
        logic in scripts/26_demo_tree.py exactly, minus the figure.
        """
        results = []
        for idx, img in enumerate(images):
            img_rgb = np.array(img.convert("RGB"))

            r = self._run_one_photo(
                img_rgb,
                image_name=f"bark_image_{idx+1}.jpg"
            )

            if r is not None:
                results.append(r)

        if not results:
            raise ValueError(f"No photo produced a usable bark mask for tree '{tree_id}'")

        per_disease = {}
        for dis in set(r["disease"] for r in results):
            if dis == self.idx_to_name[self.healthy_idx]:
                continue
            g = [r for r in results if r["disease"] == dis]
            if len(g) < self.min_views:
                continue
            w = np.array([max(r["weight"], 1e-6) for r in g])
            q = float((np.array([r["qsi"] for r in g]) * w).sum() / w.sum())
            pc = float((np.array([r["pct"] for r in g]) * w).sum() / w.sum())
            spread = len(g) / max(len(results), 1)

            sev_stage = treatment_stage(pc, dis)
            girdling = spread >= self.girdle_spread   # informational flag only

            # STAGE POLICY: by default the stage sent downstream is the
            # severity-based stage ALONE. Spread and girdling risk are
            # still returned for every disease, but they do not silently
            # change "stage" unless escalate_by_spread was explicitly
            # requested when this pipeline was constructed. This keeps
            # Module 03 as the single place that turns a percentage into
            # a stage, so the two components can never disagree about the
            # same tree.
            if self.escalate_by_spread and girdling:
                idx = STAGE_ORDER.index(sev_stage)
                action = STAGE_ORDER[min(idx + 1, 3)]
            else:
                action = sev_stage

            per_disease[dis] = {
                "qsi": q, "pct": pc, "n": len(g), "spread": spread,
                "severity_stage": sev_stage,           # from % area alone
                "girdling_risk": girdling,              # informational only
                "stage": action,                        # sent downstream
                "escalated": action != sev_stage,
            }

        if not per_disease:
            return {
                "tree_id": tree_id,
                "per_disease_bark": {},
                "per_image_results": results,
                "num_views_processed": len(results),
            }

        return {
                        "tree_id": tree_id,

                        "per_disease_bark": {
                            dis: {
                                "pct_bark": v["pct"] * 100,
                                "qsi": v["qsi"],
                                "stage": v["stage"],
                                "severity_stage": v["severity_stage"],
                                "spread": v["spread"],
                                "girdling_risk": v["girdling_risk"],
                            }
                            for dis, v in per_disease.items()
                        },

                        "per_image_results": [
                            {
                                "image_name": r["image_name"],
                                "disease": r["disease"],
                                "severity_pct": r["pct"] * 100,
                                "confidence": r["conf"] * 100,
                                "qsi": r["qsi"],
                                "stage": r["stage"]
                            }
                            for r in results
                        ],

                        "num_views_processed": len(results),
                    }


def load_images_from_streamlit_uploads(uploaded_files) -> List[Image.Image]:
    """Convert Streamlit's UploadedFile objects into PIL Images, in memory."""
    return [Image.open(f).convert("RGB") for f in uploaded_files]