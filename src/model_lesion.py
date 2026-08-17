"""
Stage 3 model — weakly-supervised lesion decoder.  Novelty Claims 2 & 3.

You have image-level disease labels and whole-trunk masks, but NO lesion
labels. This decoder learns to localise lesions from three weak signals
(see losses_lesion.py). That is the contribution.

  * FROZEN encoder: EfficientNet-B0 warm-started from the Stage-2 dualbranch_se
    checkpoint. Frozen because 950 images is too few to learn an encoder under
    weak supervision, and it ties the pipeline together.
  * FiLM conditioning (Claim 2): the predicted disease class produces per-channel
    (gamma, beta) that modulate the decoder, so ONE decoder computes different
    functions for rough bark vs stripe canker — learned, not hand-coded.
    Ablate with class_conditioned=False to prove it matters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenEncoder(nn.Module):
    def __init__(self, stage2_ckpt: str | None = None,
                 encoder: str = "efficientnet-b0"):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            encoder.replace("-", "_"), pretrained=(stage2_ckpt is None),
            features_only=True, out_indices=(1, 2, 3, 4))
        self.channels = self.backbone.feature_info.channels()

        if stage2_ckpt is not None:
            self._load_stage2(stage2_ckpt)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _load_stage2(self, ckpt_path: str):
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = st.get("model", st)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        missing, unexpected = self.backbone.load_state_dict(enc_sd, strict=False)
        print(f"  frozen encoder: {len(enc_sd)} Stage-2 tensors loaded "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    def train(self, mode: bool = True):
        return super().train(False)          # never leave eval: freeze BN stats

    @torch.no_grad()
    def forward(self, x):
        return self.backbone(x)


class FiLM(nn.Module):
    """class embedding -> per-channel (gamma, beta)."""

    def __init__(self, num_classes: int, channels: int, emb: int = 32):
        super().__init__()
        self.embed = nn.Embedding(num_classes, emb)
        self.to_gamma = nn.Linear(emb, channels)
        self.to_beta = nn.Linear(emb, channels)

    def forward(self, feat, class_idx):
        e = self.embed(class_idx)
        g = self.to_gamma(e).unsqueeze(-1).unsqueeze(-1)
        b = self.to_beta(e).unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + g) + b


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class LesionDecoder(nn.Module):
    def __init__(self, stage2_ckpt: str | None, num_classes: int = 3,
                 class_conditioned: bool = True, encoder: str = "efficientnet-b0"):
        super().__init__()
        self.class_conditioned = class_conditioned
        self.encoder = FrozenEncoder(stage2_ckpt, encoder)
        c1, c2, c3, c4 = self.encoder.channels

        self.film = FiLM(num_classes, c4) if class_conditioned else None
        self.up3 = UpBlock(c4, c3, 128)
        self.up2 = UpBlock(128, c2, 64)
        self.up1 = UpBlock(64, c1, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, img, class_idx=None):
        f1, f2, f3, f4 = self.encoder(img)
        if self.film is not None and class_idx is not None:
            f4 = self.film(f4, class_idx)
        d = self.up3(f4, f3)
        d = self.up2(d, f2)
        d = self.up1(d, f1)
        logit = self.head(d)
        return F.interpolate(logit, size=img.shape[-2:], mode="bilinear",
                             align_corners=False)


def build_lesion_decoder(stage2_ckpt: str | None, class_conditioned: bool = True,
                         num_classes: int = 3) -> LesionDecoder:
    return LesionDecoder(stage2_ckpt, num_classes, class_conditioned)
