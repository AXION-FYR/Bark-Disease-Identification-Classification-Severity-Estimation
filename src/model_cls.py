"""
Stage 2 model — dual-branch classifier with mask-conditioned SE fusion.

This is Novelty Claim 1. The parts an examiner will scrutinise are:

  1. TWO complementary descriptors, not one.
     - appearance branch: EfficientNet-B0 (semantic / colour / global shape)
     - texture branch    : a fixed LBP-style operator + small CNN. LBP is
       ordinal (compares each pixel to its neighbours), so it is robust to the
       lighting variation across an outdoor plantation and captures the
       micro-texture that separates rough bark from healthy.

  2. MASK-CONDITIONED SE fusion.
     A standard Squeeze-and-Excitation block pools over the whole feature map,
     background included. Here the squeeze is a BARK-MASKED global average, so
     the channel-recalibration weights are driven only by bark statistics and
     never by soil/leaves/sky. The gate then reweights the concatenated
     appearance+texture channels before the classifier head.

  The ablation flags below let you turn each piece off, which is how you PROVE
  the mechanism rather than assert it:

     variant="plain"        EfficientNet-B0 only, no mask, no texture
     variant="masked"       EfficientNet-B0 on masked input
     variant="concat"       appearance + texture, plain concat (no SE)
     variant="se"           appearance + texture, standard (unmasked) SE
     variant="mcse"         appearance + texture, mask-conditioned SE  <- OURS
     variant="mcse_allones" OURS but the mask is replaced by all-ones,
                            isolating exactly what the mask conditioning buys
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# texture branch
# --------------------------------------------------------------------------
class LBPConv(nn.Module):
    """
    Differentiable LBP-style texture extractor.

    True LBP thresholds each of the 8 neighbours against the centre and packs
    the result into an 8-bit code. That is non-differentiable and its 256-way
    codes are awkward for a CNN. This uses 8 fixed difference filters
    (neighbour minus centre) followed by a smooth sign (tanh), giving 8 ordinal
    "is-brighter-than-centre" channels that preserve LBP's core property —
    invariance to any monotonic change in illumination — while staying
    differentiable and cheap. The weights are FIXED (requires_grad=False); only
    the small CNN on top of them learns.
    """

    def __init__(self, sharpness: float = 8.0):
        super().__init__()
        self.sharpness = sharpness
        k = torch.zeros(8, 1, 3, 3)
        neigh = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        for i, (r, c) in enumerate(neigh):
            k[i, 0, 1, 1] = -1.0     # centre
            k[i, 0, r, c] = 1.0      # neighbour
        self.register_buffer("kernels", k)

    def forward(self, gray):                       # gray: (B,1,H,W)
        d = F.conv2d(gray, self.kernels, padding=1)
        return torch.tanh(self.sharpness * d)      # (B,8,H,W), in (-1,1)


class TextureBranch(nn.Module):
    """LBP features -> small CNN -> feature map with `out_ch` channels."""

    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.lbp = LBPConv()
        self.net = nn.Sequential(
            nn.Conv2d(8, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
        self.out_ch = out_ch

    def forward(self, rgb):
        gray = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2]
                + 0.114 * rgb[:, 2:3])
        return self.net(self.lbp(gray))


# --------------------------------------------------------------------------
# mask-conditioned SE fusion
# --------------------------------------------------------------------------
class MaskConditionedSE(nn.Module):
    """
    SE block whose squeeze is a bark-masked global average pool.

    standard SE:   z_c = (1/HW) * sum_hw  x_{c,h,w}
    mask-cond SE:  z_c = (sum_hw m*x) / (sum_hw m)      # average over bark only

    The excitation MLP and the per-channel gating are identical to standard SE;
    only the pooling denominator changes. That is deliberately the single
    isolated variable, so the ablation attributes any gain to the conditioning
    and nothing else.
    """

    def __init__(self, channels: int, reduction: int = 16, masked: bool = True):
        super().__init__()
        self.masked = masked
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x, mask=None):               # x:(B,C,H,W) mask:(B,1,H,W)
        B, C, H, W = x.shape
        if self.masked and mask is not None:
            m = F.interpolate(mask, size=(H, W), mode="nearest")
            denom = m.sum(dim=(2, 3)).clamp(min=1.0)          # (B,1)
            z = (x * m).sum(dim=(2, 3)) / denom               # (B,C)
        else:
            z = x.mean(dim=(2, 3))                             # (B,C)
        gate = self.fc(z).view(B, C, 1, 1)
        return x * gate


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------
def _build_encoder(name: str = "efficientnet-b0", pretrained: bool = True):
    import timm
    m = timm.create_model(name.replace("-", "_"), pretrained=pretrained,
                          features_only=True, out_indices=(4,))
    ch = m.feature_info.channels()[-1]
    return m, ch


class DualBranchClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, variant: str = "mcse",
                 encoder: str = "efficientnet-b0", pretrained: bool = True):
        super().__init__()
        assert variant in {"plain", "masked", "concat", "se", "mcse", "mcse_allones"}
        self.variant = variant

        self.encoder, enc_ch = _build_encoder(encoder, pretrained)
        self.use_texture = variant in {"concat", "se", "mcse", "mcse_allones"}
        self.use_mask_input = variant in {"masked", "concat", "se", "mcse",
                                          "mcse_allones"}

        if self.use_texture:
            self.texture = TextureBranch(out_ch=64)
            fused_ch = enc_ch + 64
        else:
            fused_ch = enc_ch

        if variant in {"se", "mcse", "mcse_allones"}:
            self.se = MaskConditionedSE(
                fused_ch, masked=(variant in {"mcse", "mcse_allones"}))
        else:
            self.se = None

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(fused_ch, num_classes),
        )

    def forward(self, img, mask=None):
        # "plain" ignores the mask entirely; every masked variant expects the
        # background to be zeroed already by the dataset (apply_mask=True).
        feat = self.encoder(img)[-1]                    # (B, enc_ch, h, w)

        if self.use_texture:
            t = self.texture(img)
            if t.shape[-2:] != feat.shape[-2:]:
                t = F.interpolate(t, size=feat.shape[-2:], mode="bilinear",
                                  align_corners=False)
            feat = torch.cat([feat, t], dim=1)

        if self.se is not None:
            se_mask = mask
            if self.variant == "mcse_allones" and mask is not None:
                se_mask = torch.ones_like(mask)
            feat = self.se(feat, se_mask)

        return self.head(feat)


def build_classifier(variant: str = "mcse", num_classes: int = 3,
                     encoder: str = "efficientnet-b0",
                     pretrained: bool = True) -> DualBranchClassifier:
    return DualBranchClassifier(num_classes, variant, encoder, pretrained)
