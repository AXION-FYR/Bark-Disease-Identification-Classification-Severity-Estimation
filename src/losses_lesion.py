"""
Stage 3 losses — weak supervision for lesion localisation.  Novelty Claim 3.

No lesion labels exist. Supervision comes from three signals you DO have:

  1. HEALTHY-ANCHOR (the key idea).
     Every healthy trunk must have a near-zero lesion map inside the bark.
     277 healthy images give dense, free negative supervision — the piece most
     weakly-supervised methods lack. This is what stops the map lighting up on
     ordinary bark texture.

  2. BARK-CONTAINMENT.
     Lesion probability outside the bark mask is pushed to zero. Lesions live on
     bark, not on soil or leaves.

  3. MIL POOLING.
     For diseased trunks, the masked-pooled lesion map must be high (the image
     IS diseased somewhere on the bark); this ties the map to the image-level
     label you have. Uses a soft top-k / log-sum-exp pool so gradient reaches
     the most lesion-like pixels rather than the whole trunk.

Total = w_anchor*anchor + w_contain*contain + w_mil*mil  (+ optional TV smooth).

Warm-start: train anchor+containment for a few epochs BEFORE enabling MIL, or
the map can collapse to all-zero (satisfies anchor) or all-one (satisfies MIL).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_mask(mask, hw):
    if mask.shape[-2:] != hw:
        mask = F.interpolate(mask, size=hw, mode="nearest")
    return mask


def healthy_anchor_loss(prob, bark_mask, is_healthy):
    """
    prob:(B,1,H,W) sigmoid lesion map. bark_mask:(B,1,H,W). is_healthy:(B,) bool.
    Penalise any lesion probability inside bark on healthy trunks.
    """
    if is_healthy.sum() == 0:
        return prob.new_zeros(())
    m = _resize_mask(bark_mask, prob.shape[-2:])
    sel = is_healthy.view(-1, 1, 1, 1).float()
    num = (prob * m * sel).sum()
    den = (m * sel).sum().clamp(min=1.0)
    return num / den


def bark_containment_loss(prob, bark_mask):
    """Lesion probability outside the bark mask -> 0."""
    m = _resize_mask(bark_mask, prob.shape[-2:])
    outside = (1.0 - m)
    return (prob * outside).sum() / outside.sum().clamp(min=1.0)


def mil_pooling_loss(prob, bark_mask, is_diseased, tau: float = 0.1):
    """
    Diseased trunks must have a HIGH lesion response somewhere on the bark.
    Soft-max pool (log-sum-exp) over bark pixels, then BCE against 1.
    """
    if is_diseased.sum() == 0:
        return prob.new_zeros(())
    m = _resize_mask(bark_mask, prob.shape[-2:]).bool()
    B = prob.shape[0]
    pooled = prob.new_zeros(B)
    for i in range(B):
        vals = prob[i, 0][m[i, 0]]
        if vals.numel() == 0:
            pooled[i] = prob[i].mean()
        else:
            # smooth max: tau->0 approaches true max
            pooled[i] = tau * torch.logsumexp(vals / tau, dim=0) \
                - tau * torch.log(torch.tensor(float(vals.numel()),
                                               device=vals.device))
    pooled = pooled.clamp(1e-6, 1 - 1e-6)
    sel = is_diseased.float()
    bce = -(sel * torch.log(pooled)).sum() / sel.sum().clamp(min=1.0)
    return bce


def tv_loss(prob):
    """Total variation — encourages spatially coherent blobs over speckle."""
    dh = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs().mean()
    dw = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs().mean()
    return dh + dw


def lesion_loss(logit, bark_mask, class_idx, healthy_idx: int = 1,
                w_anchor: float = 1.0, w_contain: float = 1.0,
                w_mil: float = 1.0, w_tv: float = 0.05,
                w_sparse: float = 0.3,
                enable_mil: bool = True):
    """
    Combined weak-supervision loss. Returns (total, parts_dict).

    healthy_idx: the class index of 'healthy bark' (default 1 for the order
                 ['Rough bark','healthy bark','stripecanker']).
    enable_mil:  set False during the warm-start phase.
    """
    prob = torch.sigmoid(logit)
    is_healthy = (class_idx == healthy_idx)
    is_diseased = ~is_healthy

    la = healthy_anchor_loss(prob, bark_mask, is_healthy)
    lc = bark_containment_loss(prob, bark_mask)
    lm = mil_pooling_loss(prob, bark_mask, is_diseased) if enable_mil \
        else prob.new_zeros(())
    lt = tv_loss(prob)

    # sparsity: lesions occupy a FRACTION of the bark, not all of it. Penalise
    # mean lesion probability inside bark so the map cannot win by flooding the
    # whole trunk. This is what forces it to mark the worst regions, which is
    # also what makes QSI severity meaningful rather than saturated.
    m_sp = _resize_mask(bark_mask, prob.shape[-2:])
    l_sparse = (prob * m_sp).sum() / m_sp.sum().clamp(min=1.0)

    total = w_anchor * la + w_contain * lc + w_tv * lt + w_sparse * l_sparse
    if enable_mil:
        total = total + w_mil * lm

    return total, {"anchor": la.item(), "contain": lc.item(),
                   "mil": lm.item() if enable_mil else 0.0, "tv": lt.item(),
                   "sparse": l_sparse.item()}
