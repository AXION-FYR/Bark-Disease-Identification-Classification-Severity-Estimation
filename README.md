# Bark-Disease-Identification-Classification-Severity-Estimation
This module is identifying and classifying the two largest cinnamon bark disease, Rough Bark Disease and Stripe Canker, through visual analysis of bark images. This module will fill the key gap in research of automation of disease severity.

Input Layer - Raw bark images from field (15 bark images)
Stage 1 — Bark Segmentation - Removes leaves, soil, sky — cleans input for downstream stages
Stage 2 — Disease Classification - Handles the two diseases
' different visual signatures (colour vs.
surface texture)
Stage 3 — Lesion Localisation - Weakly supervised — trained without pixel-level lesion annotations
Stage 4 — Severity (QSI) - Combines lesion extent with disease-conditioned damage intensity
Stage 5 — Tree Aggregation - Combines multiple per-tree images into one severity value per disease
Output Layer
Disease type , tree-level disease-wise severity , disease stage (Preventive/Early/Active/Severe)

