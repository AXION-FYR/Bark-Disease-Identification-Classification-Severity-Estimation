# Cinnamon bark disease pipeline — results

## DATASET

Total images: 950

```
class_name  Rough bark  healthy bark  stripecanker
split                                             
test                49            36            35
train              274           197           217
valid               51            44            47
```


## STAGE 1 — Bark segmentation (U-Net + EfficientNet-B0)

Test IoU **0.8993**, Dice 0.9350


## STAGE 2 — Disease classification (ablation, mean ± std over seeds)


## STAGE 3 — Lesion localisation (weakly supervised vs Grad-CAM)


## STAGE 4 — QSI severity
