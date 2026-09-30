# Results

Subset: 4 VisDrone2019-MOT validation sequences, the first 70 annotated frames of each at every 3rd frame (about 10 fps effective). Detector: torchvision Faster R-CNN ResNet-50 FPN v2 (COCO). Person = VisDrone classes pedestrian + people. The score threshold (0.5) and all tracker parameters were fixed before evaluation and not tuned.

## full-frame (score threshold 0.5)

| sequence | GT persons | AP@0.5 | precision | recall | MOTA | IDF1 | ID sw. | count MAE | count bias | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| uav0000086_00000_v | 2740 | 0.708 | 0.805 | 0.742 | 0.559 | 0.667 | 57 | 3.17 | -3.03 | 0.43 |
| uav0000117_02622_v | 1718 | 0.405 | 0.568 | 0.460 | 0.008 | 0.276 | 149 | 5.17 | -4.66 | 1.33 |
| uav0000137_00458_v | 2857 | 0.674 | 0.653 | 0.782 | 0.326 | 0.610 | 72 | 8.10 | +8.07 | 0.45 |
| uav0000339_00001_v | 1420 | 0.560 | 0.567 | 0.603 | 0.038 | 0.397 | 39 | 3.24 | +1.27 | 0.62 |
| **mean** | | 0.586 | 0.648 | 0.647 | 0.233 | 0.488 | 317 (sum) | 4.92 | +0.41 | 0.58 |

Recall by ground-truth person height (pooled over sequences):

| height | found / total | recall |
|---|---|---|
| 0-40 px | 206 / 654 | 0.31 |
| 40-70 px | 1585 / 2541 | 0.62 |
| 70-110 px | 2964 / 4013 | 0.74 |
| 110-inf px | 1167 / 1527 | 0.76 |

Score-threshold sensitivity (pooled over all evaluation frames):

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.3 | 0.623 | 0.701 | 0.660 |
| 0.5 | 0.668 | 0.677 | 0.673 |
| 0.7 | 0.729 | 0.634 | 0.678 |
| 0.9 | 0.822 | 0.482 | 0.607 |

Camera-motion compensation ablation (tracker input identical):

| | MOTA | IDF1 | ID switches |
|---|---|---|---|
| with compensation | 0.233 | 0.488 | 317 |
| without | 0.171 | 0.343 | 781 |

Per-frame runtime, milliseconds (mean over sequences):

| detect | camera motion | track | density | total |
|---|---|---|---|---|
| 1613.3 | 102.9 | 2.0 | 7.0 | 1725.2 |

Count error of the density map integral and of the confirmed tracks (MAE, mean over sequences):

detections 4.92, density integral 4.92, confirmed tracks 5.21; mean GT count 31.2 persons per frame.

## tiled (640 px) (score threshold 0.5)

| sequence | GT persons | AP@0.5 | precision | recall | MOTA | IDF1 | ID sw. | count MAE | count bias | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| uav0000086_00000_v | 2740 | 0.742 | 0.772 | 0.780 | 0.526 | 0.662 | 53 | 1.84 | +0.39 | 0.16 |
| uav0000117_02622_v | 1718 | 0.437 | 0.473 | 0.533 | -0.191 | 0.259 | 198 | 4.39 | +3.07 | 0.30 |
| uav0000137_00458_v | 2857 | 0.692 | 0.555 | 0.816 | 0.072 | 0.524 | 112 | 19.19 | +19.19 | 0.13 |
| uav0000339_00001_v | 1420 | 0.593 | 0.439 | 0.659 | -0.382 | 0.337 | 54 | 10.23 | +10.17 | 0.08 |
| **mean** | | 0.616 | 0.560 | 0.697 | 0.006 | 0.445 | 417 (sum) | 8.91 | +8.20 | 0.14 |

Recall by ground-truth person height (pooled over sequences):

| height | found / total | recall |
|---|---|---|
| 0-40 px | 257 / 654 | 0.39 |
| 40-70 px | 1757 / 2541 | 0.69 |
| 70-110 px | 3091 / 4013 | 0.77 |
| 110-inf px | 1223 / 1527 | 0.80 |

Score-threshold sensitivity (pooled over all evaluation frames):

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.3 | 0.501 | 0.750 | 0.601 |
| 0.5 | 0.573 | 0.723 | 0.639 |
| 0.7 | 0.652 | 0.687 | 0.669 |
| 0.9 | 0.769 | 0.568 | 0.653 |

Camera-motion compensation ablation (tracker input identical):

| | MOTA | IDF1 | ID switches |
|---|---|---|---|
| with compensation | 0.006 | 0.445 | 417 |
| without | 0.006 | 0.329 | 922 |

Per-frame runtime, milliseconds (mean over sequences):

| detect | camera motion | track | density | total |
|---|---|---|---|---|
| 7271.4 | 102.9 | 3.1 | 4.9 | 7382.3 |

Count error of the density map integral and of the confirmed tracks (MAE, mean over sequences):

detections 8.91, density integral 8.91, confirmed tracks 7.64; mean GT count 31.2 persons per frame.

## Ground-plane analytics

Primary variant: tiled (640 px).

| sequence | est. camera height (m) | est. pitch (deg) | median track speed (m/s) | p90 speed (m/s) | peak density (/m2) | alert frames |
|---|---|---|---|---|---|---|
| uav0000086_00000_v | 8.9 | 25 | 0.73 | 2.29 | 0.30 | 0 / 70 |
| uav0000117_02622_v | 13.6 | 24 | 2.35 | 7.33 | 0.33 | 0 / 70 |
| uav0000137_00458_v | 13.6 | 28 | 3.57 | 7.48 | 0.25 | 0 / 70 |
| uav0000339_00001_v | 16.8 | 34 | 0.86 | 2.23 | 0.30 | 0 / 70 |

Figures: `counts.png`, `recall_by_height.png`, `ground_<sequence>.png` in this folder.
