# Overlap Gate Summary

## Thresholds

- promote_threshold_pct: `3.0`
- drop_threshold_pct: `-2.0`
- min_pairs: `2`
- max_queue_spread_pct: `8.0`

## Per Queue Directory

| dir | gain_all_pct | gain_steady_pct | off_n | on_n | off_cv_pct | on_cv_pct |
|---|---:|---:|---:|---:|---:|---:|
| logs/awave-cw2-overlap-ab-a100-2g-pq | 4.17 | 0.85 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-2g-uq | 4.85 | 1.54 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-4g-pq | 141.42 | 112.55 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-pq | 4.60 | 1.29 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-uq | 3.83 | 1.48 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-pq | 6.14 | 6.57 | 1 | 1 | nan | nan |

## Group Decision

| gpu | gpus | gain_steady_pct | pairs | queue_spread_pct | status | recommended_action |
|---|---:|---:|---:|---:|---|---|
| a100 | 2 | 1.20 | 2 | 0.69 | needs-retest | run-more-2g-repeats-before-keep-or-drop |
| a100 | 4 | 112.55 | 1 | 0.00 | provisional-promote | keep-overlap-on-for-this-gpu-count |
| h100 | 2 | 1.39 | 2 | 0.18 | needs-retest | run-more-2g-repeats-before-keep-or-drop |
| h100 | 8 | 6.57 | 1 | 0.00 | provisional-promote | keep-overlap-on-for-this-gpu-count |

