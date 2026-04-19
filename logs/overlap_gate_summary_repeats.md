# Overlap Gate Summary

## Thresholds

- promote_threshold_pct: `3.0`
- drop_threshold_pct: `-2.0`
- no_promote_threshold_pct: `2.0`
- min_pairs: `2`
- max_queue_spread_pct: `8.0`

## Per Queue Directory

| dir | repeat | gain_all_pct | gain_steady_pct | off_n | on_n | off_cv_pct | on_cv_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| logs/awave-cw2-overlap-ab-a100-2g-pq-r1 | 1 | 6.98 | 1.01 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-2g-pq-r2 | 2 | 4.16 | 1.03 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-2g-pq-r3 | 3 | 3.93 | 0.97 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-2g-pq-r4 | 4 | 3.88 | 0.85 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-2g-pq-r5 | 5 | 5.00 | 1.01 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-2g-uq-r1 | 1 | 4.48 | 1.11 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-2g-uq-r2 | 2 | 5.19 | 1.47 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-a100-2g-uq-r3 | 3 | 4.34 | 1.03 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-pq-r1 | 1 | 4.46 | 1.38 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-pq-r2 | 2 | 4.84 | 1.41 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-pq-r3 | 3 | 5.23 | 1.35 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-pq-r4 | 4 | 3.80 | 1.23 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-uq-r1 | 1 | -3.85 | 1.22 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-uq-r2 | 2 | 8.96 | 1.19 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-uq-r3 | 3 | 4.95 | 1.11 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-uq-r4 | 4 | 4.24 | 1.30 | 1 | 1 | nan | nan |
| logs/awave-cw2-overlap-ab-h100-2g-uq-r5 | 5 | 5.99 | 1.20 | 1 | 1 | nan | nan |

## Group Decision

| gpu | gpus | gain_steady_pct | pairs | queue_spread_pct | status | recommended_action |
|---|---:|---:|---:|---:|---|---|
| a100 | 2 | 1.06 | 8 | 0.23 | no-promote | stop-overlap-campaign-and-move-to-next-optimization |
| h100 | 2 | 1.27 | 9 | 0.14 | no-promote | stop-overlap-campaign-and-move-to-next-optimization |

