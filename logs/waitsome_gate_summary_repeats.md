# Waitsome Gate Summary

## Thresholds

- promote_threshold_pct: `3.0`
- drop_threshold_pct: `-2.0`
- no_promote_threshold_pct: `2.0`
- min_pairs: `3`
- max_queue_spread_pct: `8.0`

## Per Queue Directory

| dir | repeat | gain_all_pct | gain_steady_pct | off_n | on_n | off_cv_pct | on_cv_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| logs/awave-cw2-waitsome-ab-a100-2g-uq-r1 | 1 | -0.12 | -0.08 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-a100-2g-uq-r2 | 2 | 1.84 | -0.13 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-a100-2g-uq-r3 | 3 | 0.01 | 0.02 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-a100-2g-uq-r4 | 4 | -0.04 | 0.02 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-a100-2g-uq-r5 | 5 | -0.02 | 0.03 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-pq-r1 | 1 | 2.12 | -0.01 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-pq-r2 | 2 | 0.32 | -0.00 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-pq-r3 | 3 | -8.04 | -5.75 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-pq-r4 | 4 | -0.28 | 0.02 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-pq-r5 | 5 | 1.70 | 0.02 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-uq-r1 | 1 | -0.39 | -0.05 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-uq-r2 | 2 | -0.32 | -0.07 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-uq-r3 | 3 | 0.17 | -0.02 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-uq-r4 | 4 | -0.14 | 0.07 | 1 | 1 | nan | nan |
| logs/awave-cw2-waitsome-ab-h100-2g-uq-r5 | 5 | -0.67 | 0.15 | 1 | 1 | nan | nan |

## Group Decision

| gpu | gpus | gain_steady_pct | pairs | queue_spread_pct | status | recommended_action |
|---|---:|---:|---:|---:|---|---|
| a100 | 2 | -0.03 | 5 | 0.00 | no-promote | keep-waitsome-off-and-move-next-optimization |
| h100 | 2 | -0.56 | 10 | 1.16 | no-promote | keep-waitsome-off-and-move-next-optimization |

