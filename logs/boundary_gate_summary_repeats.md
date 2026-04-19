# Boundary Split Gate Summary

## Thresholds

- promote_threshold_pct: `3.0`
- drop_threshold_pct: `-2.0`
- no_promote_threshold_pct: `2.0`
- min_pairs: `3`
- max_queue_spread_pct: `8.0`
- min_repeats_per_queue_for_spread: `2`

## Per Queue Directory

| dir | repeat | gain_all_pct | gain_steady_pct | off_n | on_n | off_cv_pct | on_cv_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| logs/awave-cw2-boundary-ab-a100-2g-uq-r1 | 1 | 5.23 | 4.49 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-a100-2g-uq-r2 | 2 | 4.16 | 4.63 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-a100-2g-uq-r3 | 3 | 4.33 | 4.36 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-a100-2g-uq-r4 | 4 | 4.36 | 4.46 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-a100-2g-uq-r5 | 5 | 1.90 | 4.60 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-a100-4g-uq-r1 | 1 | 0.26 | 5.91 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-a100-4g-uq-r2 | 2 | -4.78 | 3.98 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-a100-4g-uq-r3 | 3 | 5.16 | 6.97 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-2g-pq-r1 | 1 | 76.35 | 115.76 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-2g-uq-r1 | 1 | 9.25 | 10.21 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-2g-uq-r2 | 2 | 9.72 | 10.30 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-2g-uq-r3 | 3 | 8.41 | 10.29 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-2g-uq-r4 | 4 | 9.05 | 10.32 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-2g-uq-r5 | 5 | -1.39 | -1.61 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-8g-pq-r1 | 1 | 5.53 | 5.86 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-8g-pq-r2 | 2 | 5.91 | 6.11 | 1 | 1 | nan | nan |
| logs/awave-cw2-boundary-ab-h100-8g-pq-r3 | 3 | 4.24 | 4.59 | 1 | 1 | nan | nan |

## Group Decision

| gpu | gpus | gain_steady_pct | pairs | queue_spread_pct | spread_reliable | status | recommended_action |
|---|---:|---:|---:|---:|---|---|---|
| a100 | 2 | 4.51 | 5 | 0.00 | no | promote | enable-boundary-and-continue-next-optimizations |
| a100 | 4 | 5.60 | 3 | 0.00 | no | promote | enable-boundary-for-this-gpu-count |
| h100 | 2 | 22.18 | 6 | 107.85 | no | promote | enable-boundary-and-continue-next-optimizations |
| h100 | 8 | 5.52 | 3 | 0.00 | no | promote | enable-boundary-for-this-gpu-count |

