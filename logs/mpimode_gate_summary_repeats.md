# MPI Mode Gate Summary

## Thresholds

- host_promote_threshold_pct: `5.0`
- device_promote_threshold_pct: `-3.0`
- min_pairs: `2`
- max_queue_spread_pct: `10.0`

## Per Queue Directory

| dir | repeat | host_gain_all_pct | host_gain_steady_pct | device_n | host_n | device_cv_pct | host_cv_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| logs/awave-cw2-mpimode-ab-a100-2g-pq-r1 | 1 | -49.54 | -52.08 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-a100-2g-pq-r2 | 2 | -2.41 | -5.36 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-a100-2g-uq-r1 | 1 | -0.05 | -2.89 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-a100-2g-uq-r2 | 2 | 0.19 | -2.45 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-a100-2g-uq-r3 | 3 | -0.25 | -2.76 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-a100-2g-uq-r4 | 4 | 0.11 | -2.59 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-a100-2g-uq-r5 | 5 | -0.05 | -2.72 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-h100-2g-pq-r1 | 1 | 5.57 | -3.93 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-h100-2g-pq-r2 | 2 | 5.52 | -3.93 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-h100-2g-pq-r3 | 3 | 7.03 | -2.55 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-h100-2g-uq-r1 | 1 | 6.08 | -2.49 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-h100-2g-uq-r2 | 2 | 6.40 | -2.55 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpimode-ab-h100-2g-uq-r3 | 3 | 4.59 | -3.92 | 1 | 1 | nan | nan |

## Group Decision

| gpu | gpus | host_gain_steady_pct | pairs | queue_spread_pct | status | recommended_action |
|---|---:|---:|---:|---:|---|---|
| a100 | 2 | -10.13 | 7 | 26.03 | provisional-device | keep-device-mode-for-next-stage |
| h100 | 2 | -3.23 | 6 | 0.49 | device | keep-device-mode-for-next-stage |

