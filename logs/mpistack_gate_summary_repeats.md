# MPI Stack Gate Summary

## Thresholds

- promote_ob1_threshold_pct: `3.0`
- keep_default_threshold_pct: `-2.0`
- neutral_default_band_pct: `0.5`
- min_pairs: `2`
- max_queue_spread_pct: `10.0`

## Per Queue Directory

| dir | repeat | ob1_gain_all_pct | ob1_gain_steady_pct | default_n | ob1_n | default_cv_pct | ob1_cv_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| logs/awave-cw2-mpistack-ab-a100-2g-uq-r1 | 1 | 0.76 | -0.12 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpistack-ab-a100-2g-uq-r2 | 2 | 1.45 | -0.15 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpistack-ab-a100-2g-uq-r3 | 3 | 1.99 | 0.02 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpistack-ab-h100-2g-uq-r1 | 1 | -46.03 | -51.19 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpistack-ab-h100-2g-uq-r2 | 2 | -42.61 | -44.03 | 1 | 1 | nan | nan |
| logs/awave-cw2-mpistack-ab-h100-2g-uq-r3 | 3 | -25.41 | -29.55 | 1 | 1 | nan | nan |

## Group Decision

| gpu | gpus | ob1_gain_steady_pct | pairs | queue_spread_pct | status | recommended_action |
|---|---:|---:|---:|---:|---|---|
| a100 | 2 | -0.08 | 3 | 0.00 | default-neutral | keep-default-stack-for-next-stage |
| h100 | 2 | -41.45 | 3 | 0.00 | default | keep-default-stack-for-next-stage |

