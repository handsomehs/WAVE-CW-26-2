# MPI Stack Gate Summary

## Thresholds

- promote_ob1_threshold_pct: `3.0`
- keep_default_threshold_pct: `-2.0`
- min_pairs: `2`
- max_queue_spread_pct: `10.0`

## Per Queue Directory

| dir | repeat | ob1_gain_all_pct | ob1_gain_steady_pct | default_n | ob1_n | default_cv_pct | ob1_cv_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| logs/awave-cw2-mpistack-ab-h100-2g-uq | - | -4.73 | -6.03 | 1 | 1 | nan | nan |

## Group Decision

| gpu | gpus | ob1_gain_steady_pct | pairs | queue_spread_pct | status | recommended_action |
|---|---:|---:|---:|---:|---|---|
| h100 | 2 | -6.03 | 1 | 0.00 | provisional-default | keep-default-stack-for-next-stage |

