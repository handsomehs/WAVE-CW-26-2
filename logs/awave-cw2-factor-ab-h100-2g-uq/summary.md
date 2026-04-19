# Single-Factor A/B Summary

| case | rep | steady_sups | all_sups | samples | file |
|---|---:|---:|---:|---:|---|
| base | 1 | 131239666666.67 | 118086950000.00 | 4 | logs/awave-cw2-factor-ab-h100-2g-uq/base_r1.json |
| base | 2 | 131192666666.67 | 118533175000.00 | 4 | logs/awave-cw2-factor-ab-h100-2g-uq/base_r2.json |
| damp_off | 1 | 131263000000.00 | 118293525000.00 | 4 | logs/awave-cw2-factor-ab-h100-2g-uq/damp_off_r1.json |
| damp_off | 2 | 131194666666.67 | 117868175000.00 | 4 | logs/awave-cw2-factor-ab-h100-2g-uq/damp_off_r2.json |
| zpad_off | 1 | 131915333333.33 | 118844200000.00 | 4 | logs/awave-cw2-factor-ab-h100-2g-uq/zpad_off_r1.json |
| zpad_off | 2 | 132009333333.33 | 118398175000.00 | 4 | logs/awave-cw2-factor-ab-h100-2g-uq/zpad_off_r2.json |

## Aggregate

- base steady_mean=131216166666.67 (runs=2)
- damp_off steady_mean=131228833333.33 (runs=2)
- zpad_off steady_mean=131962333333.33 (runs=2)
- damp_branchless_on_vs_off=-0.01%
- z_padding_on_vs_off=-0.57%
