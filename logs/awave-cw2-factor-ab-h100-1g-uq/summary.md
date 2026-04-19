# Single-Factor A/B Summary

| case | rep | steady_sups | all_sups | samples | file |
|---|---:|---:|---:|---:|---|
| base | 1 | 66309000000.00 | 66233975000.00 | 4 | logs/awave-cw2-factor-ab-h100-1g-uq/base_r1.json |
| base | 2 | 66298900000.00 | 66227200000.00 | 4 | logs/awave-cw2-factor-ab-h100-1g-uq/base_r2.json |
| damp_off | 1 | 66302500000.00 | 66236050000.00 | 4 | logs/awave-cw2-factor-ab-h100-1g-uq/damp_off_r1.json |
| damp_off | 2 | 66298866666.67 | 66231350000.00 | 4 | logs/awave-cw2-factor-ab-h100-1g-uq/damp_off_r2.json |
| zpad_off | 1 | 66614600000.00 | 66553900000.00 | 4 | logs/awave-cw2-factor-ab-h100-1g-uq/zpad_off_r1.json |
| zpad_off | 2 | 66605766666.67 | 66533400000.00 | 4 | logs/awave-cw2-factor-ab-h100-1g-uq/zpad_off_r2.json |

## Aggregate

- base steady_mean=66303950000.00 (runs=2)
- damp_off steady_mean=66300683333.33 (runs=2)
- zpad_off steady_mean=66610183333.33 (runs=2)
- damp_branchless_on_vs_off=0.00%
- z_padding_on_vs_off=-0.46%
