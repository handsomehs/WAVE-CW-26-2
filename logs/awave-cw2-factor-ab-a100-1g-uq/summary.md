# Single-Factor A/B Summary

| case | rep | steady_sups | all_sups | samples | file |
|---|---:|---:|---:|---:|---|
| base | 1 | 35010066666.67 | 35006750000.00 | 4 | logs/awave-cw2-factor-ab-a100-1g-uq/base_r1.json |
| base | 2 | 35064766666.67 | 35044475000.00 | 4 | logs/awave-cw2-factor-ab-a100-1g-uq/base_r2.json |
| damp_off | 1 | 35064000000.00 | 35050500000.00 | 4 | logs/awave-cw2-factor-ab-a100-1g-uq/damp_off_r1.json |
| damp_off | 2 | 35064766666.67 | 35046025000.00 | 4 | logs/awave-cw2-factor-ab-a100-1g-uq/damp_off_r2.json |
| zpad_off | 1 | 35194333333.33 | 35180925000.00 | 4 | logs/awave-cw2-factor-ab-a100-1g-uq/zpad_off_r1.json |
| zpad_off | 2 | 35171566666.67 | 35155475000.00 | 4 | logs/awave-cw2-factor-ab-a100-1g-uq/zpad_off_r2.json |

## Aggregate

- base steady_mean=35037416666.67 (runs=2)
- damp_off steady_mean=35064383333.33 (runs=2)
- zpad_off steady_mean=35182950000.00 (runs=2)
- damp_branchless_on_vs_off=-0.08%
- z_padding_on_vs_off=-0.41%
