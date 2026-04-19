# Single-Factor A/B Summary

| case | rep | steady_sups | all_sups | samples | file |
|---|---:|---:|---:|---:|---|
| base | 1 | 69420000000.00 | 67005025000.00 | 4 | logs/awave-cw2-factor-ab-a100-2g-uq/base_r1.json |
| base | 2 | 69533500000.00 | 67042050000.00 | 4 | logs/awave-cw2-factor-ab-a100-2g-uq/base_r2.json |
| damp_off | 1 | 69546233333.33 | 67020125000.00 | 4 | logs/awave-cw2-factor-ab-a100-2g-uq/damp_off_r1.json |
| damp_off | 2 | 69537466666.67 | 66968475000.00 | 4 | logs/awave-cw2-factor-ab-a100-2g-uq/damp_off_r2.json |
| zpad_off | 1 | 69699600000.00 | 67193675000.00 | 4 | logs/awave-cw2-factor-ab-a100-2g-uq/zpad_off_r1.json |
| zpad_off | 2 | 69700700000.00 | 67176900000.00 | 4 | logs/awave-cw2-factor-ab-a100-2g-uq/zpad_off_r2.json |

## Aggregate

- base steady_mean=69476750000.00 (runs=2)
- damp_off steady_mean=69541850000.00 (runs=2)
- zpad_off steady_mean=69700150000.00 (runs=2)
- damp_branchless_on_vs_off=-0.09%
- z_padding_on_vs_off=-0.32%
