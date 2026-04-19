# Factor A/B Overall Summary

| case | damp_branchless_on_vs_off (%) | z_padding_on_vs_off (%) | base steady sups |
|---|---:|---:|---:|
| A100-1G | -0.0769 | -0.4136 | 35037416666.67 |
| A100-2G | -0.0936 | -0.3205 | 69476750000.00 |
| H100-1G | 0.0049 | -0.4597 | 66303950000.00 |
| H100-2G | -0.0097 | -0.5654 | 131216166666.67 |

- simple_mean damp_branchless_on_vs_off = -0.0438%
- simple_mean z_padding_on_vs_off = -0.4398%
- weighted_mean damp_branchless_on_vs_off = -0.0336% (weight=base steady sups)
- weighted_mean z_padding_on_vs_off = -0.4683% (weight=base steady sups)

- Decision: default `AWAVE_CUDA_DAMP_BRANCHLESS=0` (branched), default `AWAVE_CUDA_ZPAD=0` (no z padding).