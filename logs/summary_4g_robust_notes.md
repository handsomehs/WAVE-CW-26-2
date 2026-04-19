# 4-GPU Robust Statistics + Nsight Start Status

Generated after running:

- `tools/aggregate_stats.py`
- `jobs/run-cw2-profile-nsys-a100.yml`
- `jobs/run-cw2-profile-ncu-a100.yml`

## Robust statistics (all-sample vs steady-state)

- H100/A100 ratio mean (all-sample): 1.632
- H100/A100 ratio mean (steady-state): 1.880
- all-sample ratio range: 1.446 (size2048) to 1.865 (strong_np1)
- steady-state ratio range: 1.816 (weak_np4) to 1.938 (size1536)

## Scaling efficiency (all-sample vs steady-state)

- A100 strong np4 efficiency: 95.37% -> 106.02% (steady)
- A100 weak np4 efficiency: 73.09% -> 95.99% (steady)
- H100 strong np4 efficiency: 78.91% -> 109.97% (steady)
- H100 weak np4 efficiency: 66.48% -> 92.14% (steady)

Interpretation: cold-start samples materially depress all-sample scaling and cross-GPU ratios.

## Nsight start status

- Nsight Systems job completed and produced:
  - `logs/profiles/nsys/cw2_nsys_a100_np1.nsys-rep`
  - `logs/profiles/nsys/cw2_nsys_a100_np1.json`
- Nsight Compute job completed but returned:
  - `No metrics to collect found in sections.`
  - `No kernels were profiled.`

So NCU was started and executed, but this container/runtime currently does not emit kernel-level NCU artifacts.
