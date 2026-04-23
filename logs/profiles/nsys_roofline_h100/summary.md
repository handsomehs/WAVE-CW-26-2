# H100 Roofline Proxy (Nsight Systems + Manual Derivation)

## 1A) Single-GPU interior kernel manual bandwidth proxy

| Problem Size | interior kernel time (ms) | SU/s | inferred B/site (B) | naive BW utilization (%) | optimistic BW utilization (%) |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.218 | 75251345031.60 | 44.52 | 197.68 | 89.85 |
| 512 | 1.901 | 69769852739.65 | 48.02 | 183.28 | 83.31 |
| 1000 | 14.242 | 69795287799.91 | 48.00 | 183.34 | 83.34 |

Criterion: if naive utilization > 100%, strong cache reuse is required to explain measured throughput.

## 1B) Boundary vs interior kernel time share (size=1000)

| Kernel | Calls | Avg(ms) | Time Share (%) |
|---|---:|---:|---:|
| step_kernel_interior | 20 | 14.242 | 96.25 |
| step_kernel_boundary_x_faces | 20 | 0.040 | 0.27 |
| step_kernel_boundary_y_faces | 20 | 0.040 | 0.27 |
| step_kernel_boundary_z_faces | 20 | 0.475 | 3.21 |

## 1C) Multi-scale trend (256/512/1000/2000)

- Track how `SU/s` scales with problem size and whether inferred `B/site` converges.
- If inferred `B/site` approaches the optimistic assumption, cache reuse is stronger.
- If naive utilization remains far above 100% at large sizes, this supports non-pure-HBM behavior.

