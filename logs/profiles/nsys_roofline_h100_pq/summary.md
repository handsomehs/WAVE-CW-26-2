# H100 Roofline Proxy (Nsight Systems + Manual Derivation)

## 1A) Single-GPU interior kernel manual bandwidth proxy

| Problem Size | interior kernel time (ms) | SU/s | inferred B/site (B) | naive BW utilization (%) | optimistic BW utilization (%) |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.198 | 82594196430.62 | 40.56 | 216.96 | 98.62 |
| 512 | 1.729 | 76731927621.75 | 43.66 | 201.56 | 91.62 |
| 1000 | 12.948 | 76766855193.11 | 43.64 | 201.66 | 91.66 |

Criterion: if naive utilization > 100%, strong cache reuse is required to explain measured throughput.

## 1B) Boundary vs interior kernel time share (size=1000)

| Kernel | Calls | Avg(ms) | Time Share (%) |
|---|---:|---:|---:|
| step_kernel_interior | 22 | 12.948 | 96.23 |
| step_kernel_boundary_x_faces | 22 | 0.037 | 0.27 |
| step_kernel_boundary_y_faces | 22 | 0.036 | 0.27 |
| step_kernel_boundary_z_faces | 22 | 0.434 | 3.23 |

## 1C) Multi-scale trend (256/512/1000/2000)

- Track how `SU/s` scales with problem size and whether inferred `B/site` converges.
- If inferred `B/site` approaches the optimistic assumption, cache reuse is stronger.
- If naive utilization remains far above 100% at large sizes, this supports non-pure-HBM behavior.

