# Table 2: Multi-GPU Communication Fraction (Nsight Systems)

| GPU | GPU数 | 问题规模 | interior kernel (ms) | MPI_Waitall avg (ms) | MPI_Waitall total (ms) | 通信占比 (%) | overlap proxy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A100 | 4 | 1000³ | 6.770 | 0.703 | 56.253 | 16.11 | 79.23 |
