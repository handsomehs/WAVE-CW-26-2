# Table 2: Multi-GPU Communication Fraction (Nsight Systems)

| GPU | GPU数 | 问题规模 | interior kernel (ms) | MPI_Waitall avg (ms) | MPI_Waitall total (ms) | 通信占比 (%) | overlap proxy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A100 | 2 | 1000³ | 13.564 | 0.242 | 9.683 | 3.21 | 96.43 |
