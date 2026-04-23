# Table 2: Multi-GPU Communication Fraction (Nsight Systems)

| GPU | GPU数 | 问题规模 | interior kernel (ms) | MPI_Waitall avg (ms) | MPI_Waitall total (ms) | 通信占比 (%) | overlap proxy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H100 | 2 | 1000³ | 7.166 | 0.470 | 18.806 | 10.90 | 86.88 |
