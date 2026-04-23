# Table 2: Multi-GPU Communication Fraction (Nsight Systems)

| GPU | GPU数 | 问题规模 | interior kernel (ms) | MPI_Waitall avg (ms) | MPI_Waitall total (ms) | 通信占比 (%) | overlap proxy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A100 | 4 | 1000³ | 6.772 | 0.563 | 45.068 | 13.34 | 83.36 |
