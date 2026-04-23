# Table 2: Multi-GPU Communication Fraction (Nsight Systems)

| GPU | GPU数 | 问题规模 | interior kernel (ms) | MPI_Waitall avg (ms) | MPI_Waitall total (ms) | 通信占比 (%) | overlap proxy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H100 | 4 | 1000³ | 3.513 | 1.282 | 102.547 | 40.37 | 27.03 |
