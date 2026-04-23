# CW2 报告表格素材与正文草稿（按 plan.md）

更新时间：2026-04-21（UTC）

## 0. 报告统一口径（建议直接写进报告方法部分）
- 编译口径：`AWAVE_MODE=CUDA`，`CMAKE_BUILD_TYPE=Release`。
- 通信口径：默认 `AWAVE_MPI_MODE=device`，并禁用 `hcoll/ucc`（见作业模板）。
- 性能口径：强缩放效率优先用 `E_n = T1/(n*Tn)`；`MPI_Waitall` 只作为等待行为证据，不作为真实通信占比。
- 稳态口径：优先采用 best steady-state（对明显首样本异常的 case，取后续稳态样本）。
- 特殊口径（2500^3）：`H100 NP8 size2500` 的可运行测量来自“skip_cpu=1 场景下去除不必要额外内存拷贝”的临时路径（`main.cpp` + 少量 `wave_cuda.cu`），用于拿到 device 路径速度；除 2500^3 外其余结果仍按常规版本口径描述。

---

## 表 1：实验环境（可直接放报告）

| 项目 | 值 | 证据/来源 |
|---|---|---|
| GPU 型号 | A100 80GB（`NVIDIA-A100-SXM4-80GB`），H100 80GB（`NVIDIA-H100-80GB-HBM3`） | 各 job YAML `nodeSelector` |
| HBM 理论带宽 | A100: 2.04 TB/s；H100: 3.35 TB/s | NVIDIA datasheet（报告中引用） |
| L2 Cache | A100: 40 MB；H100: 50 MB | NVIDIA datasheet（报告中引用） |
| FP64 峰值（non-sparse） | A100: 9.7 TFLOP/s；H100: 33.5 TFLOP/s | NVIDIA datasheet（报告中引用） |
| 互联 | NVLink（A100 600 GB/s；H100 900 GB/s 双向） | NVIDIA datasheet（报告中引用） |
| CUDA 工具链 | CUDA 13.0 (`nvcc 13.0.48`) | `nvcc --version` |
| MPI | Open MPI `4.1.9a1` | `mpirun --version` |
| CMake | 3.28.3 | `cmake --version` |
| C++ 标准 | C++20 | `src/CMakeLists.txt` |
| 编译器（CUDA 模式） | `nvc++` + `nvcc` | `src/CMakeLists.txt` |
| 编译模式 | Release | 所有 job 模板 `-DCMAKE_BUILD_TYPE=Release` |
| 队列策略 | UQ + PQ 并行提交，保留先完成结果 | 实验执行策略 |

注：`image: $container` 为集群模板变量，若报告要写容器 digest，需要在运行环境补一次 `kubectl describe pod` 或作业日志记录。

---

## 表 2：优化 A/B 决策总结（开发阶段证据，可用于报告表2）

| 优化项 | 理论依据 | A100 实测 | H100 实测 | 决策 | 数据来源 |
|---|---|---:|---:|---|---|
| `AWAVE_CUDA_BOUNDARY_SPLIT=1` | 边界专门 kernel，减少 interior 早退浪费 | +4.51%（2G） | +22.18%（2G） | 开启 | `logs/boundary_gate_summary_repeats.md` |
| `AWAVE_CUDA_BLOCK=6`（32,4,4） | x 方向复用 + warp 友好 | +6.03%（vs blk0） | +0.04%（vs blk0） | 选为默认 | `logs/block_tune_*_summary.md` |
| `AWAVE_CUDA_TILE=1`（shared-memory tile） | 期望减少全局访存 | -2.32%（tile1 vs tile0） | -4.59%（tile1 vs tile0） | 不采用 | `logs/block_tune_*_summary.md` |
| `AWAVE_MPI_MODE=host` vs `device` | Host staging 作为 fallback | host 慢 -10.13%（2G） | host 慢 -3.23%（2G） | 保持 `device` | `logs/mpimode_gate_summary_repeats.md` |
| `ob1` stack vs default | 更换 MPI 传输栈 | -0.08%（2G） | -41.45%（2G） | 保持 default | `logs/mpistack_gate_summary_repeats.md` |
| `MPI_Waitsome` | 期望减少等待 | -0.03%（2G） | -0.56%（2G） | 保持 off | `logs/waitsome_gate_summary_repeats.md` |
| overlap 开关（2G gate） | 计算/通信重叠 | +1.06% | +1.27% | 2G 证据偏弱，默认保留 overlap 路线 | `logs/overlap_gate_summary_repeats.md` |
| damp branchless | 减少分支发散 | -0.085%（均值） | -0.002%（均值） | 不采用 | `logs/factor_ab_overall_summary.md` |
| z-padding | 对齐访问 | -0.367%（均值） | -0.513%（均值） | 不采用 | `logs/factor_ab_overall_summary.md` |

---

## 表 3：单卡 Roofline 素材（带宽受限推导）

口径：`SU/s` 来自 JSON/summary，`B/site = B_HBM_peak / SU/s`，`opt利用率 = SU/s*40 / B_HBM_peak`。

| GPU | size | SU/s | 反推 B/site | HBM peak | 带宽利用率（opt） | 数据来源 |
|---|---:|---:|---:|---:|---:|---|
| H100 | 256^3 | 8.259e10 | 40.56 | 3.35 TB/s | 98.62% | `logs/profiles/nsys_roofline_h100_pq/summary.json` |
| H100 | 512^3 | 7.673e10 | 43.66 | 3.35 TB/s | 91.62% | 同上 |
| H100 | 1000^3 | 7.677e10 | 43.64 | 3.35 TB/s | 91.66% | 同上 |
| A100 | 256^3（单卡） | 3.460e10 | 58.96 | 2.04 TB/s | 67.84% | `logs/profiles/nsys_matrix_a100_uq/nsys_a100_np1_size256.json` |
| A100 | 512^3（单卡） | 3.439e10 | 59.32 | 2.04 TB/s | 67.44% | `logs/profiles/nsys_matrix_a100_uq/nsys_a100_np1_size512.json` |
| A100 | 1000^3（单卡） | 3.561e10 | 57.28 | 2.04 TB/s | 69.83% | `logs/profiles/nsys_matrix_a100_uq/nsys_a100_np1_size1000.json` |
| A100 | 768^3（单卡） | 3.514e10 | 58.05 | 2.04 TB/s | 68.90% | `logs/awave-cw2-perf-a100-4g/strong_np1.json` |

注：A100 单卡 256/512/1000 已补齐；优先采用 UQ 这组更稳定数据作为报告口径。

---

## 表 4：Strong scaling 素材（主指标）

口径：
- `SU/s` 采用 best steady-state（A100 NP8 的首样本明显慢，使用稳态样本）。
- `效率 = SU/s_n / (n * SU/s_1)`。
- 同时给出 `额外开销 = 1 - 效率`。

| GPU 数 | A100 SU/s | A100 效率 | A100 额外开销 | H100 SU/s | H100 效率 | H100 额外开销 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.514e10 | 100.00% | 0.00% | 6.669e10 | 100.00% | 0.00% |
| 2 | 6.963e10 | 99.08% | 0.92% | 1.316e11 | 98.64% | 1.36% |
| 4 | 1.349e11 | 95.96% | 4.04% | 2.551e11 | 95.61% | 4.39% |
| 8 | 2.648e11 | 94.19% | 5.81% | 4.885e11 | 91.56% | 8.44% |

补充（你强调的口径，建议写正文）：
- 按 H100 4 卡 mean-time：`T1=0.0679244s`，`T4=0.01778585s`，`E4=T1/(4*T4)=95.48%`，额外开销 `4.52%`。
- 该结果不支持“通信占比很高”的结论，`MPI_Waitall` 只能作为等待行为证据。

数据来源：
- A100：`logs/awave-cw2-perf-a100-4g/strong_np{1,2,4}.json` + `logs/awave-cw2-perf-a100-8g/strong_np8.json`
- H100：`logs/awave-cw2-perf-h100-4g-pq/strong_np{1,2,4}.json` + `logs/awave-cw2-perf-h100-8g-pq/strong_np8.json`

---

## 表 5：通信分析（Nsight Systems）

| GPU | GPU数 | 问题规模 | interior kernel (ms) | MPI_Waitall avg (ms) | MPI_Waitall total (ms) | 通信占比 (%) | overlap proxy (%) | 数据源 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A100 | 2 | 1000^3 | 13.564 | 0.242 | 9.683 | 3.21 | 96.43 | `nsys_scale_a100_np2_uq_r2/summary.json` |
| A100 | 4 | 1000^3 | 6.772 | 0.563 | 45.068 | 13.34 | 83.36 | `nsys_scale_a100_np4_uq/summary.json` |
| H100 | 2 | 1000^3 | 7.166 | 0.470 | 18.806 | 10.90 | 86.88 | `nsys_scale_h100_np2_uq_r2/summary.json` |
| H100 | 4 | 1000^3 | 3.513 | 1.282 | 102.547 | 40.37 | 27.03 | `nsys_scale_h100_np4_uq_r5/summary.json` |
| H100 | 8 | 1000^3 | 1.753 | 2.889 | 462.277 | 75.29 | 0.00 | `nsys_scale_h100_np8_pq/summary.json` |

表注建议：
- `Waitall` 指标是 wall-clock 等待行为，不等于不可隐藏通信开销。
- 与表4联合解释：即使 H100-4 在 NSYS 中看到较高等待，强缩放仍达约 95%，说明大量通信被流水线隐藏。

---

## 表 6：结果章节可用的 size 扫描素材（用于图1/结果段）

### 6.1 H100（NP8，PQ）

| size | SU/s | ndiff | 说明 |
|---:|---:|---:|---|
| 256 | 7.954e10 | 0 | 常规口径 |
| 512 | 3.600e11 | 0 | 常规口径 |
| 768 | 4.567e11 | 0 | 常规口径 |
| 1000 | 4.830e11 | 0 | 常规口径 |
| 1500 | 5.123e11 | 0 | 常规口径 |
| 2000 | 5.273e11 | 0 | 常规口径 |
| 2500 | 4.121e11（best of c3） | 0 | **special path**（skip_cpu=1 内存拷贝优化） |

数据文件：
- `logs/profiles/nsys_matrix_h100_pq/nsys_h100_np8_size{256,512,768,1000,1500,2000}.json`
- `logs/profiles/nsys_matrix_h100_pq/nsys_h100_np8_size2500_c3.json`

### 6.2 A100（当前可用代表点）

| NP | size | SU/s | ndiff | 数据源 |
|---:|---:|---:|---:|---|
| 4 | 256 | 7.421e10 | 0 | `nsys_matrix_a100_uq/nsys_a100_np4_size256.json` |
| 4 | 512 | 1.218e11 | 0 | `...size512.json` |
| 4 | 768 | 1.320e11 | 0 | `...size768.json` |
| 4 | 1000 | 1.316e11 | 0 | `...size1000.json` |
| 4 | 1500 | 1.374e11 | 0 | `...size1500.json` |

注：A100 的 NP8 大尺寸（尤其 1500/2000/2500）目前仍不完整，报告中可标 N/A 并说明调度/超时约束。

### 6.3 A100 256/512/1000（补充说明：你已完成）

| NP | size | SU/s | ndiff | 数据源 |
|---:|---:|---:|---:|---|
| 1 | 256 | 3.460e10 | 0 | `nsys_matrix_a100_uq/nsys_a100_np1_size256.json` |
| 1 | 512 | 3.439e10 | 0 | `nsys_matrix_a100_uq/nsys_a100_np1_size512.json` |
| 1 | 1000 | 3.561e10 | 0 | `nsys_matrix_a100_uq/nsys_a100_np1_size1000.json` |
| 2 | 256 | 5.026e10 | 0 | `nsys_matrix_a100_uq/nsys_a100_np2_size256.json` |
| 2 | 512 | 6.449e10 | 0 | `nsys_matrix_a100_uq/nsys_a100_np2_size512.json` |
| 2 | 1000 | 6.690e10 | 0 | `nsys_matrix_a100_uq/nsys_a100_np2_size1000.json` |
| 4 | 256 | 7.421e10 | 0 | `nsys_matrix_a100_uq/nsys_a100_np4_size256.json` |
| 4 | 512 | 1.218e11 | 0 | `nsys_matrix_a100_uq/nsys_a100_np4_size512.json` |
| 4 | 1000 | 1.316e11 | 0 | `nsys_matrix_a100_uq/nsys_a100_np4_size1000.json` |
| 8 | 256 | 1.390e9 | 0 | `nsys_matrix_a100_pq/nsys_a100_np8_size256.json` |
| 8 | 512 | 7.053e9 | 0 | `nsys_matrix_a100_pq/nsys_a100_np8_size512.json` |
| 8 | 1000 | 3.767e10 | 0 | `nsys_matrix_a100_pq/nsys_a100_np8_size1000.json` |

---

## 图素材映射（按 plan.md 图清单）

- 图1（SU/s vs size）：
  - H100：优先用“表6.1 NP8”曲线。
  - A100：用“表6.2 NP4”曲线，图注注明“NP4 data shown; NP8 large-size incomplete”。
- 图2（Strong scaling efficiency）：用“表4”。
- 图3（B/site vs size）：用“表3”中 H100 的 256/512/1000，以及 A100 的 256/512/1000（768 可作补充点）。

---

## 可直接放入报告的正文内容（草稿）

### Section 1（Design and process）可用段落

> We implemented the multi-GPU version in CUDA with MPI and used a dual-stream pipeline (`compute_stream` + `halo_stream`) to overlap interior computation with halo packing/communication. All production runs were compiled in Release mode with CUDA 13.0 and OpenMPI 4.1.9a1.

> Optimization decisions were made by controlled A/B gates. `boundary_split` is consistently beneficial (+4.51% on A100-2G and +22.18% on H100-2G), so it was enabled. `AWAVE_CUDA_BLOCK=6` (32,4,4) was selected as the default launch shape. In contrast, shared-memory tiling, branchless damping, z-padding, and `MPI_Waitsome` did not provide stable gains and were not adopted in the final path.

> Roofline-style analysis indicates a bandwidth-dominated regime. For H100 single-GPU runs, inferred bytes/site are around 40.6-43.7 B/site across 256^3-1000^3, consistent with significant cache reuse relative to a naive 88 B/site stencil traffic model.

### Section 2（Results）可用段落

> Strong scaling is the main indicator of communication impact. Using best steady-state throughput, H100 reaches 95.61% efficiency at 4 GPUs and 91.56% at 8 GPUs; A100 reaches 95.96% at 4 GPUs and 94.19% at 8 GPUs.

> For H100, using mean-time scaling at 4 GPUs gives `E4 = 95.48%`, i.e. only `4.52%` extra overhead relative to ideal linear scaling. Therefore, NSYS `MPI_Waitall` time should be interpreted as waiting behavior evidence rather than as a strict communication-cost ratio.

> We successfully measured H100 NP8 at 2500^3 in device mode (`~4.12e11 SU/s`, best run). This point was obtained with a targeted memory-path adjustment for `skip_cpu=1` to remove unnecessary extra copies; this special path was only used for the 2500^3 measurement.

### Section 3（Further work）可用段落

> Next steps are (1) complete A100 NP8 large-size points (1500/2000/2500) under a unified runtime budget, (2) refresh NSYS communication traces on the exact final submission code path for strict version consistency, and (3) evaluate multi-node scaling with explicit NVLink/IB bandwidth-latency modeling.

---

## 当前缺口（报告里建议统一写 N/A + 原因）
- A100 NP8 的 1500/2000/2500 结果未形成稳定可用集合。
- 若最终提交代码与某些旧 NSYS trace 版本不一致，需在图表注释里显式声明版本差异。
