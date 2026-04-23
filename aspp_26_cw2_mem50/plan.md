# ASPP Coursework 2 — CUDA 最终执行计划（aspp_26_cw2）

> 目标：在单节点（最多 8 GPU）上实现正确、高性能且“解释清楚”的 CUDA 版本，并按课程要求提交 **仅一个源码文件** `src/wave_cuda.cu` + **≤5 页** PDF 报告。
>
> 本计划已按仓库 `ASPP_CW2/aspp_26_cw2` 的真实结构/脚本校准：
> - 构建：`cmake -S src -B build-cuda -DAWAVE_MODE=CUDA ...`
> - 可执行文件：`./build-cuda/awave`
> - 打包脚本：`./make_submission.sh -e ... -r ... -m CUDA`
> - 集群批量实验：优先复用 `jobs/` 模板，产物写入 `logs/`

## 0. 约束、交付物与通过标准（先写清楚，避免返工）

### 0.1 硬性约束
- **提交代码只能改 `src/wave_cuda.cu`**（其余文件保持原样；用 `git status`/`git diff` 自检）。
- `run()` 内必须完成所有 GPU 工作与通信（作业结束前不能有悬挂异步任务）。
- 结果必须正确：默认会跑 CPU reference 并比较；**`ndiff=0` 才能计入性能**。

### 0.2 交付物
- 代码：`src/wave_cuda.cu`
- 报告：PDF（≤5 页，字号 ≥10pt；无需 introduction/conclusion）
- 提交包：`aspp-cw2-BXXXXXX.tar.gz`，解压结构：
	- `BXXXXXX/wave_cuda.cu`
	- `BXXXXXX/BXXXXXX.pdf`（脚本会把 report 重命名为 examno.pdf）

### 0.3 最终验收口径
- Correctness：计划第 2 步所有 case 均 `ndiff=0`。
- Performance：A100 与 H100 两种节点上至少覆盖 **size 256 到最大可行规模**、GPU 数 1/2/4/8 的结果；课程要求建议覆盖到 2500，但本项目在单节点上会同时保留 **CPU simulation + CUDA simulation** 两份 host 端状态（`from_cpu_sim` clone），host 侧约 **96 B/point**（见第 3 步内存估算），因此在 512 GB 节点上 `2000^3` 与 `2500^3` 都很可能 OOM——报告必须写清“为什么最大 size 只能到 X”。
- Clarity：关键 kernel 与通信流水线必须有清晰注释（说明并行映射、为什么这样做、与 overlap 的依赖关系）。

## 0.4 本仓库的构建/运行/作业工具（统一口径）

### 0.4.1 构建（Release）
```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
cmake -S src -B build-cuda -DAWAVE_MODE=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

### 0.4.2 运行（二选一）

**A) 交互/单机测试（你在 GPU 节点/容器里时）**

命令示例统一使用：`mpirun -np <N> ./build-cuda/awave ...`。

**B) 集群批量跑（推荐，复用 jobs 模板）**

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
kgpu create -f jobs/run-cw2-correct-a100.yml
kgpu jls -n 'awave-cw2-correct-a100-*'
kgpu logs -j <job-name>
```

更多说明见 `jobs/JOBS.md`（包括：perf matrix、profiling queue fallback、稳态统计工具 `tools/aggregate_stats.py`）。

---

# 最终完整计划

## 第 1 步：代码清理（4–5 小时）

> 目标：把“已否决/无收益”的 A/B 分支从 `src/wave_cuda.cu` 移除，减少维护负担；同时把最终选择的并行策略写清楚（为 clarity 评分服务）。

### 1A. 删除否决项（固定最终实现路径）

删除/固定以下项（括号内为你已有的 A/B 结论，保留在报告的决策表中；**但代码里删掉开关与分支**）：

- `damp_branchless_enabled_from_env()` + `damped_update(..., bool branchless)` 分支与所有 kernel 的 `damp_branchless` 参数与路径（约 -0.08%）。
- `z_padding_enabled_from_env()` + `cudaMemcpy2D{Async}` 及相关 pitch/row 计算逻辑（约 -0.5%）。
- `padded_size(local_shape, u_stride_y)`：恢复原始签名 `padded_size(local_shape)`，不再把 stride 当入参（stride 固定 `nz+2`）。
- `boundary_split_enabled_from_env()`：固定使用 split 版边界更新（保留 `*_x_faces/y_faces/z_faces`），删除非 split 版 `step_kernel_boundary` 及其调用。
- `tile_enabled_from_env()` + `step_kernel_interior_tiled`：无收益，删除 tiled kernel 与开关。
- `block_mode_from_env()` + block shape 的 `switch`：固定最终 block 为 `(32,4,4)`，删除开关/分支。
- `__ldg` 显式只读加载：恢复普通访问（依赖 `const` + `__restrict__` 让编译器自行选择只读路径）。

> 清理原则：**所有“实验开关”要么变成最终常量，要么完全删除**；`AWAVE_MPI_MODE` 这类“功能性/鲁棒性”开关可保留（例如 CUDA-aware MPI 不稳定时切 host staging）。

### 1B. 简化后的关键函数签名（对照实现）

**`damped_update`：删除 branchless 参数**

```cpp
__device__ __forceinline__ double damped_update(
				double center, double prev, double value, double d, double dt) {
		if (d == 0.0) {
				return 2.0 * center - prev + value;
		}
		double inv_den = 1.0 / (1.0 + d * dt);
		double num = 1.0 - d * dt;
		value *= inv_den;
		return 2.0 * inv_den * center - num * inv_den * prev + value;
}
```

**`padded_size`：恢复原始签名**

```cpp
std::size_t padded_size(shape_t const& local_shape) {
		auto [nx, ny, nz] = local_shape;
		return (std::size_t)(nx + 2) * (std::size_t)(ny + 2) * (std::size_t)(nz + 2);
}
```

**构造函数中 stride 固定**

```cpp
u_stride_y = nz + 2;          // 删除 z_padding 分支
u_stride_x = (ny + 2) * u_stride_y;
```

**block 固定**

```cpp
dim3 block_ijk(32, 4, 4);     // 删除 switch
```

### 1C. 构造函数/同步路径：恢复普通 memcpy（去除 2D copy/pitch）

构造函数中删除所有 `cudaMemcpy2DAsync`，改回（示例）：

```cpp
CUDA_CHECK(cudaMemcpyAsync(d_prev, u.prev().data(),
		u_size * sizeof(double), cudaMemcpyHostToDevice, compute_stream));
// d_now, d_next 同理
```

`sync_host_fields()` 中同理改回 `cudaMemcpy`：

```cpp
CUDA_CHECK(cudaMemcpy(u.now().data(), d_now,
		u_size * sizeof(double), cudaMemcpyDeviceToHost));
if (copy_prev) {
		CUDA_CHECK(cudaMemcpy(u.prev().data(), d_prev,
				u_size * sizeof(double), cudaMemcpyDeviceToHost));
}
```

> 备注：前提是你已把 `u_stride_y` 固定为 `nz+2`，`u_size` 即完整 padding 后的线性大小。

### 1D. 为每个保留的函数补齐“并行策略”注释（clarity 关键）

老师反馈要点：
- “not so clear for the others”
- “we were looking for a discussion of the reasons why you do things”

因此要求：**每个保留 kernel / halo helper / `run()` 主循环**必须说明：
1) 线程/块如何映射到 (i,j,k) 或 face 点；
2) 为什么这样映射（合并访问、warp 组织、L2 复用、launch 规模等）；
3) 与双 stream overlap 的数据不冲突前提（disjoint region / event dependency）。

建议直接采用下列“注释模板风格”（按你最终代码调整细节）：

```cpp
/// Pack the boundary plane of d_now perpendicular to the x-axis into a
/// contiguous send buffer. Each thread handles one (j,k) point on the face.
/// Runs on halo_stream; reads only boundary rows i=1 or i=nx of d_now,
/// which are disjoint from the interior region computed on compute_stream.
__global__ void pack_face_x(...) { ... }

/// Update all interior grid points — those whose stencil neighbourhood
/// lies entirely within this rank's local data, requiring no halo input.
/// Mapped as k→threadIdx.x (coalesced), j→threadIdx.y, i→threadIdx.z.
/// Block shape (32,4,4) = 512 threads chosen for:
///   - 32 threads in k: one full warp accesses contiguous memory
///   - 4 layers in i: neighbouring x-planes share stencil reads via L2
/// Runs on compute_stream concurrently with pack+MPI on halo_stream.
__global__ void step_kernel_interior(...) { ... }

/// Update boundary points on the two x-perpendicular faces of the local
/// subdomain. Thread t maps linearly to (face, j, k), where face ∈ {0,1}
/// selects i=0 or i=nx-1. Only launched after halo data has been unpacked
/// into d_now. Each of the three face kernels covers non-overlapping point
/// sets, so they can run sequentially on halo_stream without conflicts.
__global__ void step_kernel_boundary_x_faces(...) { ... }
```

`run()` 主循环必须有“分阶段流水线解释”，建议直接用 8-phase 描述（按你最终代码保持一致）：

```cpp
/// Main time-stepping loop. Each step proceeds in 8 phases designed to
/// maximise overlap between GPU computation and MPI communication:
///
///   Phase 1 — Post MPI_Irecv for all faces (zero GPU dependency).
///   Phase 2 — Launch pack kernels on halo_stream (non-blocking).
///   Phase 3 — Launch interior kernel on compute_stream. This runs in
///             parallel with pack on halo_stream because they access
///             disjoint regions of d_now (interior vs boundary rows).
///   Phase 4 — CPU waits for pack to complete (cudaEventSynchronize),
///             then posts MPI_Isend. During the sync, the interior
///             kernel continues executing on the GPU.
///   Phase 5 — CPU waits for all MPI transfers to complete.
///   Phase 6 — Unpack received halo data into d_now ghost cells.
///   Phase 7 — Launch boundary face kernels on halo_stream.
///   Phase 8 — Record cross-stream event dependencies so that the next
///             iteration's pack waits for this step's interior, and the
///             next interior waits for this step's boundary. These are
///             device-side waits (cudaStreamWaitEvent) with zero CPU cost.
for (int i = 0; i < n; ++i) {
		...
}
```

### 1E. 清理完成后的自检（必须做）

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
git status
git diff --stat

# 只允许看到 src/wave_cuda.cu 的改动

cmake -S src -B build-cuda -DAWAVE_MODE=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

---

## 第 2 步：正确性回归验证（~2 小时）

> 目标：覆盖单卡/多卡、不同分解、非立方体、checkpoint/restart、小域（无 interior）等路径。**全部 ndiff=0 才进入第 3 步性能跑。**

### 2.1 回归原则
- 正确性回归 **不要用** `-skip_cpu 1`（否则没有 ndiff）。
- 每个 case 关注日志：`Number of differences detected = 0`（或 JSON 中 `ndiff=0`）。
- 若出现 ndiff：先在最小 case（256³、1 GPU）复现，再定位（边界/halo/stride 最常见）。

### 2.2 命令（假设可执行为 `./build-cuda/awave`）

```bash
# 单卡
mpirun -np 1 ./build-cuda/awave -mpi 1,1,1 -shape 256,256,256 -nsteps 20 -io 0

# 多卡多种分解
mpirun -np 2 ./build-cuda/awave -mpi 1,1,2 -shape 512,512,512 -nsteps 20 -io 0
mpirun -np 4 ./build-cuda/awave -mpi 1,2,2 -shape 1000,1000,1000 -nsteps 20 -io 0
mpirun -np 8 ./build-cuda/awave -mpi 2,2,2 -shape 1000,1000,1000 -nsteps 20 -io 0

# 非立方体
mpirun -np 4 ./build-cuda/awave -mpi 1,2,2 -shape 512,1000,768 -nsteps 20 -io 0
mpirun -np 8 ./build-cuda/awave -mpi 2,2,2 -shape 300,500,700 -nsteps 20 -io 0

# 非整除分解（验证 Decomposition 的 ceildiv + remainder 分配，以及不等大小 rank 的 halo 通信）
mpirun -np 8 ./build-cuda/awave -mpi 2,2,2 -shape 301,503,701 -nsteps 10 -io 0

# Checkpoint / restart
mpirun -np 4 ./build-cuda/awave -mpi 1,2,2 -shape 512,512,512 -nsteps 50 -io 1 -out_period 10 test
mpirun -np 4 ./build-cuda/awave -mpi 1,2,2 -shape 512,512,512 -nsteps 100 -restart test.gpu.vtkhdf -io 0

# Cross-NP restart（可选但推荐：验证“不同 GPU 数/分解”读取同一 checkpoint 也正确）
# 若 8GPU 排不到，至少做 4→2 或 4→1。
mpirun -np 2 ./build-cuda/awave -mpi 1,1,2 -shape 512,512,512 -nsteps 100 -restart test.gpu.vtkhdf -io 0

# 小域边界（have_interior=false 路径）
mpirun -np 8 ./build-cuda/awave -mpi 2,2,2 -shape 16,16,16 -nsteps 20 -io 0
```

> NP=1 时没有任何 MPI 邻居：虽然 CUDA 路径仍会走 `halo_post_receives/halo_launch_pack` 等统一流水线，但所有 face 都会 `!active()` 并被安全跳过；本步的 `-np 1` 回归用来确认“零活跃面”不会触发错误。

> Restart 陷阱（必须确认）：`-nsteps` 在本框架里是**目标总步数（final time step）**，不是“额外再跑多少步”。
> 例如 checkpoint 写在第 50 步，`-nsteps 100` 表示从 50 继续跑到 100。
> 实操检查：看日志里的 `Current time step: <t0>`，并确保 `-nsteps > t0`；否则会出现 chunk 为空/行为异常。

> 若你主要通过 K8s jobs 跑：对应模板在 `jobs/run-cw2-correctness-*.yml`；产物会写到 `logs/` 下固定目录（见 `jobs/JOBS.md`）。

---

## 第 3 步：补齐实验数据（~1 天）

> 目标：覆盖课程要求的 A100/H100、GPU 数 1~8、size 256~最大可行规模的性能数据，并保留可复现实验脚本/JSON（课程建议覆盖到 2500，但 2500³ 可能受单节点内存限制，见下方说明）。

### 3.1 性能测量口径（写进报告里）
- 每个配置重复 **3 次**。
- 只在第 2 步通过后，性能跑可用 `-skip_cpu 1` 提速。
- 报告吞吐用 **steady-state**（避免冷启动）：
	- 如果 JSON 有多个 GPU 采样（例如 `GPU.sups[0]`/`GPU.sups[1]`），丢弃第一个样本，报告第二个（或均值/最大值，保持一致）。
	- 若只产生一个样本：用更长 `-nsteps` 或运行两段采样的 job 模板（仓库已有 steady-state 聚合工具 `tools/aggregate_stats.py`）。
	- 若发现第 2 个 chunk 仍不稳定（例如首次 MPI 抖动跨到第 2 个 chunk）：把 `-nsteps` 增大到能产生 ≥3 个 chunk，并固定规则为“丢弃第 1 个，取最后 1 个 chunk（或取最后 N 个 chunk 均值）”。报告里必须写清该规则。

### 3A. 全矩阵性能数据（A100 & H100；NP=1/2/4/8）

#### 3A-1. 推荐：直接用 `jobs/` 模板跑矩阵
- A100 8-GPU：`jobs/run-cw2-perf-a100-8gpu.yml`
- H100 8-GPU：`jobs/run-cw2-perf-h100-8gpu.yml`
- 若 8-GPU 排队久：先跑 4-GPU fallback（`run-cw2-perf-*-4gpu.yml`）

> 8GPU 排队风险（重要）：如果截止前始终拿不到 8GPU 节点，就把“最大 GPU 数”降级为 4，并在报告中明确写：
> "8-GPU results could not be obtained due to cluster scheduling constraints; scaling trends are extrapolated from 1/2/4-GPU measurements." 
> 同时保留你尝试排队的证据（job 名称/时间戳/`kgpu jls` 输出或日志）。

这些模板会把 JSON 写到 `logs/awave-cw2-perf-*/`，并可用：

```bash
python3 tools/aggregate_stats.py \
	--a100-dir logs/awave-cw2-perf-a100-4g \
	--h100-dir logs/awave-cw2-perf-h100-4g \
	--case-out logs/summary_4g_robust.csv \
	--cross-out logs/summary_4g_cross_gpu_robust.csv \
	--scaling-out logs/summary_4g_scaling_robust.csv
```

#### 3A-2. 备选：手动 mpirun 批量生成 JSON（你已在节点上时）

> 注：`GPU` 仅用于文件命名；请在进入对应节点后手动设为 `a100` 或 `h100`（不要在同一节点上跑两次并用不同名字冒充）。
> size 覆盖要求：至少包含 256；上限以“单节点可运行最大 size”为准（见 2500³ 内存说明）。

```bash
GPU=a100  # 或 h100
	for NP in 1 2 4 8; do
		case $NP in
			1) MPI="1,1,1"; SIZES="256 512 768 1000" ;;
			2) MPI="1,1,2"; SIZES="256 512 768 1000 1500" ;;
			4) MPI="1,2,2"; SIZES="256 512 768 1000 1500" ;;
			8) MPI="2,2,2"; SIZES="256 512 768 1000 1500" ;;
		esac
	for SIZE in $SIZES; do
		for RUN in 1 2 3; do
			mpirun -np $NP ./build-cuda/awave -mpi $MPI \
				-shape ${SIZE},${SIZE},${SIZE} \
				-nsteps 20 -out_period 10 -skip_cpu 1 -io 0 \
				-json logs/final_${GPU}_np${NP}_s${SIZE}_r${RUN}.json
		done
	done
done
```

> 约束说明：`NP=1` 的 `1500^3` 在当前卡型上显存不足，matrix 中单卡点只保留到 `1000^3`。

> 2500³（重要现实约束）：默认**不把 2500 放进批量列表**。
> 在 8 ranks（常见 2×2×2）下，本项目每 rank 的 process-local 数据非常大，且 CUDA 路径会把 CPU 状态 clone 一份到 CUDA 状态（`CudaWaveSimulation::from_cpu_sim`），所以 host 端会同时驻留 **两份**全场数组；即使 `-skip_cpu 1` 也不会避免这些分配。
> 粗略内存（仅 u/cs2/damp/sos 这些主数组，不含 halo buffers、HDF5 缓冲、MPI 内部开销）：
> - device：约 78 GB / GPU（`u_prev/u_now/u_next/cs2/damp`，已接近 80 GB 上限）
> - host：约 188 GB / rank × 8 ≈ 1.5 TB（CPU state + CUDA state 两份；远超 512 GB 节点 RAM）
> 经验公式（host，便于写进报告）：每个（含 padding 的）grid point 在 host 侧约 **96 B**（6 个 double array：`u_prev/u_now/u_next/sos/cs2/damp`，乘以 CPU+CUDA 两份状态）。因此仅 host 侧（理论下限）：
> - `2000^3`：约 768 GB（在 512 GB 节点上基本必 OOM）
> - `1500^3`：约 324 GB（理论下限）；考虑 halo/pinned buffer/MPI/NSYS 额外开销，matrix 任务按 **376Gi** 申请
> 因此 2500³ 在单节点 8GPU 上基本不可行。报告里用这些估算解释，并给出你实际能跑的最大 size（常见会落在 1500³ 附近；更大 size 需先估算再尝试）。

### 3B. 8GPU MPI 分解对比（surface-to-volume 影响）

> 若 8GPU 排不到：把同样的对比降级到 NP=4（例如比较 `1,2,2` vs `1,1,4` vs `2,2,1`），仍可用来支撑“选择最对称分解以最小化 surface-to-volume”的论点。

```bash
for MPI in "2,2,2" "1,2,4" "1,1,8"; do
	for RUN in 1 2 3; do
		mpirun -np 8 ./build-cuda/awave -mpi $MPI \
			-shape 1000,1000,1000 \
			-nsteps 20 -skip_cpu 1 -io 0 \
			-json logs/decomp_h100_np8_${MPI//,/_}_r${RUN}.json
	done
done
```

### 3C. Weak scaling（固定每 GPU 的局部规模）

> strong scaling 固定全局 size；weak scaling 固定“每 GPU 的工作量”，更能体现通信开销随 GPU 数增长的趋势。
> 推荐先做 H100 一组（若时间允许再补 A100）。

示例：每 GPU 固定 500³（因此全局 shape 随 NP 增长），每个配置重复 3 次：

> 写进报告的说明（避免读者疑惑）：weak scaling 这里固定的是“每 rank 的局部 500³ 工作量”，因此全局 shape 随 NP 变化且通常不是立方体（例如 NP=4 时为 500×1000×1000）。

```bash
# weak scaling: 500³ per GPU
for RUN in 1 2 3; do
	for NP in 1 2 4 8; do
		case $NP in
			1) MPI="1,1,1"; SHAPE="500,500,500" ;;
			2) MPI="1,1,2"; SHAPE="500,500,1000" ;;
			4) MPI="1,2,2"; SHAPE="500,1000,1000" ;;
			8) MPI="2,2,2"; SHAPE="1000,1000,1000" ;;
		esac
		mpirun -np $NP ./build-cuda/awave -mpi $MPI \
			-shape $SHAPE -nsteps 20 -skip_cpu 1 -io 0 \
			-json logs/final_h100_weak_np${NP}_r${RUN}.json
	done
done
```

---

## 第 4 步：整理报告素材（~半天）

> 目标：把“可复现的实验细节 + 定量证据 + 物理解释”准备成表格/图的输入。

### 表 1：实验环境（硬件 + 软件版本）

老师反馈："What software versions? Document experiments in enough detail to enable others to reproduce them."

建议表结构：

| 项目 | 值 | 获取方式 / 备注 |
|---|---|---|
| GPU | NVIDIA A100 80GB SXM / NVIDIA H100 80GB SXM | `nvidia-smi -L` |
| HBM 带宽 | A100: 2.04 TB/s；H100: 3.35 TB/s | 引用 NVIDIA spec（需引用） |
| L2 Cache | A100: 40 MB；H100: 50 MB | spec（需引用） |
| FP64 峰值 | A100: 9.7 TFLOP/s；H100: 33.5 TFLOP/s | spec（需引用） |
| 互联 | NVLink（A100: 600 GB/s；H100: 900 GB/s 双向） | spec（需引用） |
| 节点 RAM | 512 GB | 运行时记录/集群文档 |
| CUDA 版本 | X.Y | `nvcc --version` 或 `nvidia-smi` |
| MPI 实现 | OpenMPI Z.W | `mpirun --version`/`ompi_info` |
| 编译模式 | Release | `cmake -DCMAKE_BUILD_TYPE=Release` |
| 编译器 | nvc++ + nvcc（由 CMake 选择） | `cmake --build` 输出/`CMakeCache.txt` |
| 重复次数 | 每配置 3 次，报告 best steady-state | 见第 3 步 |

> 注意：以上带宽/峰值均为 **NVIDIA data sheet 的理论峰值**，实际可达带宽通常是其 85–90%。报告中若计算“带宽利用率”，需要注明这是对理论峰值的利用率。
> 另外 H100 的 FP64 峰值存在 sparse/non-sparse 两个数字；本报告应使用 **non-sparse**（且非 TensorCore 的常规 FP64）并注明来源。最终请用 `nvidia-smi -q` 确认实际 SKU（SXM vs PCIe），并选用匹配的 spec sheet。

### 表 2：优化 A/B 决策总结（写“为什么做/为什么不做”）

| 优化 | 理论依据 | A100 实测 | H100 实测 | 决策 | 物理解释 |
|---|---|---:|---:|:---:|---|
| boundary_split | 消除空跑线程 | +4.5% | +22.2% | ✅ | split 版只启动精确边界点，避免 interior early-return 浪费 |
| block (32,4,4) | x 方向 L2 复用 | best | best | ✅ | 4 层 x 使相邻平面 stencil 读取更易命中 L2 |
| overlap reorder | 计算/通信并行 | +3.8% | ~0% | ✅ | 单卡无损；多卡可让 interior 与 pack 并行 |
| damp branchless | 消除 divergence | -0.08% | -0.09% | ❌ | `d==0` 时 warp 内一致，分支几乎零成本；branchless 引入额外 FP64 除法 |
| z_padding 对齐 | 128B 对齐 | -0.4% | -0.6% | ❌ | L2 已吸收未对齐访问；padding 增加总 DRAM 流量 |
| `__ldg` 显式只读 | 强制只读 cache | -0.01% | -0.03% | ❌ | `const __restrict__` 已足够让编译器生成只读加载 |
| shared-memory tiling | 减少全局读 | 无收益 | 无收益 | ❌ | L2 对 7-point stencil 已足够；shmem 装载/同步开销抵消收益 |

### 表 3：单卡 Roofline 数据（基于带宽受限推导）

| GPU | size | SU/s | 反推 B/site | HBM peak | 带宽利用率（opt） |
|---|---:|---:|---:|---:|---:|
| H100 | 256³ | 7.53e10 | 44.5 | 3.35 TB/s | 89.9% |
| H100 | 512³ | 6.98e10 | 48.0 | 3.35 TB/s | 83.3% |
| H100 | 1000³ | 6.98e10 | 48.0 | 3.35 TB/s | 83.3% |
| A100 | 256³ | 待测 | 待测 | 2.04 TB/s | 待测 |
| A100 | 512³ | 待测 | 待测 | 2.04 TB/s | 待测 |
| A100 | 1000³ | 待测 | 待测 | 2.04 TB/s | 待测 |
| A100 | 768³ | 3.42e10 | 59.6 | 2.04 TB/s | 83.6% |

> 注：B/site 推导方法与假设要写清楚，并在文中解释为何是 bandwidth-bound。
> 若时间允许，补齐 A100 单卡的 256³/512³/1000³，使表 3 与 H100 对称；若来不及，报告里只对比 A100 768³ 也可以，但需要说明原因（例如排队/时间）。

### 表 4：Scaling 效率（由第 3 步数据填充）

| GPU 数 | A100 SU/s | A100 效率 | H100 SU/s | H100 效率 |
|---:|---:|---:|---:|---:|
| 1 | 3.42e10 | 100% | 6.71e10 | 100% |
| 4 | 1.277e11 | 93.3% | 2.468e11 | 91.9% |
| 8 | 待测 | 待测 | 待测 | 待测 |

### 表 5：通信分析（Nsight Systems，统一口径）

> 为避免“总量 vs 平均值 / 每 step vs 每调用 / ms vs ns”的混用，表 5 统一来自 `logs/profiles/nsys_scale_*/*_stats_mpi_event_sum.csv` 与对应 `summary.md`。

| GPU | GPU数 | 问题规模 | interior kernel (ms) | MPI_Waitall avg (ms) | MPI_Waitall total (ms) | 通信占比 (%) | overlap proxy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A100 | 2 | 1000³ | 13.564 | 0.242 | 9.683 | 3.21 | 96.43 |
| A100 | 4 | 1000³ | 6.772 | 0.563 | 45.068 | 13.34 | 83.36 |
| H100 | 2 | 1000³ | 7.166 | 0.470 | 18.806 | 10.90 | 86.88 |
| H100 | 4 | 1000³ | 3.513 | 1.282 | 102.547 | 40.37 | 27.03 |
| H100 | 8 | 1000³ | 1.753 | 2.889 | 462.277 | 75.29 | 0.00 |

注释要点：
- `MPI_Waitall total` 是 trace 内所有 `MPI_Waitall` 的累加时间；`avg` 是每次调用平均。
- 若要换算成“每 step 的总等待”，用 `total / profiled_steps`（并在报告里写清你 profile 了多少步）。
- `通信占比 (%)` 定义（来自 `tools/nsys_comm_summary.py`）：
	- $f_{comm} = T_{waitall,total} / (T_{compute,total} + T_{waitall,total})$，其中 $T_{compute,total}=T_{interior,total}+T_{boundary,total}$。
- `overlap proxy (%)` 定义（来自 `tools/nsys_comm_summary.py`，这是**自定义 proxy**，不是严格的 overlap 测量）：
	- $p_{ov} = \max\left(0, 1 - T_{waitall,total} / T_{interior,total}\right)$。
	- 直觉：若 $T_{waitall,total} \ll T_{interior,total}$，则通信“理论上更可能被 interior 计算隐藏”；若 $T_{waitall,total} \ge T_{interior,total}$，proxy 记为 0。
- 版本一致性（重要）：表 5 引用的 nsys 数据必须与最终提交代码一致。当前仓库里 H100-8G 的 `overlap proxy=0.00%` 来自重排序前的 profile；若最终提交的是重排序版本，则要么在第 3 步后补一次“最终代码”的 nsys，要么在表注中明确写：
	- "profiled with the pre-reorder pipeline; the submitted code uses the reordered pipeline described in Section 1.6"。
- wall-clock 通信占比（例如 H100-4 的 40.37%）可能远大于吞吐损失（例如 ~8%），因为双 stream 流水线把部分通信隐藏在 interior 计算之后。

补充口径（避免误读）：
- 只用 `MPI_Waitall`（或 lock/walltime）来算“通信占比”通常会偏大，不等于真实通信开销。
- 按最新 H100 4 卡性能数据（`awave-cw2-perf-h100-4g-pq`）：
	- $T_1=0.0679244\ \text{s}$
	- $T_4=0.01778585\ \text{s}$
	- 强缩放效率 $E_4=T_1/(4T_4)=95.48\%$
- 解释：4 卡相对理想线性仅多出约 `4.52%` 总开销（通信 + 同步 + 负载不均合计），不支持“通信占比很高”的结论。

报告写作建议（最终口径）：
- 把强缩放效率作为“通信影响大小”的主指标。
- 把 NSYS 的 `MPI_Waitall` 指标作为“等待行为证据”，而不是严格通信占比。
- 汇总脚本应同时输出 `E_n` 与“相对理想额外开销”：
	- $E_n=\dfrac{T_1}{nT_n}$
	- $\text{extra\_overhead}_n=1-E_n$

### 图清单

- 图 1：SU/s vs 问题规模（1/2/4/8 GPU，A100 与 H100 两个子图），标注单卡理论上限
- 图 2：Strong scaling 效率 vs GPU 数（size=1000³），含理想线性虚线
- 图 3：反推 B/site vs 问题规模（H100 单卡），标注 40 B/site 下界与 88 B/site 上界

---

## 第 5 步：报告撰写（~1.5 天）

格式要求：PDF，≤ 5 页，字号 ≥ 10pt，无 introduction/conclusion。

### Section 1: Design and process（~2 页）

1) 实验环境（表 1）

建议写法：
> "All experiments were compiled with `cmake -DCMAKE_BUILD_TYPE=Release` using CUDA X.Y and OpenMPI Z.W on EIDF nodes equipped with ..."
> "Each configuration was run three times; the best steady-state throughput is reported to reduce sensitivity to system noise."

2) 选择 CUDA
- 引用 CW1 数据
- stream + event 适合计算-通信重叠

3) Roofline 分析
- 12 FLOP/site (undamped)，额外 ~6 FLOP (damped)
- Naive: 10 loads + 1 store = 88 B/site → I = 0.14
- Optimistic: 4 reads + 1 write = 40 B/site → I = 0.30
- $I_{ridge} = P_{FP64}/B_{HBM}$：
	- 统一口径：使用 **FP64 non-TensorCore、non-sparse 的峰值**，并引用对应 GPU（SXM/PCIe）的 NVIDIA data sheet。
	- A100：若采用 9.7 TFLOP/s 与 2.04 TB/s，则 $I_{ridge}\approx 4.8$；若你引用的 spec 给的是其它 FP64 定义（例如 TensorCore FP64），则必须在文中写清并保持全篇一致。
	- H100：33.5 TFLOP/s（non-sparse）与 3.35 TB/s 时 $I_{ridge}\approx 10.0$。
- 结论：$I \ll I_{ridge}$ → bandwidth-bound
- 实测反推 48 B/site（表 3），解释 L2 复用（~45% stencil 流量）

4) 单 GPU kernel 设计
- thread 映射：k→threadIdx.x 合并访问
- block (32,4,4) 选择理由：完整 warp 合并 + 4 层 x 方向 L2 复用
- boundary_split：三个 face kernel 的设计（表 2 的收益）

5) 否决的优化（表 2）

每项一小段，固定模板：
> "X was tested to address Y. Measurement showed Z% change (A100/H100). This is because ..."

6) 多 GPU 通信策略
- 双 stream 架构（compute_stream + halo_stream）
- 8 阶段重排序流水线（你的 `run()` 注释）
- cross-stream event 依赖

### Section 2: Results（~2 页）

1) 单 GPU 性能：图 1 + 表 3
- H100/A100 比值与带宽比的关系
- 图 3：反推 B/site 随 size 的变化

2) Strong scaling：图 2 + 表 4
- 93%/92% 效率来源
- H100 效率略低：计算更快 → 通信比例放大

3) 通信分析：表 5

建议解释段：
> "The nsys wall-clock breakdown overstates the communication cost because it does not account for the overlap between the interior kernel on compute_stream and the pack/MPI operations on halo_stream. The dual-stream pipeline effectively hides most of the communication behind computation, as evidenced by the 92% scaling efficiency."

4) MPI 分解对比（若有 3B 数据）
- 2×2×2 vs 1×2×4 vs 1×1×8
- surface-to-volume 比差异

补一句实验设计假设（写进报告，避免“为何这样选 mpi 形状”的疑问）：
> "For each GPU count we chose the most symmetric decomposition available to minimise the surface-to-volume ratio and hence communication volume. Section X validates this choice by comparing alternative decompositions at NP=8."

### Section 3: Further work（~1 页）

1) 多节点扩展（定量预测）
- 对比 NVLink 与 InfiniBand 带宽/延迟
- 预测通信占比变化范围（25–40%）
- 可能的改进：`cudaLaunchHostFunc` callback（可能需 `MPI_THREAD_MULTIPLE`）

2) Temporal blocking
- K 步一次 DRAM，B/site 从 48 降到 48/K
- 讨论 halo 深度×K、register 压力、shmem 占用

3) 其他方向
- 混合精度（正确性风险）
- MPI persistent requests
- CUDA Graphs（小问题 launch overhead）

---

## 第 6 步：校对与提交（~2 小时）

### 6.1 Release 编译验证

```bash
cmake -S src -B build-cuda -DCMAKE_BUILD_TYPE=Release -DAWAVE_MODE=CUDA
cmake --build build-cuda -j
```

### 6.2 正确性最终确认（Release 模式）

```bash
mpirun -np 1 ./build-cuda/awave -mpi 1,1,1 -shape 256,256,256 -nsteps 20 -io 0
mpirun -np 4 ./build-cuda/awave -mpi 1,2,2 -shape 512,512,512 -nsteps 20 -io 0
mpirun -np 4 ./build-cuda/awave -mpi 1,2,2 -shape 512,512,512 -nsteps 50 -io 1 -out_period 10 test
mpirun -np 4 ./build-cuda/awave -mpi 1,2,2 -shape 512,512,512 -nsteps 100 -restart test.gpu.vtkhdf -io 0
```

### 6.3 提交前检查清单

- 代码：`git diff` 仅包含 `src/wave_cuda.cu` 修改。
- 代码：Release 编译零 warning（或你能解释且不影响评分）。
- 代码：CUDA 资源正确释放（析构检查）。
- 代码：每个关键函数有清晰注释（kernel 映射 + overlap 逻辑）。
- 报告：PDF ≤ 5 页；字号 ≥ 10pt。
- 报告：实验环境记录完整（GPU 型号、CUDA 版本、MPI 版本、编译模式）。
- 报告：明确写 "each configuration was run N times; best steady-state reported"。
- 报告：每个优化决策有实测数据 + 物理解释。
- 报告：所有图有轴标签、单位、图例。
- 报告：引用完整（GPU spec sheet、roofline 方法论）。
- 报告：解释“通信占比 vs scaling 效率”的看似矛盾。
- 报告：Further work 含定量多节点预测。

> 5 页风险控制（提前做）：
> - 表 4（scaling）与表 5（通信）可合并为一个紧凑表，或把表 5 改成 2–3 行“代表性配置”。
> - 多张图优先用子图（同一 figure 内 2×1 或 2×2），减少 caption 占用。
> - 否决项优化不要每项一段长文：用 2–3 句合并描述“为什么无收益”的共同原因。

### 6.4 打包

> 注意：本仓库的 `make_submission.sh` **必须**提供 `-m`。

```bash
./make_submission.sh -e BXXXXXX -r report.pdf -m CUDA

# 验证包内容
tar -tzf aspp-cw2-BXXXXXX.tar.gz
```
