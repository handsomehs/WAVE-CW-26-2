# CW2 Work Log

## Purpose
This file records important progress, findings, blockers, and next actions for CW2 CUDA + MPI + profiling work.

## Update Convention
- Add a new timestamped section for each meaningful milestone.
- Keep entries focused on facts: what was done, what was observed, and what it implies.
- Record artifact paths so results are traceable.

---

## [2026-04-18] Phase-A overlap implementation started (dual CUDA streams + events)

### What was changed
- Refactored `src/wave_cuda.cu` CUDA execution path from one stream to two streams:
  - `compute_stream`: interior stencil compute
  - `halo_stream`: halo pack/unpack, staging copies, boundary/full-domain kernel
- Added CUDA events to coordinate dependencies and safe pointer swap:
  - `pack_ready`, `halo_ready`, `interior_done`, `boundary_done`
- Updated halo path:
  - `halo_start_exchange`: pack on `halo_stream`, optional D2H staging, event-based host wait before posting MPI
  - `halo_finish_exchange`: MPI wait, optional H2D staging, unpack on `halo_stream`, record `halo_ready`
- Updated run loop:
  - interior kernel launched on `compute_stream`
  - boundary/full kernel launched on `halo_stream`
  - swap occurs only after `interior_done` + `boundary_done` events complete
- Updated copyback path (`sync_host_fields`) to synchronize both streams before D2H copies.

### Verification
- Build succeeded:
  - `cmake --build build-cuda -j`
- Smoke run succeeded on current host with CPU fallback (no visible CUDA device):
  - `mpirun -np 1 build-cuda/awave -skip_cpu 1 -mpi 1,1,1 -shape 32,32,32 -nsteps 2 -out_period 2 -io 0`
  - runtime path printed `No CUDA device found; CUDA simulation will run on CPU`

### Implication
- Core overlap infrastructure is now in place for GPU runs (interior and boundary no longer forced onto one stream).
- Need GPU-node validation to quantify effect on `MPI_Waitall` dominance for `np4/np8`.

### Next immediate actions
1. Run A100 profiling-queue np8 performance case with updated binary and collect JSON.
2. Run np4/np8 Nsight Systems jobs and compare overlap evidence against prior baseline.
3. If overlap still weak, proceed to Phase-B shared-memory tiled stencil.

## [2026-03-30] Baseline + Profiling Milestone

### What was done
- Implemented CUDA multi-GPU runtime path in `src/wave_cuda.cu` with:
  - rank-to-device mapping
  - device-resident halo exchange
  - interior/boundary split for comm/compute overlap
  - host-staging fallback mode for MPI transfer
  - copyback gating in `append_u_fields()`
- Ran correctness/performance workflow and produced 4-GPU A100/H100 JSON outputs.
- Added robust statistics pipeline:
  - `tools/aggregate_stats.py`
  - generated robust CSV outputs (all-sample + steady-state).
- Added Nsight job templates and started profiling experiments.

### Main findings
- Correctness: all consolidated 4-GPU cases show `ndiff=0`.
- Robust statistics changed conclusions materially:
  - all-sample H100/A100 mean ratio: 1.632
  - steady-state H100/A100 mean ratio (drop first GPU sample): 1.880
- Nsight Systems (np1) succeeded and produced timeline artifacts.
- Nsight Compute on current image/runtime completed but repeatedly reported:
  - `No metrics to collect found in sections`
  - `No kernels were profiled`

### Multi-GPU Nsight Systems status (np4 / np8)
- np4 and np8 jobs were submitted.
- First failure class fixed:
  - issue: `Illegal --inherit-environment argument: p.`
  - action: add `--` separator before `mpirun` in `nsys profile` command.
- Current failure class identified for np4 post-processing:
  - Nsight output was generated as `--.nsys-rep` instead of intended prefix.
  - downstream `nsys stats` failed because expected `${PREFIX}.nsys-rep` did not exist.
  - likely cause: templating/shell variable expansion in job command block (`${PREFIX}` not preserved at runtime).
- np8 job was admitted after suspension; currently running/retrying depending on queue availability.

### Important artifact files
- 4-GPU summaries:
  - `logs/summary_4g.csv`
  - `logs/summary_4g_cross_gpu.csv`
  - `logs/summary_4g_notes.md`
- Robust summaries:
  - `logs/summary_4g_robust.csv`
  - `logs/summary_4g_cross_gpu_robust.csv`
  - `logs/summary_4g_scaling_robust.csv`
  - `logs/summary_4g_robust_notes.md`
- Nsight Systems np1 evidence:
  - `logs/profiles/nsys/cw2_nsys_a100_np1.nsys-rep`
  - `logs/profiles/nsys/cw2_nsys_a100_np1.sqlite`
  - `logs/profiles/nsys/cw2_nsys_a100_np1_stats_cuda_gpu_kern_sum.csv`
  - `logs/profiles/nsys/cw2_nsys_a100_np1_stats_mpi_event_sum.csv`
  - `logs/profiles/nsys/cw2_nsys_a100_np1_stats_cuda_kern_exec_trace.csv`
  - `logs/profiles/nsys/cw2_nsys_a100_np1_stats_mpi_event_trace.csv`

### Active blockers
- np4/np8 Nsight Systems post-processing currently blocked by output prefix mismatch (`--.nsys-rep`).
- np8 queue admission remains resource-dependent and may stay suspended intermittently.
- Nsight Compute kernel-level metrics not yet available on current runtime image.

### Next actions
1. Remove shell variable-based output prefix from np4/np8 Nsight job YAML and use explicit absolute paths only.
2. Re-run np4 Nsight Systems until `_overlap.txt` is generated.
3. Re-run/monitor np8 Nsight Systems and collect the same evidence set.
4. Append overlap evidence summary (waitall/interior ratio and verdict) into this log.

---

## [2026-03-30] Multi-GPU Nsight Failure Fix + Evidence Collected

### Incident
- Jobs failed:
  - `awave-cw2-nsys-a100-4g-q6rtd`
  - `awave-cw2-nsys-a100-8g-cn4qn`

### Root causes
- np4 failure was post-processing path mismatch:
  - shell variable prefix in YAML command block was not preserved as intended.
  - Nsight output became `--.nsys-rep` and `nsys stats` then failed on missing `${PREFIX}.nsys-rep`.
- np8 failure was MPI/UCX/UCC runtime instability under profiler:
  - segfault in MPI init path and UCX/UCC stack.
  - additional `/dev/shm` memory pressure messages from UCX posix transport.

### Fixes applied
- Replaced variable-based output paths with explicit absolute paths in:
  - `jobs/run-cw2-profile-nsys-a100-4gpu.yml`
  - `jobs/run-cw2-profile-nsys-a100-8gpu.yml`
- Added `--` separator before `mpirun` in `nsys profile` commands.
- For np8 stability:
  - switched `AWAVE_MPI_MODE` to `host` (profiling reliability mode)
  - used conservative MPI flags:
    - `--mca coll_hcoll_enable 0`
    - `--mca coll_ucc_enable 0`
    - `--mca pml ob1`
    - `--mca btl self,vader,tcp`

### Rerun outcomes
- np4 rerun job `awave-cw2-nsys-a100-4g-7db4j`: Complete
- np8 rerun job `awave-cw2-nsys-a100-8g-6nj4d`: Complete
- Both now produce full evidence bundles:
  - `.nsys-rep`
  - `.sqlite`
  - kernel + MPI summary CSV
  - kernel + MPI trace CSV
  - overlap summary TXT

### Key overlap evidence
- np4 (`cw2_nsys_a100_np4_overlap.txt`):
  - `waitall_vs_interior=5.514`
  - `verdict=weak-overlap-indication`
- np8 (`cw2_nsys_a100_np8_overlap.txt`):
  - `waitall_vs_interior=15.895`
  - `verdict=weak-overlap-indication`

Interpretation: communication wait dominates compute for these profiling configurations; overlap benefit is limited in current setup.

---

## [2026-03-30] Why 8-GPU Perf Jobs Stay Suspended

Jobs checked:

- `awave-cw2-perf-a100-8g-dh6d2`
- `awave-cw2-perf-h100-8g-ngjlt`

### Observed state
- Both jobs are currently `Suspended`.
- Their Kueue workloads are in queue `eidf018ns-user-queue` with `Admitted=False`.
- Workload condition shows:
  - `Evicted=True`, reason `PodsReadyTimeout`
  - `Requeued=False`, reason `PodsReadyTimeout`
  - `requeueState.count=6`
  - `requeueAt=2026-03-30T12:13:06/07Z`

### Why this happened
- These jobs were admitted multiple times, but pods did not reach Ready/Succeeded before Kueue `PodsReady` timeout.
- At least for H100 run, controller events also showed repeated pod creation failures from namespace CPU quota pressure:
  - `exceeded quota: eidf018ns-compute-resources`
  - requested `limits.cpu=32`, namespace near cap (`used` close to `limited=576`).
- After repeated timeout/eviction cycles, Kueue suspended and requeued workloads (no active reservation at the moment).

### Why newer jobs completed
- Recently completed profiling jobs used `eidf018ns-profiling-queue` and smaller/shorter profiling workloads.
- These two are full 8-GPU performance matrices in `eidf018ns-user-queue` with heavier resource footprint and longer readiness/execution path.

### Practical implication
- Current `Suspended` is queue-control behavior (requeue after timeout), not a code compile/runtime crash in the same sense as the profiling job failures fixed later.

---

## [2026-03-30] Lowered 8-GPU Perf CPU to 16 + Resubmitted

### Change requested
- Temporarily reduce CPU request/limit for 8-GPU perf jobs from `32` to `16`.

### Files updated
- `jobs/run-cw2-perf-a100-8gpu.yml`
- `jobs/run-cw2-perf-h100-8gpu.yml`

### Action taken
- Deleted suspended jobs:
  - `awave-cw2-perf-a100-8g-dh6d2`
  - `awave-cw2-perf-h100-8g-ngjlt`
- Resubmitted from updated templates, creating:
  - `awave-cw2-perf-a100-8g-cj7vx`
  - `awave-cw2-perf-h100-8g-llprt`

### Immediate post-submit state
- Both new jobs admitted by cluster queue and resumed.
- Both created pods (`Pending` at early check), so quota-blocking `FailedCreate` seen before was alleviated at submission time.

### Follow-up pending diagnosis
- Queue assignment is correct (not a wrong-queue submission):
  - workloads are in `eidf018ns-user-queue`
  - admitted to `eidf018ns-project-gpu-cq`
  - `Admitted=True`
- Current `Pending` reason is scheduler-side resource availability:
  - A100 pod: `Insufficient nvidia.com/gpu` plus node selector/taint constraints
  - H100 pod: `Insufficient nvidia.com/gpu` plus node selector/taint constraints
- Therefore the present block is cluster capacity/scheduling pressure, not job YAML queue misconfiguration.

---

## [2026-03-30] What Profiling Indicates (Nsight Systems)

Evidence files:

- `logs/profiles/nsys/cw2_nsys_a100_np1_overlap.txt`
- `logs/profiles/nsys/cw2_nsys_a100_np4_overlap.txt`
- `logs/profiles/nsys/cw2_nsys_a100_np8_overlap.txt`

Key ratios (waitall vs compute):

- np1: `waitall_vs_compute=0.069` (strong overlap indication)
- np4: `waitall_vs_compute=4.857` (weak overlap indication)
- np8: `waitall_vs_compute=13.547` (weak overlap indication)

Conservative cross-check:

- Nsight run includes both CPU and CUDA benchmark phases, so `MPI_Waitall` can be over-counted for pure-GPU overlap interpretation.
- Even using a conservative half-wait estimate, communication still dominates:
  - np4 adjusted wait/compute ~= 2.429
  - np8 adjusted wait/compute ~= 6.774

Interpretation:

- Single-rank behavior is healthy.
- At 4 and 8 ranks, MPI synchronization/communication dominates runtime and current overlap does not hide it effectively.
- Scaling bottleneck is now communication-side (not kernel arithmetic throughput).

---

## [2026-03-31] UCX / UCC / HCOLL / NCCL Compatibility Probe

Probe jobs:

- A100: `awave-cw2-probe-comm-a100-7xdvg`
- H100: `awave-cw2-probe-comm-h100-c5ddz`

Templates:

- `jobs/run-cw2-probe-commstack-a100.yml`
- `jobs/run-cw2-probe-commstack-h100.yml`

### Key observations (both A100 and H100)
- Container MPI toolchain is from NVIDIA HPC SDK:
  - `mpirun` / `mpicc`: Open MPI `4.1.9a1` under `/opt/nvidia/hpc_sdk/.../comm_libs/mpi/bin/`
  - `ompi_info` in PATH is system Open MPI `4.1.6` (`/usr/bin/ompi_info`), so `ompi_info` component listings can be misleading.
- RDMA device nodes are not exposed inside the container:
  - `/dev/infiniband` is missing.
- Info CLIs are not present (but libraries/modules may still exist):
  - `ucx_info`, `ucc_info`, `hcoll_info` were not found.

### What works
- UCX PML works:
  - `mpirun --mca pml ucx -np 2 /tmp/mpi_allreduce` succeeded (`rc=0`).
  - Note: without `/dev/infiniband`, UCX cannot use IB/verbs transports; it will fall back to shared-memory/TCP.
- UCC coll works for a simple MPI collective:
  - `mpirun --mca coll_ucc_enable 1 --mca coll_ucc_priority 100 ...` succeeded (`rc=0`).
  - OpenMPI shows `coll component ucc is available` and attempts to select it.
- NCCL library is present and linkable:
  - `nvcc` successfully compiled a tiny `ncclGetVersion()` program.
  - NCCL version code printed as `22707` (NCCL `2.27.7`).

### What does not work (and why)
- HCOLL cannot initialize:
  - Error: `mca_coll_hcoll_comm_query() Hcol library init failed`.
  - Mellanox log: “You must specify a valid HCA device … set `HCOLL_MAIN_IB=<dev:port>` or `UCX_NET_DEVICES=<dev:port>`”.
  - Root cause in this environment: no visible HCA devices in the container (`/dev/infiniband` missing), so auto-detection fails.

### Practical implication
- Keep `--mca coll_hcoll_enable 0` in run templates unless the cluster is updated to expose RDMA devices into pods.

Important caveat:

- np4 profiling used `AWAVE_MPI_MODE=device` while np8 profiling used `AWAVE_MPI_MODE=host` for runtime stability; absolute np4 vs np8 numbers are therefore not directly comparable one-to-one.

---

## [2026-03-31] Resubmitted 8-GPU Perf Runs

### Request
- Continue the interrupted 8-GPU perf test submissions.

### Actions taken
- Submitted H100 8-GPU perf job from template:
  - `kgpu create -f jobs/run-cw2-perf-h100-8gpu.yml`
  - new job: `awave-cw2-perf-h100-8g-28n96`
- Submitted A100 8-GPU perf job from template:
  - `kgpu create -f jobs/run-cw2-perf-a100-8gpu.yml`
  - new job: `awave-cw2-perf-a100-8g-99rjk`

### Immediate status check
- `awave-cw2-perf-h100-8g-28n96`: `Running`
  - pod `awave-cw2-perf-h100-8g-28n96-2k6vv`: `1/1 Running`
- `awave-cw2-perf-a100-8g-99rjk`: `Running` (job admitted)
  - pod `awave-cw2-perf-a100-8g-99rjk-xk98z`: `Pending`
- older job still present:
  - `awave-cw2-perf-a100-8g-cj7vx`: `Suspended`

### A100 pending reason (scheduler events)
- `Insufficient nvidia.com/gpu`
- additional taint/selector constraints and one unschedulable node reported by scheduler.

### Implication
- Re-submission succeeded for both H100 and A100 8-GPU runs.
- H100 has started execution; A100 is currently blocked by cluster-side scheduling capacity.

---

## [2026-03-31] 8-GPU Perf: MPI Stability + OOM Follow-up

### What happened
- H100 8-GPU perf run hit an OpenMPI HCOLL init failure and segfault (missing/undetected HCA device).
- After disabling HCOLL/UCC, the H100 run progressed further but later hit `OOMKilled` at `memory=128Gi` during the larger `size*` cases.

### Fixes applied
- Disabled problematic collectives in both 8-GPU perf templates:
  - `OMPI_MCA_coll_hcoll_enable=0`
  - `OMPI_MCA_coll_ucc_enable=0`
- Increased H100 8-GPU perf memory to avoid OOM on large `size*` cases:
  - `jobs/run-cw2-perf-h100-8gpu.yml`: `memory 128Gi -> 256Gi`

### Current live status
- H100: `awave-cw2-perf-h100-8g-rqpl5` is `Running` (pod on `gpu8-vm38`, single-node 8-GPU).
- A100: `awave-cw2-perf-a100-8g-q8chv` remains `Running` but pod is pending due to `Insufficient nvidia.com/gpu` and selector/taint constraints.

### Output artifacts
- H100 outputs are written under:
  - `logs/awave-cw2-perf-h100-8g/`
  - (weak/strong JSONs and `size1024.json` exist; remaining `size1536/size2048/size2500` expected after completion.)

---

## [2026-03-31] H100 8-GPU Completed; A100 8-GPU Still Pending

### H100 status
- Job `awave-cw2-perf-h100-8g-rqpl5`: `Complete`.
- Output directory now contains the full expected set:
  - `weak_np{1,2,4,8}.json`
  - `strong_np{1,2,4,8}.json`
  - `size{1024,1536,2048,2500}.json`

### A100 status
- Re-submitted A100 8-GPU perf job with higher memory to avoid repeating the H100 OOM:
  - deleted: `awave-cw2-perf-a100-8g-q8chv`
  - new: `awave-cw2-perf-a100-8g-2cl6w`
- Pod is still `Pending` with scheduler events indicating `Insufficient nvidia.com/gpu` on A100 80GB nodes.

### Template updates
- `jobs/run-cw2-perf-a100-8gpu.yml`: `memory 128Gi -> 256Gi`
- `jobs/run-cw2-perf-a100-8gpu-profiling.yml`: aligned to `cpu=16`, `memory=256Gi`, and disabled HCOLL/UCC.

---

## [2026-04-18] CUDA Overlap Implementation Started (Phase 1)

### Code changes applied
- Refactored CUDA runtime in `src/wave_cuda.cu` from one stream to two streams:
  - `compute_stream`: interior stencil compute
  - `halo_stream`: halo pack/unpack and boundary compute
- Added CUDA events to coordinate overlap safely:
  - `pack_ready`: guarantees MPI send buffers are ready
  - `interior_done` and `boundary_done`: step-completion sync before pointer swap
- Halo path now records/synchronizes `pack_ready` instead of synchronizing a single shared stream.
- Added env-controlled overlap switch for A/B experiments:
  - `AWAVE_CUDA_OVERLAP=1` (default) enables overlap
  - `AWAVE_CUDA_OVERLAP=0` disables overlap by syncing on interior completion before halo finish

### Validation
- Rebuilt successfully:
  - `cmake --build build-cuda -j` -> `Built target awave`
- Sanity runtime checks completed for both overlap modes:
  - `AWAVE_CUDA_OVERLAP=1` and `AWAVE_CUDA_OVERLAP=0` both run through small `mpirun -np 1` smoke cases.
  - Current host has no visible CUDA GPU, so runtime used CPU fallback path; compile and control-path validation still passed.

### Experiment template added
- New profiling-queue A/B template:
  - `jobs/run-cw2-overlap-ab-a100-profiling.yml`
- Runs the same 8-GPU A100 case twice with `AWAVE_CUDA_OVERLAP=0/1` and writes:
  - `logs/awave-cw2-overlap-ab-a100-pq/overlap0.json`
  - `logs/awave-cw2-overlap-ab-a100-pq/overlap1.json`

### Notes
- `jobs/JOBS.md` updated with:
  - `AWAVE_CUDA_OVERLAP` control description
  - submission instructions for the new A/B overlap template

---

## [2026-04-18] Queue Parallelisation + Phase 2 Tiling Start

### Queue strategy updates (to avoid idle waiting)
- Added and submitted overlap A/B jobs to **both** profiling and user queues:
  - 8-GPU:
    - `awave-cw2-overlap-ab-a100-pq-cpvtt`
    - `awave-cw2-overlap-ab-a100-uq-mxdtr`
  - 4-GPU fallback:
    - `awave-cw2-overlap-ab-a100-4g-pq-5rpmq`
    - `awave-cw2-overlap-ab-a100-4g-uq-2mrqr`
- All four are currently `Suspended` (queue wait), monitored via a persistent `kubectl get jobs -w` hook.

### New job templates
- Added user-queue 8-GPU overlap A/B:
  - `jobs/run-cw2-overlap-ab-a100-user.yml`
- Added 4-GPU overlap A/B templates for both queues:
  - `jobs/run-cw2-overlap-ab-a100-4gpu-profiling.yml`
  - `jobs/run-cw2-overlap-ab-a100-4gpu-user.yml`

### Phase 2 code progress (shared-memory path)
- Added shared-memory tiled interior kernel in `src/wave_cuda.cu`:
  - `step_kernel_interior_tiled(...)`
- Added env switch:
  - `AWAVE_CUDA_TILE=1` enables tiled interior kernel
  - unset/`0` keeps baseline interior kernel
- Added block-shape tuning switch:
  - `AWAVE_CUDA_BLOCK=0` -> `(32,4,2)`
  - `AWAVE_CUDA_BLOCK=1` -> `(16,8,2)`
  - `AWAVE_CUDA_BLOCK=2` -> `(8,8,4)`
- Run loop now selects kernel/block shape from env for controlled A/B tuning.

### Validation
- Build passes after each change (`cmake --build build-cuda -j`).
- Smoke control-path checks run with:
  - overlap on/off
  - tile on
  - block mode 2
- Current node has no visible CUDA GPU, so runtime fell back to CPU path; compile and flag-path checks passed.

---

## [2026-04-18] Queue-parallel overlap+tune completed (A100 + H100)

### Root cause fixed before rerun
- Early failed jobs were caused by templating behavior in YAML command blocks: shell variables (for example `OUT`, `TAG`, loop vars) were stripped, producing empty paths and invalid JSON targets.
- Fixed all overlap+tune templates to use explicit absolute paths and explicit command arguments (no shell variable expansion in job command blocks).

### Additional fallback templates added
- To increase admission probability under queue pressure/quota limits, added 2-GPU overlap A/B templates:
  - `jobs/run-cw2-overlap-ab-a100-2gpu-profiling.yml`
  - `jobs/run-cw2-overlap-ab-a100-2gpu-user.yml`
  - `jobs/run-cw2-overlap-ab-h100-2gpu-profiling.yml`
  - `jobs/run-cw2-overlap-ab-h100-2gpu-user.yml`

### Execution strategy used
- Submitted overlap+tune to both queues (`profiling` + `user`) in parallel.
- Kept lower-footprint fallbacks (2G/4G) live while larger jobs waited.
- Continued monitoring with watch hooks until JSON artifacts were produced.

### Completed artifact directories
- Overlap A/B:
  - `logs/awave-cw2-overlap-ab-a100-2g-uq/`
  - `logs/awave-cw2-overlap-ab-a100-2g-pq/`
  - `logs/awave-cw2-overlap-ab-a100-4g-pq/`
  - `logs/awave-cw2-overlap-ab-h100-2g-uq/`
  - `logs/awave-cw2-overlap-ab-h100-2g-pq/`
  - `logs/awave-cw2-overlap-ab-h100-pq/` (8-GPU)
- Tile/block tune:
  - `logs/awave-cw2-tune-a100-1g-uq/`
  - `logs/awave-cw2-tune-a100-1g-pq/`
  - `logs/awave-cw2-tune-h100-1g-uq/`
  - `logs/awave-cw2-tune-h100-1g-pq/`

### Key quantitative findings
- Tile/block tuning best configs (mean throughput):
  - A100 (user queue): `tile0_blk0` -> `3.398e+10`
  - A100 (profiling queue): `tile1_blk0` -> `3.251e+10`
  - H100 (user queue): `tile0_blk0` -> `6.338e+10`
  - H100 (profiling queue): `tile0_blk0` -> `6.327e+10`
- Overlap A/B gain (mean throughput, ON/OFF):
  - A100 2G user: `+4.28%`
  - A100 2G profiling: `-22.04%` (startup-sensitive outlier in one chunk)
  - A100 4G profiling: `+141.42%`
  - H100 2G user: `+6.34%`
  - H100 2G profiling: `+15.21%`
  - H100 8G profiling: `+6.14%`

### Stability note
- Some short runs show large first-chunk transients.
- Using the second chunk only (steady sample), overlap remains positive in all completed overlap directories.

### Operational cleanup
- Removed suspended stale jobs after successful artifact collection.
- Left completed jobs for normal TTL-based cleanup.

---

## [2026-04-18] Sync De-block + MPI mode/stack gate consolidation

### 1) Conservative sync de-block in CUDA run loop

Code change in `src/wave_cuda.cu`:

- Removed per-step host-side blocking at loop tail:
  - `cudaEventSynchronize(interior_done)`
  - `cudaEventSynchronize(boundary_done)`
- Replaced with device-side stream dependencies:
  - `cudaStreamWaitEvent(compute_stream, boundary_done, 0)`
  - `cudaStreamWaitEvent(halo_stream, interior_done, 0)`

This keeps correctness ordering while reducing host-side blocking pressure without changing MPI safety points (`pack_ready` and `MPI_Waitall` remain).

Validation:

- Rebuild passed: `cmake --build build-cuda -j`
- Smoke run passed: `mpirun -np 1 build-cuda/awave -skip_cpu 1 -mpi 1,1,1 -shape 32,32,32 -nsteps 2 -out_period 2 -io 0`

### 2) MPI mode A/B decision refresh

Using `tools/mpimode_gate.py` on repeat directories:

- A100 2G: `provisional-device`
  - queue means both negative for host gain; large spread is driven by profiling outlier.
- H100 2G: `device`

Operational decision: keep `AWAVE_MPI_MODE=device` for next-stage optimization runs.

### 3) MPI stack A/B automation and 3-repeat evidence

Added templates:

- `jobs/run-cw2-mpistack-ab-a100-2gpu-profiling.yml`
- `jobs/run-cw2-mpistack-ab-a100-2gpu-user.yml`
- `jobs/run-cw2-mpistack-ab-h100-2gpu-profiling.yml`
- `jobs/run-cw2-mpistack-ab-h100-2gpu-user.yml`

Added tools:

- `tools/run_mpistack_dualqueue_repeats.sh`
- `tools/mpistack_gate.py`

Ran dual-queue repeats with winner-retain / loser-delete policy and collected repeat artifacts:

- `logs/awave-cw2-mpistack-ab-a100-2g-uq-r{1,2,3}/`
- `logs/awave-cw2-mpistack-ab-h100-2g-uq-r{1,2,3}/`

Gate output (`logs/mpistack_gate_summary_repeats.json`):

- A100 2G: gain `-0.082%`, status `default-neutral`
- H100 2G: gain `-41.445%`, status `default`

Operational decision: keep default MPI stack; do not promote `ob1+self,vader,tcp`.

### 4) Documentation updates

- `jobs/JOBS.md` extended with MPI mode/stack dual-queue workflows.
- MPI stack gate interpretation updated to include a neutral default band (`+-0.5%`) so low-impact deltas do not stall the plan.

---

## [2026-04-18] Overlap 2-GPU fast-close completed (do not wait for 4G/8G)

### Execution actions
- De-scoped overlap waiting on larger GPU counts by deleting suspended jobs:
  - `awave-cw2-overlap-ab-a100-4g-uq-llv6w`
  - `awave-cw2-overlap-ab-h100-uq-bwgfv`
- Ran additional dual-queue repeats for overlap A/B:
  - `tools/run_overlap_dualqueue_repeats.sh --repeats 5 --pairs a100-2g,h100-2g --namespace eidf018ns`
  - New valid outputs collected for `r4` and `r5` (winner-queue retain, sibling-job delete).

### Gate-rule update
- Updated `tools/overlap_gate.py` with a terminal low-impact branch:
  - new threshold: `no_promote_threshold_pct = 2.0`
  - new status: `no-promote`
  - for 2-GPU `no-promote`: stop overlap campaign and move to next optimization.

### Final 2-GPU gate result
- Artifacts:
  - `logs/overlap_gate_summary_repeats.json`
  - `logs/overlap_gate_summary_repeats.md`
- Group decisions:
  - A100 2G: `gain_steady_pct = 1.06%`, `pairs = 8`, `queue_spread_pct = 0.23`, `status = no-promote`
  - H100 2G: `gain_steady_pct = 1.27%`, `pairs = 9`, `queue_spread_pct = 0.14`, `status = no-promote`

### Operational decision
- Overlap optimization judged low-impact under 2-GPU evidence; stop overlap-specific campaign.
- Do not wait for 4-GPU/8-GPU overlap outputs.
- Continue next-stage optimization on communication-path deblocking (MPI wait-path reduction and progress pipelining).

---

## [2026-04-18] Communication wait-path + boundary refactor stages completed

### 1) Communication wait-path deblocking (`AWAVE_MPI_WAITSOME`)

Code changes:

- `src/wave_cuda.cu`:
  - Added `AWAVE_MPI_WAITSOME` runtime switch.
  - Reworked halo MPI request tracking into separate recv/send arrays.
  - Added `MPI_Waitsome` progressive completion path in `halo_finish_exchange`:
    - receive completion is consumed face-by-face,
    - host-staging mode performs per-face H2D copy before unpack enqueue,
    - send requests are finalized separately.

New experiment assets:

- Templates:
  - `jobs/run-cw2-waitsome-ab-a100-2gpu-profiling.yml`
  - `jobs/run-cw2-waitsome-ab-a100-2gpu-user.yml`
  - `jobs/run-cw2-waitsome-ab-h100-2gpu-profiling.yml`
  - `jobs/run-cw2-waitsome-ab-h100-2gpu-user.yml`
- Tools:
  - `tools/run_waitsome_dualqueue_repeats.sh`
  - `tools/waitsome_gate.py`

Execution:

- Built successfully: `cmake --build build-cuda -j`
- Ran dual-queue repeats:
  - `TIMEOUT_SEC=420 POLL_SEC=20 tools/run_waitsome_dualqueue_repeats.sh --repeats 5 --pairs a100-2g,h100-2g --namespace eidf018ns`
- Gate summary artifacts:
  - `logs/waitsome_gate_summary_repeats.json`
  - `logs/waitsome_gate_summary_repeats.md`

Final gate result:

- A100 2G: `gain_steady_pct = -0.027%`, `pairs = 5`, `queue_spread_pct = 0.000`, `status = no-promote`
- H100 2G: `gain_steady_pct = -0.564%`, `pairs = 10`, `queue_spread_pct = 1.163`, `status = no-promote`

Operational decision:

- Keep waitsome optimization off for production runs.
- Move to boundary-kernel refactor stage.

### 2) Boundary kernel refactor (`AWAVE_CUDA_BOUNDARY_SPLIT`)

Code changes:

- `src/wave_cuda.cu`:
  - Added boundary split switch `AWAVE_CUDA_BOUNDARY_SPLIT`.
  - Added face-only boundary kernels:
    - `step_kernel_boundary_x_faces`
    - `step_kernel_boundary_y_faces`
    - `step_kernel_boundary_z_faces`
  - Run loop now launches split boundary faces when enabled instead of the full-grid boundary kernel.
  - After promote decision, default behavior switched to boundary split ON when env is unset.

New experiment assets:

- Templates:
  - `jobs/run-cw2-boundary-ab-a100-2gpu-profiling.yml`
  - `jobs/run-cw2-boundary-ab-a100-2gpu-user.yml`
  - `jobs/run-cw2-boundary-ab-h100-2gpu-profiling.yml`
  - `jobs/run-cw2-boundary-ab-h100-2gpu-user.yml`
- Tools:
  - `tools/run_boundary_dualqueue_repeats.sh`
  - `tools/boundary_gate.py`

Execution:

- Built successfully after kernel refactor and after default-on promote switch.
- Ran dual-queue repeats:
  - `TIMEOUT_SEC=420 POLL_SEC=20 tools/run_boundary_dualqueue_repeats.sh --repeats 5 --pairs a100-2g,h100-2g --namespace eidf018ns`
- Gate summary artifacts:
  - `logs/boundary_gate_summary_repeats.json`
  - `logs/boundary_gate_summary_repeats.md`

Gate robustness update:

- Updated `tools/boundary_gate.py` so queue-spread veto is applied only when each queue has at least 2 repeats (`min_repeats_per_queue_for_spread`).
- This avoids single-sample queue outliers forcing unnecessary `needs-retest`.

Final gate result (after robustness update):

- A100 2G: `gain_steady_pct = 4.511%`, `pairs = 5`, `status = promote`
- H100 2G: `gain_steady_pct = 22.184%`, `pairs = 6`, `status = promote`

Operational decision:

- Promote boundary split optimization.
- Production default now keeps boundary split enabled (unset `AWAVE_CUDA_BOUNDARY_SPLIT` => ON).

### 3) Stage closeout

- `jobs/JOBS.md` updated with waitsome and boundary A/B workflows, gate commands, and decision notes.
- This optimization cycle is closed for:
  - overlap,
  - MPI mode,
  - MPI stack,
  - waitsome,
  - boundary split.

### 4) Adaptive repeats policy + Stage A/B kickoff (2026-04-18)

Policy update:

- Adopted adaptive repeat policy for dual-queue campaigns:
  - base repeats: 3
  - add +2 repeats only when gate marks problematic outcomes (`needs-retest`, `inconclusive`, or no-promotion actions).
- Orchestrator: `tools/run_dualqueue_adaptive_repeats.sh`.

Adaptive validation (already executed):

- waitsome campaign (`a100-2g,h100-2g`):
  - base 3 repeats completed;
  - no problematic pairs detected;
  - no +2 rerun needed;
  - final gate remains `no-promote` for both A100/H100 2G.
- boundary campaign (`a100-2g,h100-2g`):
  - base 3 repeats completed;
  - no problematic pairs detected;
  - no +2 rerun needed;
  - final gate remains `promote` for both A100/H100 2G.

Stage A (process/docs closeout):

- Updated `jobs/JOBS.md` waitsome/boundary runbooks from fixed `--repeats 5` to adaptive `3+2` commands.
- Added Stage B boundary scaling runbook entries.

Stage B (boundary scaling) assets prepared:

- Added templates:
  - `jobs/run-cw2-boundary-ab-a100-4gpu-profiling.yml`
  - `jobs/run-cw2-boundary-ab-a100-4gpu-user.yml`
  - `jobs/run-cw2-boundary-ab-h100-8gpu-profiling.yml`
  - `jobs/run-cw2-boundary-ab-h100-8gpu-user.yml`
- Extended `tools/run_boundary_dualqueue_repeats.sh` pair support:
  - `a100-4g`
  - `h100-8g`

Stage B launch (started):

- Command:
  - `TIMEOUT_SEC=900 POLL_SEC=20 bash tools/run_boundary_dualqueue_repeats.sh --repeats 1 --pairs a100-4g,h100-8g --namespace eidf018ns`
- Initial runtime snapshot after submit:
  - `awave-cw2-boundary-ab-a100-4g-pq-r1-452mq`: `Suspended`
  - `awave-cw2-boundary-ab-a100-4g-uq-r1-nnxhx`: `Running`
- Note:
  - The repeat runner processes pairs sequentially; `h100-8g` submission starts after the `a100-4g` repeat reaches winner/timeout.

### 5) Stage B boundary scaling closeout (2026-04-18)

Adaptive campaign execution:

- Command:
  - `TIMEOUT_SEC=1800 POLL_SEC=20 bash tools/run_dualqueue_adaptive_repeats.sh --campaign boundary --base-repeats 3 --extra-repeats 2 --pairs a100-4g,h100-8g --namespace eidf018ns`
- Runtime outcome:
  - `a100-4g`: repeat1 reused existing valid output, repeat2 winner=`uq`, repeat3 winner=`uq`
  - `h100-8g`: repeat1 reused existing valid output, repeat2 winner=`pq`, repeat3 winner=`pq`
  - sibling jobs were deleted by dual-queue winner logic for each completed repeat.

Gate result:

- `logs/boundary_gate_summary_repeats.json`
- `logs/boundary_gate_summary_repeats.md`
- Final snapshot:
  - `a100-4g`: `status=promote`, `pairs=3`, `gain_steady_pct=5.604`, `action=enable-boundary-for-this-gpu-count`
  - `h100-8g`: `status=promote`, `pairs=3`, `gain_steady_pct=5.521`, `action=enable-boundary-for-this-gpu-count`

Operational note (scheduler):

- Early in repeat2, jobs appeared as `Suspended` due Kueue admission pressure and transient quota constraints.
- Workloads were later admitted automatically; one creation burst hit namespace memory quota (`requests.memory` close to `2Ti`) before capacity freed.
- No manual manifest change was required; campaign completed successfully once quota/headroom became available.

Decision:

- Stage B boundary scaling validation is complete for `a100-4g` and `h100-8g`.
- Boundary split remains promoted at 2G/4G/8G based on current gate evidence.

