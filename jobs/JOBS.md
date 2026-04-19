# CW2 job submission quick start

These job templates follow the same workflow style as your previous coursework in [wave-cw-26](../../wave-cw-26).

## 1) Build locally first

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
cmake -S src -B build-cuda -DAWAVE_MODE=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

## 2) Submit correctness job (A100 80GB, 1 GPU)

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
kgpu create -f jobs/run-cw2-correct-a100.yml
kgpu jls -n 'awave-cw2-correct-a100-*'
kgpu logs -j <job-name>
```

Expected in logs:
- `Checking CUDA results...`
- `Number of differences detected = 0`

## 3) Submit performance matrix (A100 80GB, 8 GPUs)

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
kgpu create -f jobs/run-cw2-perf-a100-8gpu.yml
kgpu jls -n 'awave-cw2-perf-a100-8g-*'
kgpu logs -j <job-name>
```

This job writes JSON and output files to fixed folders under `logs/` in your CW2 workspace:

- `logs/awave-cw2-correct-a100/`
- `logs/awave-cw2-perf-a100-8g/`
- `logs/awave-cw2-perf-h100-8g/`

Each case produces one JSON file with timing and `ndiff`.

If 8-GPU jobs stay pending for a long time, run the 4-GPU fallback first:

```bash
kgpu create -f jobs/run-cw2-perf-a100-4gpu.yml
kgpu create -f jobs/run-cw2-perf-h100-4gpu.yml
```

These write to:

- `logs/awave-cw2-perf-a100-4g/`
- `logs/awave-cw2-perf-h100-4g/`

## 4) Submit performance matrix (H100, 8 GPUs)

Before submit, verify node label for H100 on your cluster:

```bash
kubectl get nodes -L nvidia.com/gpu.product
```

If needed, edit [jobs/run-cw2-perf-h100-8gpu.yml](run-cw2-perf-h100-8gpu.yml) and update `nodeSelector`.

Then submit:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
kgpu create -f jobs/run-cw2-perf-h100-8gpu.yml
kgpu jls -n 'awave-cw2-perf-h100-8g-*'
kgpu logs -j <job-name>
```

## Notes

- Runtime mode is controlled by `AWAVE_MPI_MODE`:
  - `device` (default in templates): CUDA-aware MPI path.
  - `host`: host-staging fallback if device-pointer MPI is unstable.
- Overlap mode is controlled by `AWAVE_CUDA_OVERLAP`:
  - `1` / unset (default): allow interior compute and halo path to overlap.
  - `0`: disable overlap for A/B comparison experiments.
- Tile mode is controlled by `AWAVE_CUDA_TILE`:
  - `1`: use shared-memory tiled interior kernel.
  - `0` / unset: use baseline global-memory interior kernel.
- Block mode is controlled by `AWAVE_CUDA_BLOCK`:
  - `0`: block `(32,4,2)` (kept for A/B baseline)
  - `1`: block `(16,8,2)`
  - `2`: block `(8,8,4)`
  - `3`: block `(64,2,2)`
  - `4`: block `(128,2,1)`
  - `5`: block `(32,8,1)`
  - `6` (default): block `(32,4,4)` (selected from 1-GPU A100 tune)
- Damping update mode is controlled by `AWAVE_CUDA_DAMP_BRANCHLESS`:
  - `0` / unset (default): branched damping update (`d==0` fast-path enabled)
  - `1`: branchless damping update
- Z padding mode is controlled by `AWAVE_CUDA_ZPAD`:
  - `0` / unset (default): no extra z-padding (`u_stride_y = nz + 2`)
  - `1`: align z stride to 16 doubles
- Block tuning user-queue jobs now use a larger case (`-shape 768,768,768`, `-nsteps 12`) to reduce
  short-run noise and make small kernel gains easier to observe.
- All templates use `-io 0` to keep performance runs focused on compute + communication.
- `awave` runs a CPU reference + correctness check by default; use `-skip_cpu 1` for large performance cases.
  Note: `-skip_cpu 1` disables correctness checking and omits CPU stats from the JSON output.

## 5) Profiling queue fallback (short runs)

If `run-cw2-perf-a100-8gpu.yml` stays pending too long in `eidf018ns-user-queue`, you can try `eidf018ns-profiling-queue`.

Operational requirements on this shared queue:

- Ensure `metadata.labels.kueue.x-k8s.io/queue-name: eidf018ns-profiling-queue`
- Ensure `metadata.labels.kueue.x-k8s.io/max-exec-time-seconds: "300"`
- Set `spec.activeDeadlineSeconds: 300`

### 5.1 Check if A100 80GB is admitted on profiling queue

```bash
kgpu create -f jobs/run-cw2-probe-a10080-profiling.yml
kubectl get jobs | grep awave-cw2-probe-a10080-pq
```

- `Running/Complete`: profiling queue can currently admit A100 80GB.
- `Suspended`: not currently admitted (queue/quota/resource not available now).

### 5.2 Submit A100 8-GPU run on profiling queue

```bash
kgpu create -f jobs/run-cw2-perf-a100-8gpu-profiling.yml
kubectl get jobs | grep awave-cw2-perf-a100-8g-pq
```

Output folder:

- `logs/awave-cw2-perf-a100-8g-pq/`

This template uses `-skip_cpu 1` so it finishes quickly; run a separate correctness job without `-skip_cpu` to validate results.

### 5.3 Submit H100 8-GPU run on profiling queue

```bash
kgpu create -f jobs/run-cw2-perf-h100-8gpu-profiling.yml
kubectl get jobs | grep awave-cw2-perf-h100-8g-pq
```

Output folder:

- `logs/awave-cw2-perf-h100-8g-pq/`

### 5.4 Run overlap A/B experiment (A100, 8 GPUs, profiling queue)

This template runs the same case twice:

- `AWAVE_CUDA_OVERLAP=0` -> baseline without overlap
- `AWAVE_CUDA_OVERLAP=1` -> overlap enabled

Submit:

```bash
kgpu create -f jobs/run-cw2-overlap-ab-a100-profiling.yml
kubectl get jobs | grep awave-cw2-overlap-ab-a100-pq
```

Output folder:

- `logs/awave-cw2-overlap-ab-a100-pq/`

You can also run the same A/B experiment in user queue:

```bash
kgpu create -f jobs/run-cw2-overlap-ab-a100-user.yml
kubectl get jobs | grep awave-cw2-overlap-ab-a100-uq
```

### 5.5 Run overlap A/B fallback (A100, 4 GPUs)

Profiling queue:

```bash
kgpu create -f jobs/run-cw2-overlap-ab-a100-4gpu-profiling.yml
kubectl get jobs | grep awave-cw2-overlap-ab-a100-4g-pq
```

User queue:

```bash
kgpu create -f jobs/run-cw2-overlap-ab-a100-4gpu-user.yml
kubectl get jobs | grep awave-cw2-overlap-ab-a100-4g-uq
```

Output folders:

- `logs/awave-cw2-overlap-ab-a100-4g-pq/`
- `logs/awave-cw2-overlap-ab-a100-4g-uq/`

## 6) Fix statistics baseline (all-sample + steady-state)

Generate robust summaries from raw JSON outputs:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
python3 tools/aggregate_stats.py \
  --a100-dir logs/awave-cw2-perf-a100-4g \
  --h100-dir logs/awave-cw2-perf-h100-4g \
  --case-out logs/summary_4g_robust.csv \
  --cross-out logs/summary_4g_cross_gpu_robust.csv \
  --scaling-out logs/summary_4g_scaling_robust.csv
```

Output files:

- `logs/summary_4g_robust.csv`: per-case mean/all, mean/steady, median, std, CV, speedup.
- `logs/summary_4g_cross_gpu_robust.csv`: A100 vs H100 ratio in both all-sample and steady-state views.
- `logs/summary_4g_scaling_robust.csv`: strong/weak scaling gain + efficiency for both views.

Steady-state is defined as dropping the first GPU sample for each case when multiple samples exist.

## 7) Start Nsight profiling runs

Submit Nsight Systems:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
kgpu create -f jobs/run-cw2-profile-nsys-a100.yml
kubectl get jobs | grep awave-cw2-nsys-a100
```

Submit Nsight Compute:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
kgpu create -f jobs/run-cw2-profile-ncu-a100.yml
kubectl get jobs | grep awave-cw2-ncu-a100
```

Expected artifacts:

- `logs/profiles/nsys/` (`.qdrep`, `.sqlite`, plus run JSON)
- `logs/profiles/ncu/` (`.ncu-rep`, plus run JSON)

Current observation on this image/runtime: `nsys` completes and writes report artifacts, while `ncu` may complete with
`No metrics to collect found in sections` / `No kernels were profiled` and produce no `.ncu-rep`.

## 8) Multi-GPU Nsight Systems overlap evidence (np=4 / np=8)

Submit np=4 timeline job:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
kgpu create -f jobs/run-cw2-profile-nsys-a100-4gpu.yml
```

Submit np=8 timeline job:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
kgpu create -f jobs/run-cw2-profile-nsys-a100-8gpu.yml
```

Each job writes:

- `${PREFIX}.nsys-rep`
- `${PREFIX}_stats_cuda_gpu_kern_sum.csv`
- `${PREFIX}_stats_mpi_event_sum.csv`
- `${PREFIX}_stats_cuda_kern_exec_trace.csv`
- `${PREFIX}_stats_mpi_event_trace.csv`
- `${PREFIX}_overlap.txt` (derived overlap evidence)

Where `PREFIX` is:

- `logs/profiles/nsys/cw2_nsys_a100_np4`
- `logs/profiles/nsys/cw2_nsys_a100_np8`

Operational notes from live runs:

- Avoid shell variables for output prefixes in job command blocks; use explicit absolute paths.
- For np8 profiling stability on current runtime image, use host MPI transfer mode and conservative MPI flags
  (`coll_hcoll_enable=0`, `coll_ucc_enable=0`, `pml=ob1`, `btl=self,vader,tcp`).

## 9) Queue-Parallel Overlap + Tune (A100/H100)

Use both queues in parallel and keep whichever finishes first.

### 9.1 Overlap A/B on both queues

A100 fallback 2-GPU:

```bash
kgpu create -f jobs/run-cw2-overlap-ab-a100-2gpu-profiling.yml
kgpu create -f jobs/run-cw2-overlap-ab-a100-2gpu-user.yml
```

H100 fallback 2-GPU:

```bash
kgpu create -f jobs/run-cw2-overlap-ab-h100-2gpu-profiling.yml
kgpu create -f jobs/run-cw2-overlap-ab-h100-2gpu-user.yml
```

Existing larger overlap templates (already in repo):

- `jobs/run-cw2-overlap-ab-a100-profiling.yml`
- `jobs/run-cw2-overlap-ab-a100-user.yml`
- `jobs/run-cw2-overlap-ab-a100-4gpu-profiling.yml`
- `jobs/run-cw2-overlap-ab-a100-4gpu-user.yml`
- `jobs/run-cw2-overlap-ab-h100-8gpu-profiling.yml`
- `jobs/run-cw2-overlap-ab-h100-8gpu-user.yml`

### 9.2 Tile/block tune on both queues

```bash
kgpu create -f jobs/run-cw2-tune-tileblock-a100-1gpu-profiling.yml
kgpu create -f jobs/run-cw2-tune-tileblock-a100-1gpu-user.yml
kgpu create -f jobs/run-cw2-tune-tileblock-h100-1gpu-profiling.yml
kgpu create -f jobs/run-cw2-tune-tileblock-h100-1gpu-user.yml
```

### 9.3 Artifact directories

Overlap outputs:

- `logs/awave-cw2-overlap-ab-a100-2g-pq/`
- `logs/awave-cw2-overlap-ab-a100-2g-uq/`
- `logs/awave-cw2-overlap-ab-a100-4g-pq/`
- `logs/awave-cw2-overlap-ab-h100-2g-pq/`
- `logs/awave-cw2-overlap-ab-h100-2g-uq/`
- `logs/awave-cw2-overlap-ab-h100-pq/`

Tune outputs:

- `logs/awave-cw2-tune-a100-1g-pq/`
- `logs/awave-cw2-tune-a100-1g-uq/`
- `logs/awave-cw2-tune-h100-1g-pq/`
- `logs/awave-cw2-tune-h100-1g-uq/`

### 9.4 Queue and quota note

- H100 8-GPU overlap can be blocked by namespace GPU quota during busy periods.
- The 2-GPU overlap templates are intended as robust fallback to guarantee at least one admitted H100 overlap A/B run.

### 9.5 Template safety rule

- For these cluster templates, do not rely on shell variables in job command blocks.
- Use explicit absolute paths and explicit command arguments to avoid templating-time variable stripping.

### 9.6 Three-repeat dual-queue protocol (recommended)

To reduce noise, run each overlap task with 3 repeats, submit to both queues at the same time, and delete the sibling queue job once one queue yields valid `overlap0.json` + `overlap1.json`.

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
bash tools/run_overlap_dualqueue_repeats.sh --repeats 3 --pairs a100-2g,h100-2g --namespace eidf018ns
```

This writes:

- `logs/overlap_dualqueue_repeats_status.md`
- `logs/awave-cw2-overlap-ab-a100-2g-{pq,uq}-r{1,2,3}/`
- `logs/awave-cw2-overlap-ab-h100-2g-{pq,uq}-r{1,2,3}/`

### 9.7 Promote/drop gate after repeats

Compute repeat-only summary:

```bash
python3 tools/overlap_gate.py \
  --dir-glob 'awave-cw2-overlap-ab-*-r*' \
  --no-promote-threshold-pct 2.0 \
  --json-out logs/overlap_gate_summary_repeats.json \
  --md-out logs/overlap_gate_summary_repeats.md
```

Operational decision rule:

- Promote overlap and continue 4-GPU/8-GPU queueing if:
  - `gain_steady_pct >= 3.0`, and
  - enough repeat evidence (`pairs >= 3`), and
  - queue disagreement is bounded (`queue_spread_pct <= 8.0`).
- Drop overlap for that GPU count if:
  - `gain_steady_pct <= -2.0`, and
  - queue gains are not positive.
- Mark as `no-promote` (terminal low-impact decision) if:
  - `gain_steady_pct <= 2.0`, and
  - enough repeat evidence (`pairs >= 3`), and
  - queue disagreement is bounded (`queue_spread_pct <= 8.0`).
- For `no-promote` on 2-GPU:
  - stop overlap-specific campaign,
  - cancel pending 4-GPU/8-GPU overlap jobs,
  - move to next optimization stage.
- Reserve `needs-retest` only for ambiguous mid-gain cases after checks above
  (for example, gain between `2.0` and `3.0` with unstable queue behavior).

Fast-close policy for this project:

- If 2-GPU gate reaches `promote`, continue overlap scaling checks.
- If 2-GPU gate reaches `drop` or `no-promote`, do not wait for 4-GPU/8-GPU overlap results.

## 10) MPI mode A/B (device vs host) with dual queue

Templates:

- `jobs/run-cw2-mpimode-ab-a100-2gpu-profiling.yml`
- `jobs/run-cw2-mpimode-ab-a100-2gpu-user.yml`
- `jobs/run-cw2-mpimode-ab-h100-2gpu-profiling.yml`
- `jobs/run-cw2-mpimode-ab-h100-2gpu-user.yml`

Repeat runner (winner queue retained, sibling queue job deleted after valid JSON appears):

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
bash tools/run_mpimode_dualqueue_repeats.sh --repeats 3 --pairs a100-2g,h100-2g --namespace eidf018ns
```

Gate summary:

```bash
python3 tools/mpimode_gate.py \
  --dir-glob 'awave-cw2-mpimode-ab-*-r*' \
  --json-out logs/mpimode_gate_summary_repeats.json \
  --md-out logs/mpimode_gate_summary_repeats.md
```

## 11) MPI stack A/B (default vs `ob1+self,vader,tcp`)

Templates:

- `jobs/run-cw2-mpistack-ab-a100-2gpu-profiling.yml`
- `jobs/run-cw2-mpistack-ab-a100-2gpu-user.yml`
- `jobs/run-cw2-mpistack-ab-h100-2gpu-profiling.yml`
- `jobs/run-cw2-mpistack-ab-h100-2gpu-user.yml`

Single submit (manual):

```bash
kgpu create -f jobs/run-cw2-mpistack-ab-a100-2gpu-profiling.yml
kgpu create -f jobs/run-cw2-mpistack-ab-a100-2gpu-user.yml
kgpu create -f jobs/run-cw2-mpistack-ab-h100-2gpu-profiling.yml
kgpu create -f jobs/run-cw2-mpistack-ab-h100-2gpu-user.yml
```

Dual-queue adaptive runner (base 3 + conditional 2):

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
bash tools/run_mpistack_dualqueue_repeats.sh --repeats 3 --pairs a100-2g,h100-2g --namespace eidf018ns
```

Gate summary:

```bash
python3 tools/mpistack_gate.py \
  --dir-glob 'awave-cw2-mpistack-ab-*-r*' \
  --json-out logs/mpistack_gate_summary_repeats.json \
  --md-out logs/mpistack_gate_summary_repeats.md
```

Current decision interpretation:

- Positive `ob1_gain_steady_pct`: promote `ob1+self,vader,tcp`.
- Negative `ob1_gain_steady_pct`: keep default MPI stack.
- Near-neutral gain (default band `±0.5%`): keep default MPI stack and move on.
- High cross-queue spread or low pairs: run more repeats.

## 12) MPI waitsome A/B (off vs on) with dual queue

Switch under test:

- `AWAVE_MPI_WAITSOME=0` (or unset): baseline halo completion path.
- `AWAVE_MPI_WAITSOME=1`: progressive receive completion using `MPI_Waitsome` with per-face unpack enqueue.

Templates:

- `jobs/run-cw2-waitsome-ab-a100-2gpu-profiling.yml`
- `jobs/run-cw2-waitsome-ab-a100-2gpu-user.yml`
- `jobs/run-cw2-waitsome-ab-h100-2gpu-profiling.yml`
- `jobs/run-cw2-waitsome-ab-h100-2gpu-user.yml`

Dual-queue repeat runner:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
bash tools/run_dualqueue_adaptive_repeats.sh \
  --campaign waitsome \
  --base-repeats 3 \
  --extra-repeats 2 \
  --pairs a100-2g,h100-2g \
  --namespace eidf018ns
```

Runner outputs:

- `logs/waitsome_dualqueue_repeats_status.md`
- `logs/awave-cw2-waitsome-ab-a100-2g-{pq,uq}-r{1,2,3}/` (+ `r{4,5}` only when gate marks problematic)
- `logs/awave-cw2-waitsome-ab-h100-2g-{pq,uq}-r{1,2,3}/` (+ `r{4,5}` only when gate marks problematic)

Gate summary:

```bash
python3 tools/waitsome_gate.py \
  --logs-root logs \
  --dir-glob 'awave-cw2-waitsome-ab-*-r*' \
  --no-promote-threshold-pct 2.0 \
  --json-out logs/waitsome_gate_summary_repeats.json \
  --md-out logs/waitsome_gate_summary_repeats.md
```

Operational decision rule:

- Promote waitsome if `gain_steady_pct >= 3.0` and queue spread is bounded.
- Drop waitsome if `gain_steady_pct <= -2.0` and queue means are not positive.
- Mark `no-promote` if `gain_steady_pct <= 2.0` with enough repeat evidence.
- For 2-GPU `drop` or `no-promote`, fast-close this campaign and move to the next optimization stage.

Current decision (2026-04-18):

- A100 2G: `no-promote`
- H100 2G: `no-promote`
- Keep `AWAVE_MPI_WAITSOME` off for production runs.

## 13) Boundary split A/B (off vs on) with dual queue

Switch under test:

- `AWAVE_CUDA_BOUNDARY_SPLIT=0`: legacy boundary kernel launch over full grid.
- `AWAVE_CUDA_BOUNDARY_SPLIT=1` (promoted default): face-only boundary kernels.

Templates:

- `jobs/run-cw2-boundary-ab-a100-2gpu-profiling.yml`
- `jobs/run-cw2-boundary-ab-a100-2gpu-user.yml`
- `jobs/run-cw2-boundary-ab-h100-2gpu-profiling.yml`
- `jobs/run-cw2-boundary-ab-h100-2gpu-user.yml`

Dual-queue repeat runner:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
bash tools/run_dualqueue_adaptive_repeats.sh \
  --campaign boundary \
  --base-repeats 3 \
  --extra-repeats 2 \
  --pairs a100-2g,h100-2g \
  --namespace eidf018ns
```

Runner outputs:

- `logs/boundary_dualqueue_repeats_status.md`
- `logs/awave-cw2-boundary-ab-a100-2g-{pq,uq}-r{1,2,3}/` (+ `r{4,5}` only when gate marks problematic)
- `logs/awave-cw2-boundary-ab-h100-2g-{pq,uq}-r{1,2,3}/` (+ `r{4,5}` only when gate marks problematic)

Gate summary:

```bash
python3 tools/boundary_gate.py \
  --logs-root logs \
  --dir-glob 'awave-cw2-boundary-ab-*-r*' \
  --no-promote-threshold-pct 2.0 \
  --json-out logs/boundary_gate_summary_repeats.json \
  --md-out logs/boundary_gate_summary_repeats.md
```

Operational decision rule:

- Promote boundary split if `gain_steady_pct >= 3.0`.
- For queue spread checks, require at least `2` repeats per queue before treating spread as reliable.
- If spread is not reliable, decide using pooled gain and pair count.

Current decision (2026-04-18):

- A100 2G: `promote`
- H100 2G: `promote`
- Production default: keep boundary split enabled (unset `AWAVE_CUDA_BOUNDARY_SPLIT` => on).

### 13.1 Stage B boundary scaling validation (4G/8G)

Scaling templates:

- `jobs/run-cw2-boundary-ab-a100-4gpu-profiling.yml`
- `jobs/run-cw2-boundary-ab-a100-4gpu-user.yml`
- `jobs/run-cw2-boundary-ab-h100-8gpu-profiling.yml`
- `jobs/run-cw2-boundary-ab-h100-8gpu-user.yml`

Launch adaptive repeats for scaling pairs:

```bash
cd /home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2
bash tools/run_dualqueue_adaptive_repeats.sh \
  --campaign boundary \
  --base-repeats 3 \
  --extra-repeats 2 \
  --pairs a100-4g,h100-8g \
  --namespace eidf018ns
```

Optional per-pair gate summaries:

```bash
python3 tools/boundary_gate.py \
  --logs-root logs \
  --dir-glob 'awave-cw2-boundary-ab-a100-4g-*-r*' \
  --json-out logs/boundary_gate_summary_repeats_a100_4g.json \
  --md-out logs/boundary_gate_summary_repeats_a100_4g.md

python3 tools/boundary_gate.py \
  --logs-root logs \
  --dir-glob 'awave-cw2-boundary-ab-h100-8g-*-r*' \
  --json-out logs/boundary_gate_summary_repeats_h100_8g.json \
  --md-out logs/boundary_gate_summary_repeats_h100_8g.md
```
