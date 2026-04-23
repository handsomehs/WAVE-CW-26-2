#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/eidf018/eidf018/shared/s2792840-epcc-pvc/ASPP_CW2/aspp_26_cw2"
OUT_DIR="${AWAVE_NSYS_ROOFLINE_OUT_DIR:-$ROOT/logs/profiles/nsys_roofline_h100}"

cd "$ROOT"

if [[ ! -d build-cuda ]]; then
  cmake -S src -B build-cuda -DAWAVE_MODE=CUDA -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build-cuda -j

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found in container PATH"
  exit 127
fi

mkdir -p "$OUT_DIR"

detect_gpu_mem_bytes() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local mem_mib
    mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)
    if [[ "$mem_mib" =~ ^[0-9]+$ ]]; then
      echo $((mem_mib * 1024 * 1024))
      return
    fi
  fi
  # Fallback for environments where nvidia-smi is unavailable in PATH.
  echo $((80 * 1024 * 1024 * 1024))
}

run_case() {
  local size="$1"
  local prefix="$OUT_DIR/nsys_single_${size}"
  local json_out="${prefix}.json"

  AWAVE_CUDA_OVERLAP=1 \
  AWAVE_CUDA_TILE=0 \
  AWAVE_CUDA_BLOCK=6 \
  AWAVE_CUDA_DAMP_BRANCHLESS=0 \
  AWAVE_CUDA_ZPAD=0 \
  AWAVE_CUDA_BOUNDARY_SPLIT=1 \
  nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    -f true \
    -o "$prefix" \
    -- \
    build-cuda/awave \
      -skip_cpu 1 \
      -mpi 1,1,1 \
      -shape "${size},${size},${size}" \
      -nsteps 20 \
      -io 0 \
      -json "$json_out" \
      "$prefix"

  if [[ ! -f "${prefix}.nsys-rep" ]]; then
    echo "ERROR: expected report missing: ${prefix}.nsys-rep"
    return 1
  fi

  nsys stats \
    --force-export=true \
    --format csv \
    --report cuda_gpu_kern_sum,cuda_kern_exec_trace \
    --output "${prefix}_stats" \
    "${prefix}.nsys-rep"
}

GPU_MEM_BYTES=$(detect_gpu_mem_bytes)
MIN_DEVICE_BYTES_PER_SITE=24
RUN_COUNT=0

for SIZE in 256 512 1000 2000; do
  EST_MIN_DEVICE_BYTES=$((SIZE * SIZE * SIZE * MIN_DEVICE_BYTES_PER_SITE))
  if (( EST_MIN_DEVICE_BYTES > (GPU_MEM_BYTES * 9 / 10) )); then
    echo "Skipping size ${SIZE}: estimated minimum device footprint ${EST_MIN_DEVICE_BYTES} B exceeds 90% of GPU memory ${GPU_MEM_BYTES} B"
    continue
  fi
  run_case "$SIZE"
  RUN_COUNT=$((RUN_COUNT + 1))
done

if (( RUN_COUNT == 0 )); then
  echo "ERROR: no feasible sizes were profiled under current GPU memory constraints"
  exit 1
fi

python3 tools/nsys_roofline_fallback_summary.py \
  --dir "$OUT_DIR" \
  --hbm-peak-bytes-per-sec 3.35e12 \
  --json-out "$OUT_DIR/summary.json" \
  --md-out "$OUT_DIR/summary.md"

ls -lh "$OUT_DIR"
