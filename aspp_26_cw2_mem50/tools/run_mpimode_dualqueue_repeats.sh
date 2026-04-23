#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

NS=${NS:-eidf018ns}
REPEATS=3
PAIR_LIST="a100-2g,h100-2g"
POLL_SEC=${POLL_SEC:-20}
TIMEOUT_SEC=${TIMEOUT_SEC:-2400}

STATUS_MD="logs/mpimode_dualqueue_repeats_status.md"
RUN_LOG="logs/mpimode_dualqueue_repeats.log"

usage() {
  cat <<'EOF'
Usage: tools/run_mpimode_dualqueue_repeats.sh [--repeats N] [--pairs a100-2g,h100-2g] [--namespace eidf018ns]

Behavior:
- For each pair/repeat, submit profiling-queue and user-queue jobs simultaneously.
- As soon as one queue produces valid mode_device/mode_host JSON outputs, delete the sibling job from the other queue.
- Keep per-repeat outputs in unique directories with suffix -rN.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --pairs)
      PAIR_LIST="$2"
      shift 2
      ;;
    --namespace)
      NS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

timestamp() { date -u +%FT%TZ; }
log() { echo "[$(timestamp)] $*" | tee -a "$RUN_LOG"; }

mkdir -p logs
: > "$RUN_LOG"

declare -A MANIFEST_PQ=(
  [a100-2g]="jobs/run-cw2-mpimode-ab-a100-2gpu-profiling.yml"
  [h100-2g]="jobs/run-cw2-mpimode-ab-h100-2gpu-profiling.yml"
)
declare -A MANIFEST_UQ=(
  [a100-2g]="jobs/run-cw2-mpimode-ab-a100-2gpu-user.yml"
  [h100-2g]="jobs/run-cw2-mpimode-ab-h100-2gpu-user.yml"
)
declare -A OUTDIR_PQ=(
  [a100-2g]="logs/awave-cw2-mpimode-ab-a100-2g-pq"
  [h100-2g]="logs/awave-cw2-mpimode-ab-h100-2g-pq"
)
declare -A OUTDIR_UQ=(
  [a100-2g]="logs/awave-cw2-mpimode-ab-a100-2g-uq"
  [h100-2g]="logs/awave-cw2-mpimode-ab-h100-2g-uq"
)

render_manifest() {
  local base=$1
  local out=$2
  local old_dir=$3
  local new_dir=$4
  local new_generate_name=$5

  python3 - "$base" "$out" "$old_dir" "$new_dir" "$new_generate_name" <<'PY'
import re
import sys
from pathlib import Path

base, out, old_dir, new_dir, new_generate_name = sys.argv[1:]
text = Path(base).read_text(encoding="utf-8")
if old_dir not in text:
    raise SystemExit(f"Expected output directory not found in manifest: {old_dir}")
text = text.replace(old_dir, new_dir)
text, n = re.subn(r"generateName:\s*\S+", f"generateName: {new_generate_name}-", text, count=1)
if n != 1:
    raise SystemExit("Failed to rewrite generateName in manifest")
Path(out).write_text(text, encoding="utf-8")
PY
}

job_state() {
  local job=$1
  if ! kubectl -n "$NS" get job "$job" >/dev/null 2>&1; then
    echo deleted
    return 0
  fi
  local s f a sp
  s=$(kubectl -n "$NS" get job "$job" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)
  f=$(kubectl -n "$NS" get job "$job" -o jsonpath='{.status.failed}' 2>/dev/null || true)
  a=$(kubectl -n "$NS" get job "$job" -o jsonpath='{.status.active}' 2>/dev/null || true)
  sp=$(kubectl -n "$NS" get job "$job" -o jsonpath='{.spec.suspend}' 2>/dev/null || true)

  if [[ "$s" == "1" ]]; then
    echo complete
  elif [[ "$f" =~ ^[1-9][0-9]*$ ]]; then
    echo failed
  elif [[ "$sp" == "true" ]]; then
    echo suspended
  elif [[ "$a" =~ ^[1-9][0-9]*$ ]]; then
    echo running
  else
    echo pending
  fi
}

validate_output_dir() {
  local out_dir=$1
  python3 - "$out_dir" <<'PY'
import json
import os
import sys

out_dir = sys.argv[1]
for fname in ("mode_device.json", "mode_host.json"):
    path = os.path.join(out_dir, fname)
    if not os.path.exists(path):
        raise SystemExit(1)
    doc = json.load(open(path, "r", encoding="utf-8"))
    sups = doc.get("GPU", {}).get("sups", [])
    if not sups:
        raise SystemExit(2)
print("ok")
PY
}

submit_manifest() {
  local manifest=$1
  if command -v kgpu >/dev/null 2>&1; then
    kgpu create -f "$manifest"
  else
    kubectl -n "$NS" create -f "$manifest" -o name
  fi
}

init_status_md() {
  {
    echo "# Dual-Queue MPI Mode Repeats"
    echo
    echo "- Updated: $(timestamp)"
    echo "- Namespace: ${NS}"
    echo "- Repeats: ${REPEATS}"
    echo "- Pairs: ${PAIR_LIST}"
    echo
    echo "| pair | repeat | pq_job | uq_job | winner | winner_dir | action |"
    echo "|---|---:|---|---|---|---|---|"
  } > "$STATUS_MD"
}

append_status_row() {
  local pair=$1
  local rep=$2
  local pq_job=$3
  local uq_job=$4
  local winner=$5
  local winner_dir=$6
  local action=$7
  printf '| %s | %s | %s | %s | %s | %s | %s |\n' \
    "$pair" "$rep" "$pq_job" "$uq_job" "$winner" "$winner_dir" "$action" >> "$STATUS_MD"
}

run_one_repeat() {
  local pair=$1
  local rep=$2

  local base_pq=${MANIFEST_PQ[$pair]:-}
  local base_uq=${MANIFEST_UQ[$pair]:-}
  local outbase_pq=${OUTDIR_PQ[$pair]:-}
  local outbase_uq=${OUTDIR_UQ[$pair]:-}

  if [[ -z "$base_pq" || -z "$base_uq" || -z "$outbase_pq" || -z "$outbase_uq" ]]; then
    log "Unknown pair '$pair'"
    return 1
  fi

  local outdir_pq="${outbase_pq}-r${rep}"
  local outdir_uq="${outbase_uq}-r${rep}"
  mkdir -p "$outdir_pq" "$outdir_uq"

  if validate_output_dir "$outdir_pq" >/dev/null 2>&1; then
    log "skip pair=${pair} repeat=${rep}: pq output already valid at ${outdir_pq}"
    append_status_row "$pair" "$rep" "-" "-" "pq" "$outdir_pq" "skip_existing_valid"
    return 0
  fi
  if validate_output_dir "$outdir_uq" >/dev/null 2>&1; then
    log "skip pair=${pair} repeat=${rep}: uq output already valid at ${outdir_uq}"
    append_status_row "$pair" "$rep" "-" "-" "uq" "$outdir_uq" "skip_existing_valid"
    return 0
  fi

  local tmp_pq tmp_uq
  tmp_pq=$(mktemp "/tmp/${pair}_pq_r${rep}_XXXX.yaml")
  tmp_uq=$(mktemp "/tmp/${pair}_uq_r${rep}_XXXX.yaml")

  render_manifest "$base_pq" "$tmp_pq" "$outbase_pq" "$outdir_pq" "awave-cw2-mpimode-ab-${pair}-pq-r${rep}"
  render_manifest "$base_uq" "$tmp_uq" "$outbase_uq" "$outdir_uq" "awave-cw2-mpimode-ab-${pair}-uq-r${rep}"

  local pq_job uq_job
  pq_job=$(submit_manifest "$tmp_pq" | sed -n 's#.*job.batch/##p' | tail -n 1)
  uq_job=$(submit_manifest "$tmp_uq" | sed -n 's#.*job.batch/##p' | tail -n 1)

  if [[ -z "$pq_job" || -z "$uq_job" ]]; then
    log "failed to parse created job names for pair=${pair} repeat=${rep}"
    rm -f "$tmp_pq" "$tmp_uq"
    return 1
  fi

  rm -f "$tmp_pq" "$tmp_uq"

  log "submitted pair=${pair} repeat=${rep} pq_job=${pq_job} uq_job=${uq_job}"

  local start_ts now elapsed
  start_ts=$(date +%s)

  local winner=""
  local winner_dir=""
  while true; do
    local pq_state uq_state
    pq_state=$(job_state "$pq_job")
    uq_state=$(job_state "$uq_job")

    if [[ "$pq_state" == "complete" ]]; then
      if validate_output_dir "$outdir_pq" >/dev/null 2>&1; then
        winner="pq"
        winner_dir="$outdir_pq"
        break
      fi
    fi
    if [[ "$uq_state" == "complete" ]]; then
      if validate_output_dir "$outdir_uq" >/dev/null 2>&1; then
        winner="uq"
        winner_dir="$outdir_uq"
        break
      fi
    fi

    if [[ "$pq_state" == "failed" && "$uq_state" == "failed" ]]; then
      break
    fi

    now=$(date +%s)
    elapsed=$((now - start_ts))
    if (( elapsed > TIMEOUT_SEC )); then
      log "timeout pair=${pair} repeat=${rep} pq_state=${pq_state} uq_state=${uq_state}"
      break
    fi

    sleep "$POLL_SEC"
  done

  if [[ -n "$winner" ]]; then
    local loser_job action
    if [[ "$winner" == "pq" ]]; then
      loser_job="$uq_job"
    else
      loser_job="$pq_job"
    fi

    kubectl -n "$NS" delete job "$loser_job" --ignore-not-found >/dev/null 2>&1 || true
    action="deleted_loser_job=${loser_job}"
    log "winner pair=${pair} repeat=${rep} winner=${winner} winner_dir=${winner_dir} ${action}"
    append_status_row "$pair" "$rep" "$pq_job" "$uq_job" "$winner" "$winner_dir" "$action"
  else
    local action
    action="no_valid_winner_keep_both_for_debug"
    log "no valid winner pair=${pair} repeat=${rep}"
    append_status_row "$pair" "$rep" "$pq_job" "$uq_job" "none" "-" "$action"
  fi
}

main() {
  init_status_md

  local pair
  IFS=',' read -r -a pairs <<< "$PAIR_LIST"
  for pair in "${pairs[@]}"; do
    pair=$(echo "$pair" | xargs)
    [[ -n "$pair" ]] || continue
    local rep
    for rep in $(seq 1 "$REPEATS"); do
      run_one_repeat "$pair" "$rep"
    done
  done

  log "all repeats completed"
  echo "$STATUS_MD"
}

main "$@"
