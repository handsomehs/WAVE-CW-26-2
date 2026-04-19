#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

CAMPAIGN=""
NS=${NS:-eidf018ns}
PAIR_LIST="a100-2g,h100-2g"
BASE_REPEATS=3
EXTRA_REPEATS=2
POLL_SEC=${POLL_SEC:-20}
TIMEOUT_SEC=${TIMEOUT_SEC:-2400}
NO_PROMOTE_THRESHOLD_PCT=2.0
PROBLEM_ACTION_REGEX=${PROBLEM_ACTION_REGEX:-run-more|collect-more}

RUNNER_SCRIPT=""
GATE_JSON=""
RUN_LOG=""
declare -a GATE_CMD

usage() {
  cat <<'EOF'
Usage: tools/run_dualqueue_adaptive_repeats.sh --campaign <name> [options]

Campaign names:
  overlap | mpimode | mpistack | waitsome | boundary

Options:
  --campaign NAME                Campaign name (required)
  --pairs a100-2g,h100-2g        Pair list (default: a100-2g,h100-2g)
  --base-repeats N               Initial repeats before first gate (default: 3)
  --extra-repeats N              Extra repeats if gate is problematic (default: 2)
  --namespace eidf018ns          Kubernetes namespace
  --poll-sec N                   Poll interval passed to runner (default: env POLL_SEC or 20)
  --timeout-sec N                Timeout passed to runner (default: env TIMEOUT_SEC or 2400)
  --no-promote-threshold-pct X   Threshold for overlap/waitsome/boundary gates (default: 2.0)
  --problem-action-regex REGEX   Regex on gate recommended_action indicating more repeats

Behavior:
  1) Run dual-queue repeats with --repeats BASE_REPEATS for all pairs.
  2) Run gate summary for the selected campaign.
  3) If any target pair is problematic (status in {needs-retest,inconclusive}
     or recommended_action matches PROBLEM_ACTION_REGEX), rerun only those
     pairs with --repeats BASE_REPEATS+EXTRA_REPEATS and gate again.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign)
      CAMPAIGN="$2"
      shift 2
      ;;
    --pairs)
      PAIR_LIST="$2"
      shift 2
      ;;
    --base-repeats)
      BASE_REPEATS="$2"
      shift 2
      ;;
    --extra-repeats)
      EXTRA_REPEATS="$2"
      shift 2
      ;;
    --namespace)
      NS="$2"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --no-promote-threshold-pct)
      NO_PROMOTE_THRESHOLD_PCT="$2"
      shift 2
      ;;
    --problem-action-regex)
      PROBLEM_ACTION_REGEX="$2"
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

configure_campaign() {
  case "$CAMPAIGN" in
    overlap)
      RUNNER_SCRIPT="tools/run_overlap_dualqueue_repeats.sh"
      GATE_JSON="logs/overlap_gate_summary_repeats.json"
      GATE_CMD=(
        python3 tools/overlap_gate.py
        --logs-root logs
        --dir-glob "awave-cw2-overlap-ab-*-r*"
        --no-promote-threshold-pct "$NO_PROMOTE_THRESHOLD_PCT"
        --json-out "$GATE_JSON"
        --md-out "logs/overlap_gate_summary_repeats.md"
      )
      ;;
    mpimode)
      RUNNER_SCRIPT="tools/run_mpimode_dualqueue_repeats.sh"
      GATE_JSON="logs/mpimode_gate_summary_repeats.json"
      GATE_CMD=(
        python3 tools/mpimode_gate.py
        --logs-root logs
        --dir-glob "awave-cw2-mpimode-ab-*-r*"
        --json-out "$GATE_JSON"
        --md-out "logs/mpimode_gate_summary_repeats.md"
      )
      ;;
    mpistack)
      RUNNER_SCRIPT="tools/run_mpistack_dualqueue_repeats.sh"
      GATE_JSON="logs/mpistack_gate_summary_repeats.json"
      GATE_CMD=(
        python3 tools/mpistack_gate.py
        --logs-root logs
        --dir-glob "awave-cw2-mpistack-ab-*-r*"
        --json-out "$GATE_JSON"
        --md-out "logs/mpistack_gate_summary_repeats.md"
      )
      ;;
    waitsome)
      RUNNER_SCRIPT="tools/run_waitsome_dualqueue_repeats.sh"
      GATE_JSON="logs/waitsome_gate_summary_repeats.json"
      GATE_CMD=(
        python3 tools/waitsome_gate.py
        --logs-root logs
        --dir-glob "awave-cw2-waitsome-ab-*-r*"
        --no-promote-threshold-pct "$NO_PROMOTE_THRESHOLD_PCT"
        --json-out "$GATE_JSON"
        --md-out "logs/waitsome_gate_summary_repeats.md"
      )
      ;;
    boundary)
      RUNNER_SCRIPT="tools/run_boundary_dualqueue_repeats.sh"
      GATE_JSON="logs/boundary_gate_summary_repeats.json"
      GATE_CMD=(
        python3 tools/boundary_gate.py
        --logs-root logs
        --dir-glob "awave-cw2-boundary-ab-*-r*"
        --no-promote-threshold-pct "$NO_PROMOTE_THRESHOLD_PCT"
        --json-out "$GATE_JSON"
        --md-out "logs/boundary_gate_summary_repeats.md"
      )
      ;;
    *)
      echo "Unsupported campaign: ${CAMPAIGN}" >&2
      exit 2
      ;;
  esac
}

run_repeats() {
  local repeats=$1
  local pairs=$2
  log "run repeats campaign=${CAMPAIGN} repeats=${repeats} pairs=${pairs}"
  TIMEOUT_SEC="$TIMEOUT_SEC" POLL_SEC="$POLL_SEC" bash "$RUNNER_SCRIPT" \
    --repeats "$repeats" \
    --pairs "$pairs" \
    --namespace "$NS"
}

run_gate() {
  log "run gate campaign=${CAMPAIGN}"
  "${GATE_CMD[@]}" >/tmp/${CAMPAIGN}_gate_run.out
}

find_problem_pairs() {
  python3 - "$GATE_JSON" "$PAIR_LIST" "$PROBLEM_ACTION_REGEX" <<'PY'
import json
import re
import sys

gate_json, pair_csv, action_regex = sys.argv[1:4]
targets = {p.strip() for p in pair_csv.split(",") if p.strip()}
rx = re.compile(action_regex)

doc = json.load(open(gate_json, "r", encoding="utf-8"))
problem = []
for row in doc.get("groups", []):
    pair = f"{row.get('gpu')}-{row.get('gpus')}g"
    if pair not in targets:
        continue
    status = str(row.get("status", ""))
    action = str(row.get("recommended_action", ""))
    if status in {"needs-retest", "inconclusive"} or rx.search(action):
        problem.append(pair)

ordered = []
seen = set()
for p in problem:
    if p not in seen:
        ordered.append(p)
        seen.add(p)

print(",".join(ordered))
PY
}

print_final_snapshot() {
  python3 - "$GATE_JSON" "$PAIR_LIST" <<'PY'
import json
import sys

gate_json, pair_csv = sys.argv[1:3]
targets = {p.strip() for p in pair_csv.split(",") if p.strip()}
doc = json.load(open(gate_json, "r", encoding="utf-8"))

for row in doc.get("groups", []):
    pair = f"{row.get('gpu')}-{row.get('gpus')}g"
    if pair not in targets:
        continue
    gain = row.get("gain_steady_pct")
    gain_txt = "nan" if gain is None else f"{gain:.3f}"
    print(
        f"{pair} status={row.get('status')} pairs={row.get('pairs')} "
        f"gain_steady_pct={gain_txt} action={row.get('recommended_action')}"
    )
PY
}

main() {
  if [[ -z "$CAMPAIGN" ]]; then
    echo "--campaign is required" >&2
    usage
    exit 2
  fi

  configure_campaign

  mkdir -p logs
  RUN_LOG="logs/${CAMPAIGN}_dualqueue_adaptive.log"
  : > "$RUN_LOG"

  log "start campaign=${CAMPAIGN} ns=${NS} base_repeats=${BASE_REPEATS} extra_repeats=${EXTRA_REPEATS} pairs=${PAIR_LIST}"

  run_repeats "$BASE_REPEATS" "$PAIR_LIST"
  run_gate

  local problem_pairs
  problem_pairs=$(find_problem_pairs)

  if [[ -n "$problem_pairs" ]]; then
    local total_repeats
    total_repeats=$((BASE_REPEATS + EXTRA_REPEATS))
    log "problematic pairs detected: ${problem_pairs}; rerun to repeats=${total_repeats}"
    run_repeats "$total_repeats" "$problem_pairs"
    run_gate
  else
    log "no problematic pairs after base repeats"
  fi

  log "final group snapshot"
  print_final_snapshot | tee -a "$RUN_LOG"
  log "done gate_json=${GATE_JSON}"
  echo "$GATE_JSON"
}

main "$@"