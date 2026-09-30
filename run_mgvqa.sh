#!/usr/bin/env bash
# MG-VQA-Bench evaluation wrapper (uses the caller's Python environment).
# Usage: ./run_mgvqa.sh --mode direct,perception,agentic --model MODEL [OPTIONS]
# Options match python -m mgvqa --help; this wrapper additionally loops modes.
# Examples:
#   ./run_mgvqa.sh --dataset /data/MG-VQA-Bench --mode direct --workers 4
#   ./run_mgvqa.sh --dataset /data/MG-VQA-Bench --mode direct,perception,agentic
#   ./run_mgvqa.sh --dataset /data/MG-VQA-Bench --mode agentic --model azure/openai/gpt-6-astra --workers 4
#   ./run_mgvqa.sh --dataset /data/MG-VQA-Bench --mode direct --dry-run
set -euo pipefail
CALLER_DIR="$PWD"
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-python}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
MODE="agentic"
ARGS=()
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 && -n "$2" ]] || { echo '--mode requires a value' >&2; exit 2; }
      MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; shift ;;
    --dataset|--snapshot)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "$1 requires a path" >&2; exit 2; }
      dataset="$2"
      [[ "$dataset" = /* || "$dataset" = '~'* ]] || dataset="$CALLER_DIR/$dataset"
      ARGS+=(--dataset "$dataset"); shift 2 ;;
    --dataset=*|--snapshot=*)
      dataset="${1#*=}"
      [[ -n "$dataset" ]] || { echo '--dataset requires a path' >&2; exit 2; }
      [[ "$dataset" = /* || "$dataset" = '~'* ]] || dataset="$CALLER_DIR/$dataset"
      ARGS+=(--dataset "$dataset"); shift ;;
    -h|--help)
      echo 'MG-VQA-Bench: --mode accepts a comma-list in this shell wrapper.'
      "$PYTHON" -m mgvqa --help
      exit 0 ;;
    --dry-run) DRY_RUN=1; ARGS+=("$1"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ -n "$MODE" && "$MODE" != ,* && "$MODE" != *, && "$MODE" != *,,* ]] || {
  echo 'Invalid mode list' >&2; exit 2;
}
IFS=',' read -r -a MODES <<< "$MODE"
# Validate every invocation before launching any paid run; dry-run also needs
# neither credentials nor the snapshot/model dependencies.
for mode in "${MODES[@]}"; do
  "$PYTHON" -m mgvqa --mode "$mode" "${ARGS[@]}" --dry-run > /dev/null
done
if [[ "$DRY_RUN" == 1 ]]; then
  for mode in "${MODES[@]}"; do
    "$PYTHON" -m mgvqa --mode "$mode" "${ARGS[@]}"
  done
  exit 0
fi
if [[ -f .env ]]; then set -a; source ./.env; set +a; fi
mkdir -p rollouts
for mode in "${MODES[@]}"; do
  log="rollouts/mgvqa_${mode}_$(date +%Y%m%d_%H%M%S)_launch.log"
  echo "MODE=$mode | log=$log"
  start=$(date +%s)
  set +e
  "$PYTHON" -m mgvqa --mode "$mode" "${ARGS[@]}" 2>&1 | tee "$log"
  statuses=("${PIPESTATUS[@]}")
  set -e
  for status in "${statuses[@]}"; do
    if [[ "$status" -ne 0 ]]; then
      echo "MODE=$mode failed (exit $status); see $log" >&2
      exit "$status"
    fi
  done
  elapsed=$(( $(date +%s) - start ))
  outdir=$(grep '^MGVQA_RUN_DIR=' "$log" | head -1 | cut -d= -f2- || true)
  if [[ -n "$outdir" && -d "$outdir" ]]; then
    printf 'mode=%s\nwall_time_seconds=%s\n' "$mode" "$elapsed" > "$outdir/WALL_TIME.txt"
    mv "$log" "$outdir/launch.log"
  fi
  echo "MODE=$mode completed in ${elapsed}s. Accuracy and non-answer counts: summary.json"
done
