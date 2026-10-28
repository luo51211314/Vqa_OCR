#!/usr/bin/env bash
# Incrementally rebuild paper tables whenever a completed metrics file appears.
# This process is deliberately independent of the GPU queue and protected by
# flock, so restarting the queue/watcher cannot create concurrent TeX writers.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO=$(pwd)
RUN_DIR="${1:-20261018_e67003c}"
RESULT_ROOT="$REPO/results_repro/$RUN_DIR"
NIGHT_DIR="$RESULT_ROOT/_night"
STATE_DIR="$NIGHT_DIR/supplement_state"
LOG="$NIGHT_DIR/live_table_watch.log"
mkdir -p "$NIGHT_DIR" "$STATE_DIR"

exec 8>"$STATE_DIR/live_table_watch.lock"
flock -n 8 || exit 0

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train

log() {
  echo "[$(date '+%F %T')] $*" >>"$LOG"
}

signature() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo missing
    return
  fi
  find "$dir" -maxdepth 1 -type f -name '*_metrics.json' \
    -printf '%T@ %s %f\n' 2>/dev/null | sort | sha256sum | awk '{print $1}'
}

refresh_main() {
  log "refresh main_table/table3"
  if python scripts/make_table3.py \
      --root "$RESULT_ROOT/main_table" --allow-missing >>"$LOG" 2>&1; then
    log "done main_table/table3"
  else
    log "warning: main_table/table3 refresh failed"
  fi
}

refresh_ablation() {
  local kind="$1"
  log "refresh ablation kind=$kind"
  if python scripts/make_ablation_tables.py \
      --run-dir "$RESULT_ROOT" --kind "$kind" >>"$LOG" 2>&1; then
    log "done ablation kind=$kind"
  else
    log "warning: ablation kind=$kind refresh incomplete"
  fi
}

refresh_order() {
  log "refresh ablation order"
  if python scripts/make_order_ablation_table.py \
      --run-dir "$RESULT_ROOT" >>"$LOG" 2>&1; then
    log "done ablation order"
  else
    log "warning: ablation order refresh incomplete"
  fi
}

# Force one complete refresh at startup. Newly added YAML rows then appear as
# TBD even before their first metrics file exists.
last_main=""
last_template=""
last_budget=""
last_order=""
last_l2=""
log "watcher started run_dir=$RUN_DIR"

while :; do
  current=$(signature "$RESULT_ROOT/main_table")
  if [ "$current" != "$last_main" ]; then
    refresh_main
    last_main="$current"
  fi

  current=$(signature "$RESULT_ROOT/ablation_template_full")
  if [ "$current" != "$last_template" ]; then
    refresh_ablation template_full
    last_template="$current"
  fi

  current=$(signature "$RESULT_ROOT/ablation_budget")
  if [ "$current" != "$last_budget" ]; then
    refresh_ablation budget
    last_budget="$current"
  fi

  current=$(signature "$RESULT_ROOT/ablation_order")
  if [ "$current" != "$last_order" ]; then
    refresh_order
    last_order="$current"
  fi

  current=$(signature "$RESULT_ROOT/ablation_l2")
  if [ "$current" != "$last_l2" ]; then
    refresh_ablation l2
    last_l2="$current"
  fi

  if [ -f "$STATE_DIR/all.done" ]; then
    log "all.done observed; final refresh"
    refresh_main
    refresh_ablation all
    refresh_order
    log "watcher exits"
    exit 0
  fi
  sleep 15
done
