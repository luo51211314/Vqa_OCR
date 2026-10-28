#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
RUN_DIR="20261018_e67003c"
STATE_DIR="$REPO/results_repro/$RUN_DIR/_night/supplement_state"
LOG="$REPO/results_repro/$RUN_DIR/_night/supplement_watch.log"
mkdir -p "$STATE_DIR"
attempt=0
max_attempts=2
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

while [ ! -f "$STATE_DIR/all.done" ]; do
  if ! pgrep -f '[r]un_supplement_night.sh' >/dev/null 2>&1; then
    if [ "$attempt" -ge "$max_attempts" ]; then
      log "queue still incomplete after $attempt restarts; stopping watcher"
      touch "$STATE_DIR/watcher.blocked"
      exit 1
    fi
    attempt=$((attempt + 1))
    log "queue absent/incomplete; restart $attempt/$max_attempts"
    bash experiments/run_supplement_night.sh >>"$LOG" 2>&1 &
  fi
  sleep 60
done
log "all.done observed; watcher exits"
