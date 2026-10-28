#!/usr/bin/env bash
# Switch to the updated short-first queue immediately after the active stage.
set -uo pipefail
cd "$(dirname "$0")/.."

TARGET_PID="${1:?usage: run_mydata_expand_handoff.sh <driver-pid> <old-pgid>}"
OLD_PGID="${2:?usage: run_mydata_expand_handoff.sh <driver-pid> <old-pgid>}"
RUN_DIR="20261018_e67003c"
LOG="results_repro/$RUN_DIR/_night/mydata_expand_handoff.log"
mkdir -p "$(dirname "$LOG")"

log() {
  echo "[$(date '+%F %T')] $*" >>"$LOG"
}

log "armed target_pid=$TARGET_PID old_pgid=$OLD_PGID"
while kill -0 "$TARGET_PID" 2>/dev/null; do
  sleep 0.2
done

log "active template stage finished; stopping old queue"
kill -TERM -- "-$OLD_PGID" 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 -- "-$OLD_PGID" 2>/dev/null || break
  sleep 0.2
done
if kill -0 -- "-$OLD_PGID" 2>/dev/null; then
  log "old queue still present; sending KILL"
  kill -KILL -- "-$OLD_PGID" 2>/dev/null || true
fi

while pgrep -f "python -m main_inference" >/dev/null 2>&1; do
  sleep 0.2
done
log "starting updated queue with MyData budget/order/L2 first"
exec bash experiments/run_supplement_night.sh
