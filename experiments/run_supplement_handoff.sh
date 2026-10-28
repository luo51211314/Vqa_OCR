#!/usr/bin/env bash
set -uo pipefail
OLD_PGID="$1"
ACTIVE_DRIVER="$2"
cd "$(dirname "$0")/.."
REPO=$(pwd)
LOG="$REPO/results_repro/20261018_e67003c/_night/supplement_handoff.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

driver_cmd=$(tr '\0' ' ' <"/proc/$ACTIVE_DRIVER/cmdline" 2>/dev/null || true)
case "$driver_cmd" in
  *run_config.py*main_table.yaml*mydatavqa*) ;;
  *) log "refuse: PID $ACTIVE_DRIVER is not MyData main driver: $driver_cmd"; exit 2 ;;
esac
log "armed fast handoff for MyData main driver PID=$ACTIVE_DRIVER, old PGID=$OLD_PGID"

while kill -0 "$ACTIVE_DRIVER" 2>/dev/null; do
  state=$(ps -o stat= -p "$ACTIVE_DRIVER" 2>/dev/null | tr -d ' ' || true)
  case "$state" in
    Z*|"") break ;;
  esac
  sleep 0.2
done

log "MyData main driver finished; stopping old queue process group"
kill -STOP -- "-$OLD_PGID" 2>/dev/null || true
kill -KILL -- "-$OLD_PGID" 2>/dev/null || true
sleep 1
log "starting short-first queue"
exec bash experiments/run_supplement_night.sh
