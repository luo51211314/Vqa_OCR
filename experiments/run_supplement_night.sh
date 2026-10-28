#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
RUN_DIR="20261018_e67003c"
NIGHT_DIR="$REPO/results_repro/$RUN_DIR/_night"
STATE_DIR="$NIGHT_DIR/supplement_state"
mkdir -p "$NIGHT_DIR" "$STATE_DIR"
exec 9>"$STATE_DIR/queue.lock"
flock -n 9 || exit 0

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="$NIGHT_DIR/supplement_night.log"
FAILED=0
SNAPSHOT_DIR="$NIGHT_DIR/supplement_snapshot"
mkdir -p "$SNAPSHOT_DIR"
cp experiments/configs/main_table.yaml \
   experiments/configs/ablation_template_full.yaml \
   experiments/configs/ablation_budget.yaml \
   experiments/configs/ablation_order.yaml \
   experiments/configs/ablation_l2.yaml "$SNAPSHOT_DIR/"
CODE_SNAPSHOT_DIR="$SNAPSHOT_DIR/code"
mkdir -p "$CODE_SNAPSHOT_DIR"
cp --parents \
   experiments/run_config.py \
   experiments/run_supplement_night.sh \
   experiments/run_live_table_watch.sh \
   experiments/run_mydata_expand_handoff.sh \
   experiments/ablations/run_budget.sh \
   experiments/ablations/run_order.sh \
   experiments/ablations/run_l2.sh \
   scripts/make_ablation_tables.py \
   scripts/ablation_ci_tex.py \
   scripts/make_order_ablation_table.py \
   expert/expert_manager.py Algorithm/ocr_filters.py "$CODE_SNAPSHOT_DIR/"
sha256sum /root/autodl-tmp/dataset/MyDataVQA/MyDataVQA_test.parquet \
  >"$SNAPSHOT_DIR/mydatavqa.sha256"
git rev-parse HEAD >"$SNAPSHOT_DIR/git_commit.txt"
git status --short >"$SNAPSHOT_DIR/git_status.txt"
git diff >"$SNAPSHOT_DIR/tracked_changes.patch"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Keep TeX/CSV artifacts synchronized with each completed metrics file. flock
# inside the watcher makes this safe when a manually started watcher exists.
bash experiments/run_live_table_watch.sh "$RUN_DIR" \
  >>"$NIGHT_DIR/live_table_watch.launch.log" 2>&1 &

run_stage() {
  local name="$1"; shift
  log "START $name: $*"
  if "$@" >>"$NIGHT_DIR/${name}.driver.log" 2>&1; then
    touch "$STATE_DIR/${name}.done"
    rm -f "$STATE_DIR/${name}.failed"
    log "DONE $name"
  else
    local rc=$?
    touch "$STATE_DIR/${name}.failed"
    FAILED=1
    log "FAILED $name rc=$rc (queue continues)"
  fi
}

while pgrep -f "python -m main_inference" >/dev/null 2>&1; do
  log "waiting for existing inference process"
  sleep 20
done

run_stage 01_mydata_main python experiments/run_config.py \
  experiments/configs/main_table.yaml --run --run-dir "$RUN_DIR" \
  --datasets mydatavqa --retries 1
run_stage 02_mydata_template python experiments/run_config.py \
  experiments/configs/ablation_template_full.yaml --run --run-dir "$RUN_DIR" \
  --datasets mydatavqa --retries 1
run_stage 03_mydata_budget python experiments/run_config.py \
  experiments/configs/ablation_budget.yaml --run --run-dir "$RUN_DIR" \
  --datasets mydatavqa --retries 1
run_stage 04_mydata_order python experiments/run_config.py \
  experiments/configs/ablation_order.yaml --run --run-dir "$RUN_DIR" \
  --datasets mydatavqa --retries 1
run_stage 05_mydata_l2 python experiments/run_config.py \
  experiments/configs/ablation_l2.yaml --run --run-dir "$RUN_DIR" \
  --datasets mydatavqa --retries 1
run_stage 06_mplug_template python experiments/run_config.py \
  experiments/configs/ablation_template_full.yaml --run --run-dir "$RUN_DIR" \
  --datasets chartqa,docvqa --only mplug --retries 1
run_stage 07_chartqa_l2 python experiments/run_config.py \
  experiments/configs/ablation_l2.yaml --run --run-dir "$RUN_DIR" \
  --datasets chartqa --only llava,qwen --retries 1
run_stage 08_mplug_budget python experiments/run_config.py \
  experiments/configs/ablation_budget.yaml --run --run-dir "$RUN_DIR" \
  --datasets docvqa --only mplug --retries 1
run_stage 09_mplug_l2 python experiments/run_config.py \
  experiments/configs/ablation_l2.yaml --run --run-dir "$RUN_DIR" \
  --datasets docvqa --only mplug --retries 1

run_stage 90_table3 python scripts/make_table3.py \
  --root "$REPO/results_repro/$RUN_DIR/main_table" --allow-missing
run_stage 91_ablation_tables python scripts/make_ablation_tables.py \
  --run-dir "$REPO/results_repro/$RUN_DIR" --kind all
run_stage 92_order_table python scripts/make_order_ablation_table.py \
  --run-dir "$REPO/results_repro/$RUN_DIR"
run_stage 93_offline_matrix python scripts/make_offline_matrix.py \
  --run-dir "$REPO/results_repro/$RUN_DIR"

if [ "$FAILED" -eq 0 ]; then
  touch "$STATE_DIR/all.done"
  rm -f "$STATE_DIR/all.failed"
  log "supplement queue completed"
  exit 0
fi
touch "$STATE_DIR/all.failed"
log "supplement queue completed with failures"
exit 1
