#!/usr/bin/env bash
# =====================================================================
# 全集模板敏感性检验 · 队尾接跑 watcher
#
# 不打断任何在跑任务：阻塞等待 /tmp/vqa_tail_done（接管器跑完
# main -> template(dev300) -> budget -> l2 后写入），确认 GPU 清空，
# 然后跑 ablation_template_full.yaml（8 格，断点续跑/失败重试1次），
# 最后生成 table_template_full 并写 /tmp/vqa_tplfull_done。
#
# 用法: bash experiments/run_template_full_watch.sh
# 在 tmux 窗口 hub:tplfull 中常驻；所有输出见 _night/template_full_watch.log
# =====================================================================
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
RUN_DIR="20261018_e67003c"
NIGHT_DIR="$REPO/results_repro/$RUN_DIR/_night"
mkdir -p "$NIGHT_DIR"
LOG="$NIGHT_DIR/template_full_watch.log"

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "watch: 等待 /tmp/vqa_tail_done（budget+l2 队列结束），不打断在跑任务"
while [ ! -f /tmp/vqa_tail_done ]; do sleep 10; done
log "watch: tail sentinel 已出现"

# 双保险：等所有推理进程退出（接管器写 sentinel 后理论上已无）
while pgrep -f "python -m main_inference" >/dev/null 2>&1; do
  log "  仍有推理进程，等待 GPU 清空 ..."; sleep 10
done
sleep 5
log "watch: GPU 已清空，开始全集模板 8 格"

python experiments/run_config.py experiments/configs/ablation_template_full.yaml \
    --run --run-dir "$RUN_DIR" --retries 1 >>"$NIGHT_DIR/template_full.driver.log" 2>&1
rc=$?
log "watch: run_config 结束 rc=$rc，刷新全集模板表"

python scripts/make_ablation_tables.py \
    --run-dir "$REPO/results_repro/$RUN_DIR" --kind template_full \
    >>"$NIGHT_DIR/autotable_template_full.log" 2>&1

touch /tmp/vqa_tplfull_done
log "watch: 全部结束 sentinel=/tmp/vqa_tplfull_done"
