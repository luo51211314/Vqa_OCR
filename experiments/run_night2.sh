#!/usr/bin/env bash
# 第二阶段补跑：等待第一阶段 night 编排结束后执行
#  1) main_table 补齐（llava docvqa ocr：逐样本 OOM 跳过版代码，断点续跑只跑缺格）
#  2) ablation_template 24 格（dev300 子集修复后）
#  3) 刷新两张论文表
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
WAIT_PID="${1:?用法: run_night2.sh <第一阶段编排器PID>}"
RUN_DIR="20261018_e67003c"
NIGHT_DIR="$REPO/results_repro/$RUN_DIR/_night"
mkdir -p "$NIGHT_DIR"

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*" | tee -a "$NIGHT_DIR/night2.log"; }

log "night2: 等待第一阶段编排器 PID=$WAIT_PID ..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
log "night2: 第一阶段结束，开始补跑"

# 双保险：GPU 上还有残余 python 推理进程时等它退出
while pgrep -f "python -m main_inference" >/dev/null; do sleep 20; done

log "night2: [1/3] 主表补齐（断点续跑，仅缺格）"
python experiments/run_config.py experiments/configs/main_table.yaml --run \
    --run-dir "$RUN_DIR" --retries 1 >>"$NIGHT_DIR/main_table2.driver.log" 2>&1
log "night2: 主表补齐结束 rc=$?"

log "night2: [2/3] 模板消融 24 格（dev300）"
python experiments/run_config.py experiments/configs/ablation_template.yaml --run \
    --run-dir "$RUN_DIR" --retries 1 >>"$NIGHT_DIR/ablation_template2.driver.log" 2>&1
log "night2: 模板消融结束 rc=$?"

log "night2: [3/3] 刷新论文表"
python scripts/make_table3.py --root "$REPO/results_repro/$RUN_DIR/main_table" \
    --allow-missing >>"$NIGHT_DIR/table3.log" 2>&1

touch /tmp/vqa_night2_done
log "night2: 全部结束"
