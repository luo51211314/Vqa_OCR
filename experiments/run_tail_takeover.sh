#!/usr/bin/env bash
# =====================================================================
# 尾部接管编排（不打断正在运行的 order 驱动器）
#
# 背景：旧编排器 run_all_night.sh 的顺序是 order -> budget -> l2，
# 便宜且关键的模板消融被排到了最后。本脚本：
#   1) 等 ablation_order 驱动器（ORDER_PID）自然跑完（绝不打断）
#   2) 立即停掉旧编排器（OLD_PID），抢在它启动 budget/l2 之前
#      并清扫任何已漏网启动的 budget/l2 驱动器或推理进程
#   3) 按新顺序串行：main_table补缺 -> template(dev300) -> budget -> l2
# 所有驱动器均断点续跑（已有产物自动跳过），失败重试 1 次，互不阻断。
#
# 用法: bash run_tail_takeover.sh <ORDER_DRIVER_PID> <OLD_ORCH_PID>
# =====================================================================
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)

ORDER_PID="${1:?需要 order 驱动器 PID}"
OLD_PID="${2:?需要旧编排器 PID}"
RUN_DIR="20261018_e67003c"
NIGHT_DIR="$REPO/results_repro/$RUN_DIR/_night"
mkdir -p "$NIGHT_DIR"

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*" | tee -a "$NIGHT_DIR/tail.log"; }

log "takeover: 等待 order 驱动器 PID=$ORDER_PID 跑完（不打断）"
while kill -0 "$ORDER_PID" 2>/dev/null; do sleep 3; done
log "takeover: order 已结束，立即接管，阻止旧编排器启动 budget/l2"

# 1) 停掉旧编排脚本本体（wrapper bash -c 会退化为交互 shell，不影响）
kill "$OLD_PID" 2>/dev/null && log "  已停旧编排器 $OLD_PID"
sleep 2

# 2) 清扫漏网的 budget/l2 驱动器（不含 main/template/order）
for cfg in ablation_budget ablation_l2; do
  pkill -f "run_config.py experiments/configs/${cfg}.yaml" 2>/dev/null \
    && log "  已杀漏网驱动器 $cfg"
done
# 清扫已进入这两个目录的推理子进程（按 output_dir 精确匹配，避免误伤）
pkill -f "main_inference.*--output_dir $REPO/results_repro/$RUN_DIR/ablation_budget" 2>/dev/null \
  && log "  已杀 budget 推理进程"
pkill -f "main_inference.*--output_dir $REPO/results_repro/$RUN_DIR/ablation_l2" 2>/dev/null \
  && log "  已杀 l2 推理进程"

# 3) 等 GPU 上所有推理进程退出（进程被杀显存自动释放）
log "  等待 GPU 清空 ..."
while pgrep -f "python -m main_inference" >/dev/null 2>/dev/null; do sleep 3; done
sleep 5
log "  GPU 已清空，开始新顺序队列"

run_cfg() {
  local cfg="$1"; local name="$2"
  log "==== 开始 $name ($cfg) ===="
  python experiments/run_config.py "experiments/configs/$cfg" --run \
      --run-dir "$RUN_DIR" --retries 1 >>"$NIGHT_DIR/${name}_tail.driver.log" 2>&1
  log "==== $name 结束 rc=$? ===="
  # 主表或模板结束都顺手刷新 Table 3
  if [ "$name" = "main_table" ]; then
    python scripts/make_table3.py --root "$REPO/results_repro/$RUN_DIR/main_table" \
      --allow-missing >>"$NIGHT_DIR/table3_tail.log" 2>&1 && log "  Table 3 已刷新"
  fi
}

# 新顺序：主表补缺（1 格，先补全 Table 3）-> 模板(dev300,便宜) -> 预算 -> L2
run_cfg main_table.yaml main_table
run_cfg ablation_template.yaml ablation_template
run_cfg ablation_budget.yaml ablation_budget
run_cfg ablation_l2.yaml ablation_l2

python scripts/make_table3.py --root "$REPO/results_repro/$RUN_DIR/main_table" \
  --allow-missing >>"$NIGHT_DIR/table3_tail.log" 2>&1

touch /tmp/vqa_tail_done
log "takeover: 全部结束 sentinel=/tmp/vqa_tail_done"
