#!/usr/bin/env bash
# =====================================================================
# 过夜全量实验编排（断点续跑 + 失败自动重试 + OOM 跳过不中断）
#
# 队列顺序（重要的在前）：
#   0) 等待当前主表驱动器结束
#   1) main_table       12 格（已完成自动跳过；失败格自动重试 1 次）
#   2) ablation_template 24 格（dev300 子集，便宜，模板选择只碰 dev）
#   3) ablation_order   18 格（random/ros/coordinate，全集）
#   4) ablation_budget   8 格（DocVQA 全集 token 预算）
#   5) ablation_l2       8 格（DocVQA 全集 L2 有损规则）
# 每个配置独立失败不影响后续；重复执行本脚本会自动跳过已有产物。
#
# 用法: bash experiments/run_all_night.sh [WAIT_PID]
# =====================================================================
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)

WAIT_PID="${1:-}"
RUN_DIR="20261018_e67003c"
NIGHT_DIR="$REPO/results_repro/$RUN_DIR/_night"
mkdir -p "$NIGHT_DIR"
SENTINEL="/tmp/vqa_night_done"
rm -f "$SENTINEL"

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate vqa_train
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*" | tee -a "$NIGHT_DIR/night.log"; }

if [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; then
  log "等待当前主表驱动器 PID=$WAIT_PID 结束 ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "主表驱动器已结束，开始过夜全量队列"
fi

# 0) dev300 子集（模板消融前置，已存在会幂等重写同内容）
log "[0/6] 生成 dev300 子集"
python scripts/make_dev_subset.py >>"$NIGHT_DIR/subsets.log" 2>&1 \
  && log "  子集就绪" || log "  !! 子集生成失败（模板消融将受影响），继续"

run_cfg() {
  local cfg="$1"; local name="$2"
  log "==== 开始 $name ($cfg) ===="
  python experiments/run_config.py "experiments/configs/$cfg" --run \
      --run-dir "$RUN_DIR" --retries 1 >>"$NIGHT_DIR/${name}.driver.log" 2>&1
  log "==== $name 结束 rc=$? ===="
}

# 1) 主表（含失败格重试）
run_cfg main_table.yaml main_table
# 主表一结束立刻刷新论文表
python scripts/make_table3.py \
  --root "$REPO/results_repro/$RUN_DIR/main_table" --allow-missing \
  >>"$NIGHT_DIR/table3.log" 2>&1 && log "Table 3 已刷新"

# 2-5) 消融（按时间成本从低到高排队）
run_cfg ablation_template.yaml ablation_template
run_cfg ablation_order.yaml ablation_order
run_cfg ablation_budget.yaml ablation_budget
run_cfg ablation_l2.yaml ablation_l2

# 最终再刷一次论文表
python scripts/make_table3.py \
  --root "$REPO/results_repro/$RUN_DIR/main_table" --allow-missing \
  >>"$NIGHT_DIR/table3.log" 2>&1
python scripts/make_ablation_tables.py \
  --run-dir "$REPO/results_repro/$RUN_DIR" --kind all \
  >>"$NIGHT_DIR/ablation_tables.log" 2>&1 || true
python scripts/make_order_ablation_table.py \
  --run-dir "$REPO/results_repro/$RUN_DIR" \
  >>"$NIGHT_DIR/order_table.log" 2>&1 || true

touch "$SENTINEL"
log "全部队列结束，sentinel=$SENTINEL"
