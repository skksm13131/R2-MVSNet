#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "用法: $0 <checkpoint路径> <结果tag> [额外 test.py 参数...]" >&2
  exit 1
fi
CKPT="$1"
RESULT_TAG="$2"
shift 2
TEST_ARGS=()
FUSE_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--fuse_fgdr_candidates" ]; then
    FUSE_ARGS+=(--use_fgdr_candidates)
  else
    TEST_ARGS+=("$arg")
  fi
done
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="${R2MVSNET_SHARED_ROOT:-/root/shared-nvme}"
EXPERIMENT_ROOT="${R2MVSNET_EXPERIMENT_ROOT:-$SHARED_ROOT/experiments/R2-MVSNet}"
PYTHON="${PYTHON:-/root/envs/r2mvsnet/bin/python}"
OUT_DIR="$EXPERIMENT_ROOT/outputs_retest/$RESULT_TAG"
RES_DIR="$EXPERIMENT_ROOT/results_m/retest_$RESULT_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$RES_DIR" "$LOG_DIR"
TEST_CMD=("$PYTHON" "$ROOT/test.py" --loadckpt "$CKPT" --outdir "$OUT_DIR" "${TEST_ARGS[@]}")
FUSE_CMD=("$PYTHON" "$ROOT/fusion-normal.py" --outdir "$OUT_DIR" "${FUSE_ARGS[@]}")
EVAL_CMD=("$PYTHON" "$ROOT/matlab.py" --plyPath "$OUT_DIR" --resultPath "$RES_DIR")
cat > "$OUT_DIR/RUN_INFO.md" <<INFO
# 测试、融合、本地评估任务: $RESULT_TAG

- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')
- Checkpoint: $CKPT
- 输出目录: $OUT_DIR
- 结果目录: $RES_DIR
- 测试日志: $LOG_DIR/test.log
- 融合日志: $LOG_DIR/fusion.log
- 评估日志: $LOG_DIR/eval.log
INFO
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "cd $(printf '%q' "$ROOT")"
  printf '%q ' "${TEST_CMD[@]}"; echo ' > '"$(printf '%q' "$LOG_DIR/test.log")"' 2>&1'
  printf '%q ' "${FUSE_CMD[@]}"; echo ' > '"$(printf '%q' "$LOG_DIR/fusion.log")"' 2>&1'
  printf '%q ' "${EVAL_CMD[@]}"; echo ' > '"$(printf '%q' "$LOG_DIR/eval.log")"' 2>&1'
} > "$OUT_DIR/run_workflow.sh"
chmod +x "$OUT_DIR/run_workflow.sh"
nohup "$OUT_DIR/run_workflow.sh" > "$LOG_DIR/workflow.stdout.log" 2>&1 &
echo $! > "$OUT_DIR/workflow.pid"
echo "测试 + 融合 + 本地评估已启动。PID=$(cat "$OUT_DIR/workflow.pid")"
echo "输出目录: $OUT_DIR"
