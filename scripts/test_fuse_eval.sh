#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <checkpoint_path> <result_tag> [extra test.py args...]" >&2
  exit 1
fi
CKPT="$1"
RESULT_TAG="$2"
shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/u104754251515/miniconda3/bin/python}"
OUT_DIR="$ROOT/outputs_retest/$RESULT_TAG"
RES_DIR="$ROOT/results_m/retest_$RESULT_TAG"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$RES_DIR" "$LOG_DIR"
TEST_CMD=("$PYTHON" "$ROOT/test.py" --loadckpt "$CKPT" --outdir "$OUT_DIR" "$@")
FUSE_CMD=("$PYTHON" "$ROOT/fusion-normal.py" --outdir "$OUT_DIR")
EVAL_CMD=("$PYTHON" "$ROOT/matlab.py" --plyPath "$OUT_DIR" --resultPath "$RES_DIR")
cat > "$OUT_DIR/RUN_INFO.md" <<INFO
# Test, fusion, and local evaluation run: $RESULT_TAG

- Started: $(date '+%Y-%m-%d %H:%M:%S')
- Checkpoint: $CKPT
- Output dir: $OUT_DIR
- Result dir: $RES_DIR
- Test log: $LOG_DIR/test.log
- Fusion log: $LOG_DIR/fusion.log
- Eval log: $LOG_DIR/eval.log
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
echo "Test + fusion + local eval started. PID=$(cat "$OUT_DIR/workflow.pid")"
echo "Output: $OUT_DIR"
