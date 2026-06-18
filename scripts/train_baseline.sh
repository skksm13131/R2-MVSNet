#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_tag> [extra train.py args...]" >&2
  exit 1
fi
TAG="$1"
shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/u104754251515/miniconda3/bin/python}"
TRAIN_DIR="$ROOT/checkpoints/$TAG"
LOG_DIR="$TRAIN_DIR/logs"
mkdir -p "$LOG_DIR"
CMD=("$PYTHON" "$ROOT/train.py" --logdir "$TRAIN_DIR" "$@")
{
  echo "# Training run: $TAG"
  echo
  echo "- Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- Repo: $ROOT"
  echo "- Checkpoint dir: $TRAIN_DIR"
  echo "- Command:"
  printf '  `'; printf '%q ' "${CMD[@]}"; echo '`'
} > "$TRAIN_DIR/RUN_INFO.md"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "cd $(printf '%q' "$ROOT")"
  printf '%q ' "${CMD[@]}"; echo
} > "$TRAIN_DIR/run_train.sh"
chmod +x "$TRAIN_DIR/run_train.sh"
nohup "${CMD[@]}" > "$LOG_DIR/stdout.log" 2>&1 &
echo $! > "$TRAIN_DIR/train.pid"
echo "Training started. PID=$(cat "$TRAIN_DIR/train.pid")"
echo "Log: $LOG_DIR/stdout.log"
