#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "??: $0 <??tag> [?? train.py ??...]" >&2
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
  echo "# ?????$TAG"
  echo
  echo "- ???$(date '+%Y-%m-%d %H:%M:%S')"
  echo "- ?????$ROOT"
  echo "- ?????$TRAIN_DIR"
  echo "- ???"
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
echo "??????PID=$(cat "$TRAIN_DIR/train.pid")"
echo "???$LOG_DIR/stdout.log"
