#!/bin/bash
set -euo pipefail
EVAL_ROOT=/data_b/qzluo/workspace/wlx/Repro-verl-agent/evaluations/gradus-webshop-polarity-panel-20261111
R2_ROOT=/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-polarity-from115-20261111
PY=/data_b/qzluo/workspace/wlx/Repro-verl-agent/.venv/bin/python
source "$R2_ROOT/environment.sh"
export PYTHONPATH="$EVAL_ROOT:$PYTHONPATH"
export VLLM_CACHE_ROOT="$EVAL_ROOT/vllm_cache"
export TMPDIR=/tmp/ws-polarity-panel-driver-20261111
mkdir -p "$VLLM_CACHE_ROOT" "$TMPDIR"
cd "$EVAL_ROOT"
case "${1:-}" in
  freeze)
    CUDA_VISIBLE_DEVICES='' "$PY" "$EVAL_ROOT/freeze_reuse.py"
    CUDA_VISIBLE_DEVICES='' "$PY" "$EVAL_ROOT/verify.py" --tag aligned_150
    CUDA_VISIBLE_DEVICES='' "$PY" "$EVAL_ROOT/verify.py" --tag gigpo_150 --compare;;
  run)
    CUDA_VISIBLE_DEVICES='' "$PY" "$EVAL_ROOT/merge.py" --tag r2_150
    "$PY" "$EVAL_ROOT/evaluate.py" --tag r2_150
    CUDA_VISIBLE_DEVICES='' "$PY" "$EVAL_ROOT/verify.py" --tag r2_150 --compare;;
  *) echo 'Usage: launch.sh {freeze|run}'; exit 64;;
esac
