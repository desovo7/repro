#!/bin/bash
# Copy this file to a PREPARED isolated run directory. check is read-only.
# start/resume requires a reviewed config and is not executed by code generation.
set -euo pipefail
RUNDIR=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-check}
GPUS=${GPUS:-0,1}
PY=/data_b/qzluo/workspace/wlx/Repro-verl-agent/.venv/bin/python
JVM='-Xmx128m -Xms16m -XX:+UseSerialGC'
UNIT="gradus-ws-$(basename "$RUNDIR")"
check_resources() {
  local gpu free count available problems=0
  for gpu in ${GPUS//,/ }; do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    count=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$free" -lt 76000 ] || [ "$count" -ne 0 ]; then
      echo "GPU $gpu not ready: $free MiB free, $count compute processes"; problems=1
    fi
  done
  available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$available" -lt 200 ]; then echo "Host has $available GiB available; needs 200"; problems=1; fi
  return "$problems"
}
case "$MODE" in
  check) check_resources; exit $?;;
  --exec)
    source "$RUNDIR/environment.sh"
    export CUDA_VISIBLE_DEVICES=$GPUS JAVA_TOOL_OPTIONS="$JVM"
    export PYTHONPATH="$RUNDIR:$PYTHONPATH"
    cd "$RUNDIR/source"
    exec "$PY" -m webshop.entry "$RUNDIR/config.json";;
  start|resume)
    [ -f "$RUNDIR/config.json" ] && [ -d "$RUNDIR/webshop" ]
    if systemctl --user is-active --quiet "$UNIT"; then echo 'Already running'; exit 1; fi
    if [ "$MODE" = start ] && [ -d "$RUNDIR/checkpoints" ]; then echo 'Existing state: use resume'; exit 1; fi
    if [ "$MODE" = resume ] && [ ! -d "$RUNDIR/checkpoints" ]; then echo 'No run to resume'; exit 1; fi
    exec 9>"$RUNDIR/.launch.lock"
    flock -n 9
    check_resources
    systemctl --user reset-failed "$UNIT" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" -p Type=exec -p KillMode=control-group -p Restart=no -p LimitCORE=0 -p TimeoutStopSec=30s -p RuntimeMaxSec=24h \
      -p StandardOutput=append:"$RUNDIR/train.log" -p StandardError=append:"$RUNDIR/train.log" \
      --setenv=GPUS="$GPUS" --setenv="JAVA_TOOL_OPTIONS=$JVM" /bin/bash "$RUNDIR/launch.sh" --exec
    systemctl --user reset-failed "$UNIT-monitor" 2>/dev/null || true
    systemd-run --user --unit="$UNIT-monitor" -p Restart=no \
      "$PY" -I "$RUNDIR/monitor.py" "$UNIT" "$RUNDIR/resources.jsonl";;
  *) echo 'Usage: launch.sh {check|start|resume}'; exit 64;;
esac
