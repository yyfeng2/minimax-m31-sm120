#!/bin/bash
# 后台监控 vLLM 启动：直到 ready / 错误 / 超时
LOG=/tmp/m31_boot_monitor.log
STATUS=/tmp/m31_boot_status.txt
: > "$LOG"
for i in $(seq 1 240); do
  (
    docker logs --tail 60 minimax-m31 2>&1
  ) > "$LOG"
  if grep -aq "OpenAI server is ready" "$LOG"; then
    echo "READY round=$i $(date +%H:%M:%S)" > "$STATUS"
    exit 0
  fi
  if grep -aqE "Traceback|ERROR.*(Failed|failed)|raise ValueError|RuntimeError|exited with" "$LOG"; then
    echo "ERROR round=$i $(date +%H:%M:%S)" > "$STATUS"
    grep -aE "Traceback|Error|error" "$LOG" | tail -6 >> "$STATUS"
    exit 1
  fi
  echo "WAIT round=$i $(date +%H:%M:%S) $(tail -1 "$LOG" | cut -c1-150)" >> "$STATUS"
  sleep 12
done
echo "TIMEOUT" > "$STATUS"
