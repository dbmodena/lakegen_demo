#!/usr/bin/env bash
log_path=/data/pneuma/nyc/bootstrap.log
resource_log_path=/data/pneuma/nyc/bootstrap.resources.log

printf '\n[monitor] bootstrap started %s\n' "$(date --iso-8601=seconds)" | tee -a "$log_path"

.venv-pneuma/bin/python scripts/bootstrap_pneuma.py \
  --portal nyc \
  --out-path /data/pneuma/nyc \
  --index-name lakegen \
  --openai-base-url http://127.0.0.1:11434/v1 \
  > >(tee -a "$log_path") 2>&1 &

bootstrap_pid=$!
printf '[monitor] pid=%s\n' "$bootstrap_pid" | tee -a "$log_path"

while kill -0 "$bootstrap_pid" 2>/dev/null; do
  timestamp=$(date --iso-8601=seconds)
  memory=$(awk '/^(VmRSS|VmHWM|VmSwap):/ {printf "%s=%s%s ", $1, $2, $3}' "/proc/$bootstrap_pid/status" 2>/dev/null)
  printf '%s pid=%s %s\n' "$timestamp" "$bootstrap_pid" "$memory" >> "$resource_log_path"
  sleep 30
done

wait "$bootstrap_pid"
bootstrap_status=$?
printf '[monitor] bootstrap exited %s status=%s\n' \
  "$(date --iso-8601=seconds)" "$bootstrap_status" | tee -a "$log_path"
exit "$bootstrap_status"
