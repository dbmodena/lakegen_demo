#!/usr/bin/env bash
set -u

portal=${PNEUMA_PORTAL:-${1:-nyc}}
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! "$portal" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  printf 'Invalid portal name: %s\n' "$portal" >&2
  exit 2
fi

out_path=${PNEUMA_OUT_PATH:-/data/pneuma/$portal}
index_name=${PNEUMA_INDEX_NAME:-lakegen}
openai_base_url=${PNEUMA_OPENAI_BASE_URL:-http://127.0.0.1:11434/v1}
provider=${PNEUMA_PROVIDER:-openai}
mkdir -p "$out_path"

log_path="$out_path/bootstrap.log"
resource_log_path="$out_path/bootstrap.resources.log"

printf '\n[monitor] bootstrap started %s\n' "$(date --iso-8601=seconds)" | tee -a "$log_path"

.venv-pneuma/bin/python scripts_pneuma/bootstrap_pneuma.py \
  --portal "$portal" \
  --out-path "$out_path" \
  --index-name "$index_name" \
  --provider "$provider" \
  --openai-base-url "$openai_base_url" \
  "$@" \
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
