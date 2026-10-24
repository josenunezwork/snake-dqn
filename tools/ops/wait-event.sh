#!/bin/bash
# Block until there are events in ops/events.log beyond the persisted read cursor (max 3 h); print them and advance
# the cursor. Using a cursor (not "lines at start") means events written between two calls are never missed.
OPS=${OPS_DIR:-/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops}; mkdir -p "$OPS"; touch "$OPS/events.log"
HERE=$(cd "$(dirname "$0")" && pwd)
CONSUMER=${OPS_EVENT_CONSUMER:-operator}
CUR=$OPS/events.cursor.$CONSUMER; [ -f "$CUR" ] || echo 0 > "$CUR"
end=$(( $(date +%s) + 10800 ))
while [ $(date +%s) -lt $end ]; do
  # Task notices are at-least-once until the consumer explicitly acknowledges
  # them with task_journal.py ack; printing is never an acknowledgment.
  notices=$(python3 "$HERE/task_journal.py" --ops "$OPS" deliver "$CONSUMER" 2>/dev/null)
  if [ -n "$notices" ] && [ "$notices" != '[]' ]; then printf '%s\n' "$notices"; exit 0; fi
  start=$(cat "$CUR"); now=$(wc -l < "$OPS/events.log" | tr -d ' ')
  if [ "$now" -gt "$start" ]; then tail -n +$((start+1)) $OPS/events.log | head -n $((now-start)); echo $now > $CUR; exit 0; fi
  sleep 30
done
echo "NO EVENT in 3 h"
