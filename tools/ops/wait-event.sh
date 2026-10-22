#!/bin/bash
# Block until there are events in ops/events.log beyond the persisted read cursor (max 3 h); print them and advance
# the cursor. Using a cursor (not "lines at start") means events written between two calls are never missed.
OPS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops; touch $OPS/events.log
CUR=$OPS/events.cursor; [ -f $CUR ] || wc -l < $OPS/events.log | tr -d ' ' > $CUR
end=$(( $(date +%s) + 10800 ))
while [ $(date +%s) -lt $end ]; do
  start=$(cat $CUR); now=$(wc -l < $OPS/events.log | tr -d ' ')
  if [ "$now" -gt "$start" ]; then tail -n +$((start+1)) $OPS/events.log | head -n $((now-start)); echo $now > $CUR; exit 0; fi
  sleep 30
done
echo "NO EVENT in 3 h"
