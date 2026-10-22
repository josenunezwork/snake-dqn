#!/bin/bash
# Block until a new line appears in ops/events.log after the call (max 3 h); print it. Claude runs this in background.
OPS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops; touch $OPS/events.log
start=$(wc -l < $OPS/events.log); end=$(( $(date +%s) + 10800 ))
while [ $(date +%s) -lt $end ]; do now=$(wc -l < $OPS/events.log); [ $now -gt $start ] && { tail -n +$((start+1)) $OPS/events.log; exit 0; }; sleep 30; done
echo "NO EVENT in 3 h"
