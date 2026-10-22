#!/bin/bash
# Persistent watcher daemon (tmux session 'watchd'). Appends events to ops/events.log:
# job ended (EXIT=), Sol PID finished, Colab under-used (<3 assignments for >10 min), Mac idle (>15 min).
OPS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops; mkdir -p $OPS; touch $OPS/watch-logs.txt $OPS/watch-pids.txt $OPS/seen.txt $OPS/events.log
export PATH=$HOME/.local/bin:$PATH
colab_low_since=""; mac_idle_since=""
ev(){ echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $OPS/events.log; }
while true; do
  while read -r f name; do [ -n "$f" ] && grep -q '^EXIT=' "$f" 2>/dev/null && ! grep -qx "log:$f" $OPS/seen.txt && { echo "log:$f" >> $OPS/seen.txt; ev "JOB_ENDED $name $(grep '^EXIT=' $f) $(grep -o 'INCOMPLETE[^;]*' $f | tail -1 | cut -c1-120)"; }; done < $OPS/watch-logs.txt
  while read -r pid name; do [ -n "$pid" ] && ! ps -p $pid >/dev/null 2>&1 && ! grep -qx "pid:$pid" $OPS/seen.txt && { echo "pid:$pid" >> $OPS/seen.txt; ev "SOL_DONE $name"; }; done < $OPS/watch-pids.txt
  n=$(colab --auth oauth2 sessions 2>/dev/null | grep -c Hardware)
  okuntil=$(cat $OPS/colab-idle-ok-until 2>/dev/null || echo 0)
  if [ "${n:-0}" -lt 3 ] && [ $(date +%s) -ge "${okuntil:-0}" ]; then [ -z "$colab_low_since" ] && colab_low_since=$(date +%s); [ $(( $(date +%s) - colab_low_since )) -ge 600 ] && { ev "COLAB_UNDERUSED sessions=$n for >=10min"; colab_low_since=$(date +%s); }; else colab_low_since=""; fi
  busy=$(ps -Ao pcpu,comm | awk '$2 ~ /[Pp]ython/ && $1>50' | wc -l | tr -d ' ')
  if [ "$busy" = "0" ]; then [ -z "$mac_idle_since" ] && mac_idle_since=$(date +%s); [ $(( $(date +%s) - mac_idle_since )) -ge 900 ] && { ev "MAC_IDLE no python worker >50% CPU for >=15min"; mac_idle_since=$(date +%s); }; else mac_idle_since=""; fi
  sleep 60
done
