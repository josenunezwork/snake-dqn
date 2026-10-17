#!/bin/zsh
set -u
cd /Users/josenunez/Projects/ml/snake-dqn-t7
RUNROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t7-phase-r-b8-20261016
/usr/bin/caffeinate -is ./venv/bin/python research/redesign_t7_phase_r_20261016/phase_r_t7.py run --procs 6 > "$RUNROOT/run.log" 2>&1
run_status=$?
printf '%s\n' "$run_status" > "$RUNROOT/run.exit"
if [ "$run_status" -eq 0 ]; then
  ./venv/bin/python -I research/redesign_t7_phase_r_20261016/phase_r_t7.py audit > "$RUNROOT/audit.log" 2>&1
  printf '%s\n' "$?" > "$RUNROOT/audit.exit"
fi
