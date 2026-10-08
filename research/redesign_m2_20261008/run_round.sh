#!/bin/zsh
# Usage: run_round.sh ROUND FIRST_CHUNK LAST_CHUNK STEP [extra gen_data.py args...]
# One queue (one process at a time, 1 thread, nice 10). Two queues with interleaved chunks
# (STEP=2) give the <= 2 processes x 1 thread the owner approved. gen_data.py refuses a
# chunk (exit 3) when AC / lid / thermal guard is not ok; the queue then waits and retries.
cd "$(dirname "$0")/../.."
ROUND=$1; FIRST=$2; LAST=$3; STEP=$4; shift 4
OUT=${M2_DATA:-/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2-20261008/data}
for ((c=FIRST; c<=LAST; c+=STEP)); do
  [[ -f $OUT/round-$ROUND/chunk-$(printf %04d $c).json ]] && continue
  while true; do
    nice -n 10 ./venv/bin/python research/redesign_m2_20261008/gen_data.py --round $ROUND --chunk $c --out $OUT "$@" 2>&1 | grep -v "Using CPU" | tail -1
    rc=${pipestatus[1]}
    [[ $rc == 3 ]] && { echo "chunk $c refused by guard; waiting"; sleep 120; continue; }
    echo "chunk $c rc=$rc"; break
  done
done
echo "QUEUE_DONE round=$ROUND first=$FIRST"
