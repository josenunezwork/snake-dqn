#!/bin/zsh
# Usage: run_ni.sh OUT ARM [ARM...]  -- one queue: each ARM over the three mixes, in order.
# ni_check.py refuses (exit 3) when the AC / lid / thermal guard is not ok; wait and retry.
cd "$(dirname "$0")/../.."
OUT=$1; shift
for arm in "$@"; do for mix in frozen scripted mixed; do
  [[ -f $OUT/$arm-$mix.json ]] && continue
  while true; do
    nice -n 10 ./venv/bin/python research/redesign_m2_20261008/ni_check.py run --arm $arm --mix $mix --out $OUT 2>&1 | grep -v "Using CPU" | tail -1
    rc=${pipestatus[1]}
    [[ $rc == 3 ]] && { echo "refused by guard; waiting"; sleep 120; continue; }
    echo "$arm $mix rc=$rc"; break
  done
done; done
echo NI_QUEUE_DONE
