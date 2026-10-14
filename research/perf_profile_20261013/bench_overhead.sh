#!/usr/bin/env bash
# Profiler overhead on the B8 actor (perf-profile, 2026-10-13): the same bench_actor cell
# unprofiled, under pyinstrument and under cProfile, interleaved for ROUNDS rounds.
# One CPU process at a time. Usage (from the repo root):
#   research/perf_profile_20261013/bench_overhead.sh OUT_DIR [ROUNDS] [GROW_TO] [WARM]
set -euo pipefail
OUT=${1:?out dir}
ROUNDS=${2:-2}
GROW=${3:-0}
WARM=${4:-30}
PY=${PY:-./venv/bin/python}
mkdir -p "$OUT"
common=(--b8 --configs numba --worlds 128 --devices "${DEVICE:-mps}" --frames 150 --warm "$WARM"
        --grow-to "$GROW")
for r in $(seq 1 "$ROUNDS"); do
  $PY research/perf_actor_20261013/bench_actor.py "${common[@]}" \
    --out "$OUT/off_r$r.json" | grep -v '^wrote'
  for prof in pyinstrument cprofile; do
    $PY research/perf_actor_20261013/bench_actor.py "${common[@]}" --profile \
      --profiler "$prof" --profile-dir "$OUT/${prof}_r$r" | grep '^CELL' \
      | sed "s/^CELL /$prof r$r /" | cut -c1-400
  done
done
