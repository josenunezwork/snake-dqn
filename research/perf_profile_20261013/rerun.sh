#!/usr/bin/env bash
# perf-profile (item 9) re-run: the whole measurement set of REPORT.md, one process (group) at
# a time, for a quiet Mac or a CUDA box. Everything lands in OUT_DIR; nothing else is written.
#
#   research/perf_profile_20261013/rerun.sh DEVICE OUT_DIR
#     DEVICE   mps | cuda | cpu (learner and actor device)
#     OUT_DIR  must not exist (train_m3c refuses an existing --out)
#   env: INIT (ego2s-b/bv8 student .pth), DEMO_DATA (m2b data dir), PY (python),
#        TRANSITIONS (default 300000), WORLDS (128), ACTORS (2), STEPS (all|train|bench)
#
# Steps: (1) B8 training run under M3C_PROFILE=pyinstrument (warm 30 s), (2) the same run
# unprofiled (throughput control), (3) single-process actor profiles, short and long bodies,
# (4) hero upload bench, (5) learner sync bench, (6) profiler overhead on the actor,
# (7) analyze_profiles over (1) and (3). Run from the repository root.
set -euo pipefail
DEVICE=${1:?device (mps|cuda|cpu)}
OUT=${2:?out dir}
PY=${PY:-./venv/bin/python}
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008
INIT=${INIT:-$ART/student_m2b_final.pth}
DEMO_DATA=${DEMO_DATA:-$ART/data}
TRANSITIONS=${TRANSITIONS:-300000}
WORLDS=${WORLDS:-128}
ACTORS=${ACTORS:-2}
STEPS=${STEPS:-all}
mkdir -p "$OUT"
B8=(--reward mass --mass-kappa 1.0 --kill-scale 0 --gamma-end 0.998 --gamma-hold 2000000
    --gamma-anneal-end 8000000 --mixed-frac 0.25 --anchor-mode v8margin --anchor-hold 8000000
    --anchor-floor-at 15000000 --onpolicy-labels --obs-version bv8 --label-frac 1.0
    --v61-backend numba --learner-fast)

train() {  # name, profile mode (off | pyinstrument | cprofile)
  local name=$1 mode=$2
  uptime > "$OUT/$name.uptime"
  if [ "$mode" = off ]; then
    env -u M3C_PROFILE "$PY" research/redesign_m3c_20261008/train_m3c.py "${B8[@]}" \
      --worlds "$WORLDS" --actors "$ACTORS" --device "$DEVICE" --init "$INIT" \
      --demo-data "$DEMO_DATA" --no-guard --seed 0 --transitions "$TRANSITIONS" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  else
    M3C_PROFILE=$mode M3C_PROFILE_WARM_S=30 "$PY" research/redesign_m3c_20261008/train_m3c.py \
      "${B8[@]}" --worlds "$WORLDS" --actors "$ACTORS" --device "$DEVICE" --init "$INIT" \
      --demo-data "$DEMO_DATA" --no-guard --seed 0 --transitions "$TRANSITIONS" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  fi
  uptime >> "$OUT/$name.uptime"
}

bench_profile() {  # name, grow_to, warm, frames
  uptime > "$OUT/$1.uptime"
  "$PY" research/perf_actor_20261013/bench_actor.py --profile --b8 --profiler pyinstrument \
    --profile-dir "$OUT/$1" --configs numba --worlds "$WORLDS" --devices "$DEVICE" \
    --grow-to "$2" --warm "$3" --frames "$4" --init "$INIT" > "$OUT/$1.log" 2>&1
  uptime >> "$OUT/$1.uptime"
}

if [ "$STEPS" = all ] || [ "$STEPS" = train ]; then
  train train_pyinstrument pyinstrument
  train train_off off
fi
if [ "$STEPS" = all ] || [ "$STEPS" = bench ]; then
  bench_profile bench_short 0 60 400
  bench_profile bench_long 1200 400 300
  "$PY" research/perf_profile_20261013/bench_hero_h2d.py --device "$DEVICE" \
    --out "$OUT/hero_h2d.json" > "$OUT/hero_h2d.log" 2>&1
  "$PY" research/perf_profile_20261013/bench_learner_sync.py --devices "$DEVICE" \
    --out "$OUT/learner_sync.json" > "$OUT/learner_sync.log" 2>&1
  DEVICE=$DEVICE PY=$PY research/perf_profile_20261013/bench_overhead.sh "$OUT/overhead" 2 \
    > "$OUT/overhead.log" 2>&1
  "$PY" research/perf_profile_20261013/summarize_overhead.py "$OUT/overhead" >> "$OUT/overhead.log"
fi
dirs=()
for d in train_pyinstrument bench_short bench_long; do
  [ -d "$OUT/$d" ] && dirs+=("$OUT/$d")
done
if [ ${#dirs[@]} -gt 0 ]; then
  "$PY" research/perf_profile_20261013/analyze_profiles.py "${dirs[@]}" \
    --json "$OUT/analysis.json" > "$OUT/analysis.md"
fi
for d in train_pyinstrument train_off; do
  [ -f "$OUT/$d/log.jsonl" ] && "$PY" research/perf_profile_20261013/telemetry_split.py \
    "$OUT/$d" --step 100000 --worlds "$WORLDS" >> "$OUT/throughput.txt"
done
echo "done: $OUT (analysis.md, analysis.json, throughput.txt)"
