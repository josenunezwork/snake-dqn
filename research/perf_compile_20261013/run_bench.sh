#!/usr/bin/env bash
# perf-compile (2026-10-13): re-runnable torch.compile benchmarks for the ego2s trainer
# (train_m3c --compile-learner / --compile-actor). Run on a QUIET machine; the target is a
# CUDA pod (the Mac numbers in results/ were taken while training runs shared the machine).
#
#   research/perf_compile_20261013/run_bench.sh DEVICE [OUT_DIR] [NOTE]
#     DEVICE   cuda | mps | cpu
#     OUT_DIR  default research/perf_compile_20261013/results/<device>-<UTC stamp>
#     NOTE     provenance stored in every JSON (default "quiet-<device>")
#   env: PY (python, default ./venv/bin/python or python), DRY_RUN=1 (print the commands)
#
# Writes (one JSON each; every compiled cell carries its compile status - a cell whose
# "fallback" is set timed EAGER code, e.g. the learner on MPS in torch 2.9.1):
#   learner.json  learner updates/s: eager t1 / t1_bf16 / t1_k2 and their compiled twins
#                 <config>@<mode> (speedup_vs_eager_twin), + the pre-perf baseline
#   drift.json    compiled training vs eager t1_fp32 after N updates: parameter / Q drift,
#                 greedy agreement, next to the t1_fp32_init_1ulp chaos floor (any 1e-7
#                 relative difference grows to this under Adam) - compiled is NOT bitwise
#   actor.json    actor frames/s and the hero-forward ms/frame, eager vs compiled, at
#                 64 / 128 / 256 worlds
# Modes: cuda: default, reduce-overhead (CUDA graphs), max-autotune-no-cudagraphs;
#        mps / cpu: default (reduce-overhead only adds CUDA graphs on CUDA).
set -euo pipefail

DEV=${1:?usage: run_bench.sh cuda|mps|cpu [OUT_DIR] [NOTE]}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${2:-$REPO/research/perf_compile_20261013/results/$DEV-$(date -u +%Y%m%dT%H%M%SZ)}
NOTE=${3:-quiet-$DEV}
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$REPO/venv/bin/python" ]]; then PY="$REPO/venv/bin/python"; else PY=python; fi
fi

case "$DEV" in
  cuda)
    MODES=default,reduce-overhead,max-autotune-no-cudagraphs
    LEARN=(--devices cuda --updates 100 --rounds 7 --warm 10)
    DRIFT=(--devices cuda --drift-updates 200 --compile default,reduce-overhead)
    ;;
  mps)
    MODES=default
    LEARN=(--devices mps --updates 30 --rounds 5 --warm 5)
    DRIFT=(--devices mps --drift-updates 200 --compile default)
    ;;
  cpu)
    MODES=default
    LEARN=(--devices cpu --updates-cpu 4 --rounds-cpu 3 --warm-cpu 2)
    DRIFT=(--devices cpu --drift-updates 40 --compile default)
    ;;
  *)
    echo "DEVICE must be cuda, mps or cpu" >&2
    exit 2
    ;;
esac

run() {
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" != 1 ]]; then (cd "$REPO" && "$@"); fi
}

[[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p "$OUT"
run "$PY" research/perf_learner_20261013/bench_learner.py "${LEARN[@]}" \
  --configs baseline,t1,t1_bf16,t1_k2 --compile "$MODES" \
  --note "$NOTE" --out "$OUT/learner.json"
run "$PY" research/perf_learner_20261013/bench_learner.py --drift "${DRIFT[@]}" \
  --drift-variants t1_fp32_plain_adam,t1_fp32_init_1ulp,t1_bf16 \
  --note "$NOTE" --out "$OUT/drift.json"
run "$PY" research/perf_actor_20261013/bench_actor.py --configs numba \
  --worlds 64,128,256 --devices "$DEV" --compile-actor "off,$MODES" \
  --frames 300 --warm 30 --note "$NOTE" --out "$OUT/actor.json"
echo "results: $OUT"
