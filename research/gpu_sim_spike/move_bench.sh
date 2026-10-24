#!/usr/bin/env bash
# Qualified-supervisor runtime entrypoint; no provider calls here.
set -euo pipefail
: "${COLAB_CLI_ROOT:?MOVE runs only under the qualified supervisor}"
: "${COLAB_INPUT_DIR:?MOVE requires uploaded inputs}"
: "${MOVE_BENCH_SECONDS:?manifest must bind wall time}"
: "${MOVE_BENCH_DEVICE:?manifest must explicitly declare cpu or cuda}"
: "${MOVE_BENCH_LEDGER_PATH:?manifest must bind its verified ledger}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 MAX_JOBS=2
if [[ "$MOVE_BENCH_DEVICE" == cuda ]]; then
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" == *A100* ]]
  python -c 'import torch; assert torch.cuda.is_available()'
fi
input_root="$COLAB_INPUT_DIR"
if [[ "${BL_PUBLIC_INPUTS:-0}" == 1 ]]; then
  python -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR"
  input_root="$COLAB_INPUT_DIR/public-checkpoints"
fi
checkpoint="$input_root/B8_s3_final.pth"
export SNAKE_CKPT_DIR="$input_root/checkpoints"
[[ -f "$checkpoint" ]] || { echo "missing B8 checkpoint" >&2; exit 2; }
mkdir -p "$COLAB_CLI_ROOT/output"
# One test process at low priority. The log is immutable after process exit.
nice -n 15 python -m pytest -q research/gpu_sim_spike/test_move_cuda.py \
  > "$COLAB_CLI_ROOT/move-native-tests.log" 2>&1
python - <<'PY'
import os
from pathlib import Path
from research.gpu_sim_spike import move_units
root = Path(os.environ["COLAB_CLI_ROOT"])
p = root / "move-native-tests.log"
meta = move_units.publish_bytes(root / "output/move-native-tests.log", p.read_bytes())
move_units.publish_json(root / "output/move-native-tests.json", {
    "schema": "gpu-sim-spike-move-native-tests/v1", "log": "move-native-tests.log",
    "device": os.environ["MOVE_BENCH_DEVICE"], "status": "PASS", **meta,
})
PY
args=(--device "$MOVE_BENCH_DEVICE" --checkpoint "$checkpoint"
  --ledger "$COLAB_INPUT_DIR/$MOVE_BENCH_LEDGER_PATH"
  --output-dir "$COLAB_CLI_ROOT/output" --seconds "$MOVE_BENCH_SECONDS"
  --repeats "${MOVE_BENCH_REPEATS:-3}" --warmups "${MOVE_BENCH_WARMUPS:-3}")
[[ "${MOVE_BENCH_MODE:-}" != --smoke ]] || args+=(--smoke)
[[ "${MOVE_BENCH_MODE:-}" != --canary ]] || args+=(--canary)
nice -n 15 python -m research.gpu_sim_spike.move_bench "${args[@]}"
for f in move-bench.json verified-ledger.json cpu-parity.json; do
  [[ -s "$COLAB_CLI_ROOT/output/$f" && -s "$COLAB_CLI_ROOT/output/$f.sha256" && -s "$COLAB_CLI_ROOT/output/$f.complete" ]]
done
