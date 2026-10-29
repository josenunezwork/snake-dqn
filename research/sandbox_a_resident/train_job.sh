#!/usr/bin/env bash
# TRAIN-01 adapter only; provider operations belong to the qualified supervisor.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${SANDBOX_A_SOURCE_COMMIT:?pinned source required}"
: "${SANDBOX_A_CONTROL_JSON:?frozen training control required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST='8.0;9.0'
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
nice -n 15 python -c 'import os; from pathlib import Path; Path(os.environ["COLAB_CLI_ROOT"], "train-control.json").write_text(os.environ["SANDBOX_A_CONTROL_JSON"])'
diagnostics="$COLAB_CLI_ROOT/output/training/stall-watchdog"

seal_stall_diagnostics() {
  # This parent shell remains alive after faulthandler's os._exit(1), so it can
  # publish the native dump for the supervisor without importing torch.
  local payload
  for payload in "$diagnostics/stall-watchdog.json" "$diagnostics/faulthandler-stacks.log"; do
    if [[ ! -f "$payload" || -e "$payload.sha256" || -e "$payload.complete" ]]; then
      continue
    fi
    nice -n 15 python - "$payload" <<'PY' || true
import hashlib
import sys
from pathlib import Path

payload = Path(sys.argv[1])
if not payload.is_file() or payload.stat().st_size > 1024 * 1024:
    raise SystemExit("refusing absent or oversized stall diagnostic")
sidecar = Path(f"{payload}.sha256")
complete = Path(f"{payload}.complete")
if sidecar.exists() or complete.exists():
    raise SystemExit("stall diagnostic already has a sidecar")
digest = hashlib.file_digest(payload.open("rb"), "sha256").hexdigest()
sidecar.write_text(f"{digest}  {payload.name}\n")
complete.write_text("complete\n")
PY
  done
  printf 'STALL_WATCHDOG_DIAGNOSTICS sealed_dir=%s\n' "$diagnostics" >&2
}

on_exit() {
  local status=$?
  trap - EXIT
  seal_stall_diagnostics
  exit "$status"
}
trap on_exit EXIT

child_pid=""
on_usr1() {
  if [[ -n "$child_pid" ]]; then
    kill -USR1 "$child_pid" 2>/dev/null || true
  fi
}
on_term() {
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}
trap on_usr1 USR1
trap on_term TERM

set +e
nice -n 15 python -m research.sandbox_a_resident.train \
  --checkpoint "$COLAB_INPUT_DIR/b8-s3.pth" \
  --frp3 "$COLAB_INPUT_DIR/frp3-s12.pth" \
  --champion "$COLAB_INPUT_DIR/champion.pth" \
  --control "$COLAB_CLI_ROOT/train-control.json" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --input-root "$COLAB_INPUT_DIR" \
  --output "$COLAB_CLI_ROOT/output/training" --device cuda \
  --stall-timeout-seconds "${SANDBOX_A_STALL_TIMEOUT_SECONDS:-900}" \
  --stall-diagnostics "$diagnostics" &
child_pid=$!
while true; do
  wait "$child_pid"
  status=$?
  if ! kill -0 "$child_pid" 2>/dev/null; then
    break
  fi
done
set -e
exit "$status"
