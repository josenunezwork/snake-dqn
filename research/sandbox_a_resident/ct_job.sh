#!/usr/bin/env bash
# C/T (t34-cf-1028) runtime adapter; provider operations belong to the qualified supervisor.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${SANDBOX_A_SOURCE_COMMIT:?pinned source required}"
: "${SANDBOX_CT_CONTROL_JSON:?frozen C/T control required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.0
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export GPU_SIM_VECTOR_GEOMETRY=native GPU_SIM_VECTOR_FLOOD=static GPU_SIM_V8_INDEXED=0
# Record the actual post-bootstrap VM package; continuation admission accepts
# only its closed version-line grammar and the literal observed warnings.
nice -n 15 uv --version
nice -n 15 python -c 'import os; from pathlib import Path; Path(os.environ["COLAB_CLI_ROOT"], "ct-control.json").write_text(os.environ["SANDBOX_CT_CONTROL_JSON"])'
diagnostics="$COLAB_CLI_ROOT/output/ct/stall-watchdog"

seal_stall_diagnostics() {
  local payload
  for payload in "$diagnostics/stall-watchdog.json" "$diagnostics/faulthandler-stacks.log" "$COLAB_CLI_ROOT/output/ct/progress-history.log"; do
    if [[ ! -f "$payload" || -e "$payload.sha256" || -e "$payload.complete" ]]; then
      continue
    fi
    nice -n 15 python - "$payload" <<'PY' || true
import hashlib
import sys
from pathlib import Path

payload = Path(sys.argv[1])
limit = (4 if payload.name == "progress-history.log" else 1) * 1024 * 1024
if not payload.is_file() or payload.stat().st_size > limit:
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
# The runtime signals the entire process group. Keep this shell alive while
# Python handles its SIGUSR1 fence; otherwise EXIT seals watchdog bytes that
# the still-running child can mutate. Bash defers this trap until the foreground
# command returns, so the EXIT publication follows the child's watchdog.close().
trap ':' USR1

set +e
nice -n 15 python -m research.sandbox_a_resident.ct_entrypoint runner \
  --checkpoint "$COLAB_INPUT_DIR/b8-s3.pth" \
  --frp3 "$COLAB_INPUT_DIR/frp3-s12.pth" \
  --champion "$COLAB_INPUT_DIR/champion.pth" \
  --control "$COLAB_CLI_ROOT/ct-control.json" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --input-root "$COLAB_INPUT_DIR" \
  --output "$COLAB_CLI_ROOT/output/ct" --device cuda \
  --stall-timeout-seconds "${SANDBOX_A_STALL_TIMEOUT_SECONDS:-900}" \
  --stall-diagnostics "$diagnostics"
status=$?
set -e
exit "$status"
