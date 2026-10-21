#!/usr/bin/env bash
# Speed program S0 on a Colab runtime. Supervisor-only (research/colab_cli owns allocation,
# export, stop). Phase 1: coverage round alone; phase 2: S0_PROCESSES concurrent
# single-thread timed rounds. World payloads stay in a work dir outside the export root;
# only the receipt and one bounded tarball of the anchor round's worlds are exported.
set -euo pipefail

[[ -n "${COLAB_CLI_ROOT:-}" ]] || { echo "S0 colab entrypoint is supervisor-only" >&2; exit 2; }
processes="${S0_PROCESSES:-5}"
plain="${S0_PLAIN_ROUNDS:-3}"
instrumented="${S0_INSTRUMENTED_ROUNDS:-2}"
deadline="${S0_DEADLINE_SECONDS:-2700}"
label="${S0_HOST_LABEL:-colab-l4}"

input_root="${COLAB_CLI_ROOT}/t34-inputs"
output_root="${COLAB_CLI_ROOT}/output"
work_root="${COLAB_CLI_ROOT}/s0-work"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu PYTHONUNBUFFERED=1
mkdir -p "${output_root}"
python -m research.t34_exit_b8.public_inputs --catalog "${COLAB_INPUT_DIR:?}/public-inputs.json" \
  --out-dir "${input_root}"
export SNAKE_CKPT_DIR="${input_root}/checkpoints"
python -m research.speed_s0_baseline.harness \
  --checkpoint "${input_root}/B8_s3_final.pth" \
  --output-dir "${output_root}/s0" \
  --work-dir "${work_root}" \
  --host-label "${label}" \
  --processes "${processes}" \
  --plain-rounds "${plain}" \
  --instrumented-rounds "${instrumented}" \
  --deadline-seconds "${deadline}"
python - "${work_root}" "${output_root}/s0/anchor-worlds.tar.gz" <<'PY'
import hashlib
import sys
import tarfile
from pathlib import Path

work, target = Path(sys.argv[1]), Path(sys.argv[2])
anchor = work / "phase-0" / "proc-00" / "round-00-coverage"
with tarfile.open(target, "x:gz") as archive:
    archive.add(anchor, arcname="round-00-coverage")
if target.stat().st_size > 32 * 1024 * 1024:
    raise SystemExit("anchor tarball exceeds 32 MiB")
digest = hashlib.sha256(target.read_bytes()).hexdigest()
target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n")
target.with_name(target.name + ".complete").write_text("complete\n")
PY
