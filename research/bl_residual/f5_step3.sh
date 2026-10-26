#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?requires qualified supervisor}"
: "${COLAB_INPUT_DIR:?requires uploaded sealed inputs}"
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR"
F5_STEP3_LOCK=research/bl_residual/f5_adoption_environment_20261026_v6/requirements-f5-l4-cu126.lock
if [ "$(sha256sum "$F5_STEP3_LOCK" | awk '{print $1}')" != 2577ed9dd9fe34468f5562262def7bccb352c9c0bdce06b293bb4b12a0ea5301 ]; then
  echo 'F5 step-3 requirements lock mismatch' >&2; exit 2
fi
F5_STEP3_VENV="$COLAB_CLI_ROOT/f5-cp3123"
uv python install 3.12.3
uv venv --python 3.12.3 --managed-python --seed "$F5_STEP3_VENV"
export F5_WHEELHOUSE="$COLAB_CLI_ROOT/f5-wheelhouse"
export F5_INSTALL_REPORT="$COLAB_CLI_ROOT/f5-install-report.json"
"$F5_STEP3_VENV/bin/python" -m pip download --require-hashes --only-binary=:all: \
  --dest "$F5_WHEELHOUSE" --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple -r "$F5_STEP3_LOCK"
"$F5_STEP3_VENV/bin/python" -m pip install --force-reinstall --no-index \
  --find-links "$F5_WHEELHOUSE" --require-hashes --no-compile \
  --report "$F5_INSTALL_REPORT" -r "$F5_STEP3_LOCK"
if [ "${SNAKE_COLAB_CANARY:-}" = "1" ]; then
  F5_STEP3_SITE=$("$F5_STEP3_VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  {
    printf '%s\n' "$(pwd)"
    printf '%s\n' "import importlib; importlib.import_module('research.colab_cli.canary_observer').install()"
  } > "$F5_STEP3_SITE/snake_colab_canary.pth"
fi
exec nice -n 15 "$F5_STEP3_VENV/bin/python" -m research.bl_residual.f5_step3_driver \
  --control "$COLAB_INPUT_DIR/control.json" --fixtures "$COLAB_INPUT_DIR/timing-fixtures.json" \
  --inputs "$COLAB_INPUT_DIR" --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --output "$COLAB_CLI_ROOT/output" "--${F5_STEP3_MODE:?requires qualify or measure}"
