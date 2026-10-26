#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?requires qualified supervisor root}"
: "${COLAB_INPUT_DIR:?requires hash-bound uploaded F5 inputs}"
: "${F5_MODE:?requires capture, prepare, qualification or fit mode}"
: "${F5_DEVICE:?requires explicit cuda:0 device declaration}"
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR"
F5_LOCK_SHA256=2577ed9dd9fe34468f5562262def7bccb352c9c0bdce06b293bb4b12a0ea5301
F5_REQUIREMENTS_LOCK=${F5_REQUIREMENTS_LOCK:-research/bl_residual/f5_adoption_environment_20261026_v6/requirements-f5-l4-cu126.lock}
if [ "$(sha256sum "$F5_REQUIREMENTS_LOCK" | awk '{print $1}')" != "$F5_LOCK_SHA256" ]; then
  echo "F5 requirements lock digest mismatch" >&2; exit 2
fi
F5_VENV="$COLAB_CLI_ROOT/f5-cp3123"
uv python install 3.12.3
uv venv --python 3.12.3 --managed-python --seed "$F5_VENV"
export F5_WHEELHOUSE="$COLAB_CLI_ROOT/f5-wheelhouse"
export F5_INSTALL_REPORT="$COLAB_CLI_ROOT/f5-install-report.json"
"$F5_VENV/bin/python" -m pip download --require-hashes --only-binary=:all: --dest "$F5_WHEELHOUSE" \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple -r "$F5_REQUIREMENTS_LOCK"
"$F5_VENV/bin/python" -m pip install --force-reinstall --no-index --find-links "$F5_WHEELHOUSE" --require-hashes \
  --no-compile --report "$F5_INSTALL_REPORT" -r "$F5_REQUIREMENTS_LOCK"
F5_PYTHON="$F5_VENV/bin/python"
if [ "${SNAKE_COLAB_CANARY:-}" = "1" ]; then
  F5_SITE_PACKAGES=$("$F5_PYTHON" -c "import site; print(site.getsitepackages()[0])")
  F5_SOURCE_ROOT=$(pwd)
  {
    printf '%s\n' "$F5_SOURCE_ROOT"
    printf '%s\n' "import importlib; importlib.import_module('research.colab_cli.canary_observer').install()"
  } > "$F5_SITE_PACKAGES/snake_colab_canary.pth"
  "$F5_PYTHON" -c "import importlib.util; from pathlib import Path; s=importlib.util.find_spec('research.colab_cli.canary_observer'); assert s and s.origin and Path(s.origin).resolve().is_relative_to(Path.cwd().resolve())"
fi
if [ "$F5_DEVICE" != "cuda:0" ]; then
  echo "F5 requires cuda:0" >&2
  exit 2
fi
if [ "$F5_MODE" = "capture" ] || [ "$F5_MODE" = "capture-qualification" ]; then
  "$F5_PYTHON" -m research.bl_residual.f5_job capture-environment --output "$COLAB_CLI_ROOT/output/environment.json"
  if [ "$F5_MODE" = "capture-qualification" ]; then
    "$F5_PYTHON" -m research.bl_residual.f5_qualification --inputs "$COLAB_INPUT_DIR" \
      --fixtures "$COLAB_INPUT_DIR/timing-fixtures.json" --output "$COLAB_CLI_ROOT/output" \
      --device "$F5_DEVICE" --transport-only --bootstrap
  fi
elif [ "$F5_MODE" = "prepare" ] || [ "$F5_MODE" = "prepare-qualification" ]; then
  "$F5_PYTHON" -m research.bl_residual.f5_job prepare --inputs "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/prepare"
  if [ "$F5_MODE" = "prepare-qualification" ]; then
    "$F5_PYTHON" -m research.bl_residual.f5_qualification --inputs "$COLAB_INPUT_DIR" \
      --fixtures "$COLAB_INPUT_DIR/timing-fixtures.json" --output "$COLAB_CLI_ROOT/output" \
      --device "$F5_DEVICE" --transport-only
  fi
elif [ "$F5_MODE" = "qualification-transport" ]; then
  "$F5_PYTHON" -m research.bl_residual.f5_qualification --inputs "$COLAB_INPUT_DIR" \
    --fixtures "$COLAB_INPUT_DIR/timing-fixtures.json" --output "$COLAB_CLI_ROOT/output" \
    --device "$F5_DEVICE" --transport-only
elif [ "$F5_MODE" = "qualification" ]; then
  "$F5_PYTHON" -m research.bl_residual.f5_qualification --inputs "$COLAB_INPUT_DIR" \
    --fixtures "$COLAB_INPUT_DIR/timing-fixtures.json" --output "$COLAB_CLI_ROOT/output" --device "$F5_DEVICE"
elif [ "$F5_MODE" = "fit" ]; then
  F5_PLAN=${F5_PLAN:-$COLAB_INPUT_DIR/f5-batch-plan.pt}
  F5_ACCEPTANCE=${F5_ACCEPTANCE:-$COLAB_INPUT_DIR/acceptance.json}
  F5_LEDGER=${F5_LEDGER:-$COLAB_INPUT_DIR/verified-ledger.json}
  "$F5_PYTHON" -m research.bl_residual.f5_job fit-pair --inputs "$COLAB_INPUT_DIR" --plan "$F5_PLAN" \
    --acceptance "$F5_ACCEPTANCE" --ledger "$F5_LEDGER" --output "$COLAB_CLI_ROOT/output/paired"
else
  echo "F5_MODE must be capture, prepare, qualification or fit" >&2
  exit 2
fi
