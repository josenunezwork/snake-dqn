#!/usr/bin/env bash
# Fresh-VM bootstrap.  The supervisor must already have written state/lease
# before invoking this script, so package installation cannot outlive its lease.
set -euo pipefail

ROOT=${1:?usage: bootstrap.sh ROOT REPO_RELATIVE_SCRIPT}
SCRIPT=${2:?usage: bootstrap.sh ROOT REPO_RELATIVE_SCRIPT}
REPO="$ROOT/repo"
REQUIREMENTS=${COLAB_CLI_REQUIREMENTS:-requirements.txt}
VENV="$ROOT/.venv"
case "$ROOT" in /content/snake-cli/*) ;; *) echo "unsafe root: $ROOT" >&2; exit 2;; esac
if [ ! -d "$REPO/.git" ] && [ ! -f "$REPO/pyproject.toml" ]; then
  echo "pinned repository bundle missing: $REPO" >&2
  exit 2
fi
case "$SCRIPT" in /*|*'..'*) echo "script must be repo-relative: $SCRIPT" >&2; exit 2;; esac
[ -f "$REPO/$SCRIPT" ] || { echo "job script missing from pinned bundle: $SCRIPT" >&2; exit 2; }

# Optional immutable, manifest-bound wheel inputs. An absent cache preserves the
# ordinary bootstrap exactly; no network source or dependency pin is changed.
WHEELHOUSE=""
if [ -n "${COLAB_CLI_WHEELHOUSE:-}" ]; then
  [ "$COLAB_CLI_WHEELHOUSE" = inputs/wheelhouse ] || { echo 'unsafe wheelhouse' >&2; exit 2; }
  [ -d "$ROOT/$COLAB_CLI_WHEELHOUSE" ] || { echo 'missing bound wheelhouse' >&2; exit 2; }
  WHEELHOUSE="$ROOT/$COLAB_CLI_WHEELHOUSE"
fi
pip_with_cache() {
  if [ -n "$WHEELHOUSE" ]; then
    "$@" --find-links "$WHEELHOUSE"
  else
    "$@"
  fi
}
export PIP_CACHE_DIR="$ROOT/.pip-cache"
export UV_CACHE_DIR="$ROOT/.uv-cache"
pip_with_cache python3 -m pip install -q uv
python3 -m uv python install 3.12
python3 -m uv venv --python 3.12 --managed-python --seed "$VENV"
PY="$VENV/bin/python"
INDEX=https://download.pytorch.org/whl/cpu
CUDA_EXPECTED=cpu
if command -v nvidia-smi >/dev/null 2>&1; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  INDEX=https://download.pytorch.org/whl/cu126
  CUDA_EXPECTED=12.6
  "$PY" -c "import sys; sys.exit(0 if float('$CAP') >= 10 else 1)" && INDEX=https://download.pytorch.org/whl/cu128
  "$PY" -c "import sys; sys.exit(0 if float('$CAP') >= 10 else 1)" && CUDA_EXPECTED=12.8
fi
pip_with_cache "$PY" -m pip install -q --index-url "$INDEX" torch==2.9.1 numpy==1.26.4
CONSTRAINTS="$ROOT/.bootstrap-constraints.txt"
printf 'torch==2.9.1\nnumpy==1.26.4\n' > "$CONSTRAINTS"
if [ -n "$REQUIREMENTS" ]; then
  case "$REQUIREMENTS" in /*|*'..'*) echo "requirements must be repo-relative: $REQUIREMENTS" >&2; exit 2;; esac
  [ -f "$REPO/$REQUIREMENTS" ] || { echo "requirements file missing: $REQUIREMENTS" >&2; exit 2; }
  pip_with_cache "$PY" -m pip install -q -c "$CONSTRAINTS" -r "$REPO/$REQUIREMENTS"
fi
if [ "$CUDA_EXPECTED" = cpu ]; then
  "$PY" -c "import torch; assert torch.version.cuda is None; print(torch.__version__, 'cpu')"
else
  "$PY" -c "import torch; assert torch.cuda.is_available(); assert torch.version.cuda.startswith('$CUDA_EXPECTED'); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
fi
if [ "${SNAKE_COLAB_CANARY:-}" = "1" ]; then
  SITE_PACKAGES=$("$PY" -c "import site; print(site.getsitepackages()[0])")
  CANARY_PTH="$SITE_PACKAGES/snake_colab_canary.pth"
  # The path belongs to the fetched pinned bundle.  The executable .pth line
  # installs before the job interpreter imports torch, with no PYTHONPATH
  # override and no dependence on the current working directory.
  {
    printf '%s\n' "$REPO"
    printf '%s\n' "import importlib; importlib.import_module('research.colab_cli.canary_observer').install()"
  } > "$CANARY_PTH"
  "$PY" -c "import importlib.util; from pathlib import Path; spec=importlib.util.find_spec('research.colab_cli.canary_observer'); assert spec and spec.origin and Path(spec.origin).resolve().is_relative_to(Path('$REPO').resolve()), 'pinned CUDA canary observer missing'"
fi
cd "$REPO"
export COLAB_CLI_ROOT="$ROOT"
export COLAB_INPUT_DIR="${COLAB_INPUT_DIR:-$ROOT/inputs}"
export PATH="$VENV/bin:$PATH"
exec bash "$REPO/$SCRIPT"
