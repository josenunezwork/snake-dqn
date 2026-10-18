#!/usr/bin/env bash
# Prepare a fresh Colab runtime for snake-dqn jobs WITHOUT running the smoke matrix: pinned checkout at
# /content/t16-snake-dqn, uv Python 3.12 venv at /content/t16-colab-venv (parity pins), torch wheel chosen
# from the GPU's compute capability (cu128 for Blackwell, cu126 otherwise), verified T16 input bundle at
# /content/t16-colab-extracted. CPU runtimes use the CPU wheel (or set T16_CPU_ONLY=1).
# Needs Drive mounted at /content/drive. Usage: setup_runtime.sh <commit>
set -euo pipefail
COMMIT=${1:?usage: setup_runtime.sh <commit>}
REPO=https://github.com/josenunezwork/snake-dqn.git
ARCHIVE=/content/drive/MyDrive/t16_colab/t16_colab_inputs.tar.gz
SHA=229be2949c99f198b27decdf403b83ec4085d8df2dcfe629b7eff9208593be8e
CK=/content/t16-snake-dqn; VENV=/content/t16-colab-venv; EX=/content/t16-colab-extracted
[ -f "$ARCHIVE" ] || { echo "missing $ARCHIVE (mount Drive first)"; exit 2; }
[ "$(sha256sum "$ARCHIVE" | cut -d' ' -f1)" = "$SHA" ] || { echo "bundle sha mismatch"; exit 2; }
rm -rf "$CK" "$VENV" "$EX"
git init -q "$CK" && git -C "$CK" remote add origin "$REPO"
git -C "$CK" fetch -q --depth 1 origin "$COMMIT" && git -C "$CK" checkout -q --detach FETCH_HEAD
[ "$(git -C "$CK" rev-parse HEAD)" = "$COMMIT" ] || { echo "checkout mismatch"; exit 2; }
python3 -m pip install -q uv
python3 -m uv python install 3.12
python3 -m uv venv --python 3.12 --managed-python --seed "$VENV"
PY=$VENV/bin/python
"$PY" -m pip install -q -r "$CK/requirements-gpu.txt" numpy==1.26.4 numba==0.68.0 llvmlite==0.50.0 'pytest>=9,<10'
if [ "${T16_CPU_ONLY:-0}" = 1 ] || ! command -v nvidia-smi >/dev/null 2>&1 \
    || ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader >/dev/null 2>&1; then
  IDX=https://download.pytorch.org/whl/cpu
  echo "CPU runtime -> $IDX"
else
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  IDX=https://download.pytorch.org/whl/cu126
  python3 -c "import sys; sys.exit(0 if float('$CAP') >= 10.0 else 1)" && IDX=https://download.pytorch.org/whl/cu128
  echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader) cap $CAP -> $IDX"
fi
"$PY" -m pip install -q --index-url "$IDX" torch==2.9.1
if [ "$IDX" = https://download.pytorch.org/whl/cpu ]; then
  "$PY" -c "import torch; print('torch', torch.__version__, 'CPU')"
else
  "$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
fi
(cd "$CK" && PYTHONPATH="$CK" "$PY" -c "from pathlib import Path; from research.t16_colab.bundle import verify_and_extract; print(verify_and_extract(Path('$ARCHIVE'), Path('$EX')))")
echo "SETUP OK commit $COMMIT"
