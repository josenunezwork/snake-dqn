#!/usr/bin/env bash
# The workload imports its frozen F1 closure, with the reviewed profile as a sibling.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified runtime required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SNAKE_DQN_DEVICE=cpu F1_COMMIT=2af48cd0b98a2063f46871cf4d320a5541b789af
python -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR"
python - <<'PY'
import hashlib, json, os, shutil, tarfile
from pathlib import Path
root = Path(os.environ['COLAB_CLI_ROOT'])
inputs = Path(os.environ['COLAB_INPUT_DIR']) / 'inputs'
archive = inputs / 'f1-source.tar'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == os.environ['RQ_SOURCE_CLOSURE_SHA256']
manifest = inputs / 'frozen-source.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == os.environ['RQ_SOURCE_MANIFEST_SHA256']
assert json.loads(manifest.read_text())['archive_sha256'] == os.environ['RQ_SOURCE_CLOSURE_SHA256']
frozen = root / 'frozen'
frozen.mkdir(exist_ok=False)
with tarfile.open(archive) as source:
    for member in source.getmembers():
        target = (frozen / member.name).resolve()
        assert target.is_relative_to(frozen.resolve()) and (member.isfile() or member.isdir())
    source.extractall(frozen, filter='data')
shutil.copyfile(manifest, frozen / 'frozen-source.json')
PY
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/public-checkpoints/checkpoints"
export PYTHONPATH="$COLAB_CLI_ROOT/frozen:$COLAB_CLI_ROOT/repo"
cd "$COLAB_CLI_ROOT/frozen"
mode=run
if [ "${RQ_MODE:?mode required}" = injection ]; then mode=prepare-injection; fi
exec python -m research.rpc_qualification.loop "$mode" \
  --bank "$COLAB_INPUT_DIR/inputs/f1-root-bank.json.xz" \
  --ledger "$COLAB_INPUT_DIR/inputs/verified-ledger.json" \
  --b8 "$COLAB_INPUT_DIR/public-checkpoints/B8_s3_final.pth" \
  --frp3 "$COLAB_INPUT_DIR/public-checkpoints/checkpoints/frp3_m3_s12_u60000_20261005.pth" \
  --output "$COLAB_CLI_ROOT/output" --platform "$RQ_PLATFORM" --workers "$RQ_WORKERS" \
  --namespace "$RQ_NAMESPACE" --allocation "$RQ_ALLOCATION" --loop-seconds "$RQ_LOOP_SECONDS" \
  --source-closure-manifest "$COLAB_CLI_ROOT/frozen/frozen-source.json" \
  --source-closure-sha256 "$RQ_SOURCE_CLOSURE_SHA256" \
  --source-manifest-sha256 "$RQ_SOURCE_MANIFEST_SHA256"
