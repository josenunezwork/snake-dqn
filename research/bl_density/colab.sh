#!/usr/bin/env bash
# Runtime entrypoint only; no Colab provider calls.
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified supervisor required}"
: "${COLAB_INPUT_DIR:?uploaded catalog required}"
: "${COLAB_OUTPUT_DIR:?bounded exports required}"
: "${BL_DENSITY_COMMIT:?source pin required}"
: "${BL_DENSITY_PHASE:?canary or production required}"
: "${BL_DENSITY_ROUND14_AMENDMENT_SHA256:?round14 amendment pin required}"
: "${BL_DENSITY_METRICS_SHA256:?metric pin required}"
: "${BL_DENSITY_AMENDMENT_SHA256:?amendment pin required}"
: "${BL_DENSITY_CAPACITY_AMENDMENT_SHA256:?capacity amendment pin required}"
: "${BL_DENSITY_REPLICATION_AMENDMENT_SHA256:?replication amendment pin required}"
: "${BL_DENSITY_SCIENTIFIC_ADMISSION:?explicit scientific admission required}"
if [[ "$BL_DENSITY_SCIENTIFIC_ADMISSION" != "1" ]]; then
  echo "DENSITY-2 scientific admission is not enabled" >&2
  exit 2
fi
PYTHON_BIN="$COLAB_CLI_ROOT/.venv/bin/python"
test -x "$PYTHON_BIN"
cd "$COLAB_CLI_ROOT/repo"
export PYTHONUNBUFFERED=1 SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
case "$BL_DENSITY_PHASE" in
  canary) ARGS=(--canary --horizon 8 --episodes-per-cell 1 --workers 1 --max-seconds 600) ;;
  skip-ledger-canary|production-completion)
    : "${BL_DENSITY_COMPLETION_AMENDMENT_SHA256:?completion amendment pin required}"
    : "${BL_DENSITY_LEDGER_SHA256:?verified ledger pin required}"
    : "${BL_DENSITY_CANARY_LEDGER_SHA256:?canary ledger pin required}"
    : "${BL_DENSITY_INPUT_HASHES_SHA256:?input hashes pin required}"
    "$PYTHON_BIN" - <<'PYVERIFY'
import hashlib
import json
import os
from pathlib import Path
root = Path(os.environ["COLAB_INPUT_DIR"])
def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
checks = root / "input-hashes.json"
assert sha(checks) == os.environ["BL_DENSITY_INPUT_HASHES_SHA256"], "input-hashes changed"
expected = json.loads(checks.read_text())
assert set(expected) == {"public-inputs.json", "skip-ledger.json", "canary-ledger.json"}
for name, digest in expected.items():
    assert sha(root / name) == digest, "uploaded input changed: " + name
assert expected["skip-ledger.json"] == os.environ["BL_DENSITY_LEDGER_SHA256"]
assert expected["canary-ledger.json"] == os.environ["BL_DENSITY_CANARY_LEDGER_SHA256"]
assert sha(Path("research/bl_density/METRICS_AMENDMENT_05.md")) == os.environ["BL_DENSITY_COMPLETION_AMENDMENT_SHA256"]
PYVERIFY
    if [[ "$BL_DENSITY_PHASE" == "skip-ledger-canary" ]]; then
      ARGS=(--canary --horizon 8 --episodes-per-cell 1 --workers 1 --max-seconds 1200
        --skip-ledger "$COLAB_INPUT_DIR/canary-ledger.json")
    else
      : "${BL_DENSITY_CANARY_PROOF_JSON:?recovered skip-ledger canary required}"
      "$PYTHON_BIN" -m research.bl_density.canary \
        --proof-json "$BL_DENSITY_CANARY_PROOF_JSON" --commit "$BL_DENSITY_COMMIT" \
        --amendment-sha256 "$BL_DENSITY_ROUND14_AMENDMENT_SHA256" \
        --ledger-sha256 "$BL_DENSITY_LEDGER_SHA256" \
        --canary-ledger-sha256 "$BL_DENSITY_CANARY_LEDGER_SHA256" \
        --input-hashes-sha256 "$BL_DENSITY_INPUT_HASHES_SHA256"
      ARGS=(--workers 10 --horizon 5000 --max-seconds 10800
        --skip-ledger "$COLAB_INPUT_DIR/skip-ledger.json" --identity-replay)
    fi ;;
  production)
    : "${BL_DENSITY_CANARY_PROOF_JSON:?recovered canary proof JSON required}"
    "$PYTHON_BIN" -m research.bl_density.canary \
      --proof-json "$BL_DENSITY_CANARY_PROOF_JSON" --commit "$BL_DENSITY_COMMIT" \
      --amendment-sha256 "$BL_DENSITY_ROUND14_AMENDMENT_SHA256"
    ARGS=(--episodes-per-cell 6 --workers 10 --horizon 5000 --max-seconds 10800) ;;
  *) echo "invalid density phase" >&2; exit 2 ;;
esac
"$PYTHON_BIN" -m research.bl_upload.inputs --input-dir "$COLAB_INPUT_DIR"
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/public-checkpoints/checkpoints"
"$PYTHON_BIN" -m research.bl_density.runner \
  --b8 "$COLAB_INPUT_DIR/public-checkpoints/B8_s3_final.pth" \
  --out "$COLAB_OUTPUT_DIR/density" \
  "${ARGS[@]}"
