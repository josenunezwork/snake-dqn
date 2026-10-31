#!/usr/bin/env bash
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified runtime required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${COLAB_OUTPUT_DIR:?output root required}"
cd "$COLAB_CLI_ROOT/repo"
export SNAKE_DQN_DEVICE=cpu SNAKE_WATCH_SOURCE=live
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/checkpoints"
export WATCH6_RELEASE_EVIDENCE="$COLAB_INPUT_DIR/release-evidence"
PY="$COLAB_CLI_ROOT/.venv/bin/python"
"$PY" -c 'from pathlib import Path; import os; from research.watch6_search_efficacy_v3.uploads import stage_metadata; stage_metadata(Path(os.environ["COLAB_INPUT_DIR"]), os.environ["WATCH6_METADATA_SHA256"])'
"$PY" -c 'from pathlib import Path; import os; from research.watch6_search_efficacy_v3.uploads import stage_checkpoints; stage_checkpoints(Path(os.environ["COLAB_INPUT_DIR"]))'
"$PY" -c 'from pathlib import Path; import os; from research.watch6_search_efficacy.product import stage_release_evidence; stage_release_evidence(Path(os.environ["COLAB_INPUT_DIR"]))'
"$PY" -c 'from research.watch6_search_efficacy_v3.admission import runtime_admit; runtime_admit()'
args=(--ledger "$COLAB_INPUT_DIR/verified-ledger.json" --out "$COLAB_OUTPUT_DIR/efficacy"
      --shard "$WATCH6_SHARD" --workers "$WATCH6_WORKERS" --shards "$WATCH6_SHARDS")
if [[ "$WATCH6_FIXTURE" == 1 ]]; then
  args+=(--fixture --fixture-horizon 12 --canary-performance-out "$COLAB_OUTPUT_DIR/performance.json")
fi
"$PY" -m research.watch6_search_efficacy_v3.run "${args[@]}"
if [[ "$WATCH6_FIXTURE" != 1 ]]; then
  "$PY" -c 'import json, os; from pathlib import Path; from tools.ops.package_spec import publish_json; publish_json(Path(os.environ["COLAB_OUTPUT_DIR"]) / "performance.json", json.loads(os.environ["WATCH6_MEASUREMENTS"]))'
fi
