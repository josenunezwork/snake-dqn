#!/usr/bin/env bash
# Bank-v2 paired screen (exact-stop-l1 vs exact-stop-l1-os) on a Colab CPU host (G4).
# Same staging as research/watch_replay/job.sh; resume by default (verified ledger skip
# + per-unit action-log checkpoints from checkpoints-resume.tar.gz when uploaded).
set -euo pipefail
: "${COLAB_CLI_ROOT:?qualified runtime required}"
: "${COLAB_INPUT_DIR:?uploaded inputs required}"
: "${COLAB_OUTPUT_DIR:?output root required}"
cd "$COLAB_CLI_ROOT/repo"
export SNAKE_DQN_DEVICE=cpu SNAKE_WATCH_SOURCE=live
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
export SNAKE_CKPT_DIR="$COLAB_INPUT_DIR/checkpoints"
export WATCH6_RELEASE_EVIDENCE="$COLAB_INPUT_DIR/release-evidence"
PY="$COLAB_CLI_ROOT/.venv/bin/python"
"$PY" -c 'from pathlib import Path; import os; from research.watch_replay.uploads import stage_metadata; stage_metadata(Path(os.environ["COLAB_INPUT_DIR"]), os.environ["WATCH_REPLAY_METADATA_SHA256"])'
"$PY" -c 'from pathlib import Path; import os; from research.watch_replay.product import stage_release_evidence; stage_release_evidence(Path(os.environ["COLAB_INPUT_DIR"]))'
LABEL=main
if [[ "${SNAKE_COLAB_CANARY:-0}" == 1 ]]; then LABEL=canary; fi
CKPT_ARGS=()
if [[ -f "$COLAB_INPUT_DIR/checkpoints-resume.tar.gz" ]]; then
  "$PY" -c 'from pathlib import Path; import os; from research.watch_replay.screen_l1_os import stage_checkpoint_resume; stage_checkpoint_resume(Path(os.environ["COLAB_INPUT_DIR"]))'
  CKPT_ARGS+=(--checkpoint-in "$COLAB_INPUT_DIR/checkpoints-resume")
fi
"$PY" -m research.watch_replay.screen_l1_os \
  --contract "$COLAB_INPUT_DIR/$LABEL-contract.json" \
  --ledger "$COLAB_INPUT_DIR/$LABEL-ledger.json" \
  --out "$COLAB_OUTPUT_DIR/screen" \
  --workers "$SCREEN_WORKERS" \
  --checkpoint-seconds "${SCREEN_CHECKPOINT_SECONDS:-1200}" \
  ${SCREEN_SESSION_SECONDS:+--session-seconds "$SCREEN_SESSION_SECONDS"} \
  ${SCREEN_SHARD:+--shard "$SCREEN_SHARD"} \
  ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}
