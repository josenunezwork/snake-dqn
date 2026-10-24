#!/usr/bin/env bash
# Device-repair successor; retain RUN_D256_NEXT.sh unchanged.
# Same-pin no-fit preflight is already complete; resume at prepare.
# One operator-requested phase per invocation. Never call this during P1's window.
set -euo pipefail
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f4fix
Q=/Users/josenunez/Projects/ml/snake-dqn-f4-d256-qualified-03.noindex
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
F4="$ART/big-lever/exp/f4"
PACK="$F4/d256-pack-02"
F4_INPUTS="$F4/d1024-staging-01/inputs"
SPEED="$ART/bl-labels-f0-speed"
FIT_PIN=c2c82ff240a28247dd9b18608831eecea6d94205
OPS_PIN=3c5fc403e5167a47ebb8b2fd090fd1e024da8e76
SPEED_PIN=bf8ac97dc66d1b4515e4c5510db3ad5886fa695d
FIT_PHASE=research/bl_residual/ledger/F4_D256_FIT_PHASE_20261023.json
SPEED_PHASE=research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES.json
export SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

if [ -e "$ART/ops/mac-timing-armed" ]; then
  echo 'HOLD: P1/timing window remains armed; only its owner may close it' >&2
  exit 2
fi
if [ "$(git -C "$Q" rev-parse HEAD)" != "$FIT_PIN" ] || \
   [ -n "$(git -C "$Q" status --porcelain)" ]; then
  echo 'HOLD: qualified F4 source must be clean and pinned' >&2
  exit 2
fi
cd "$Q"

publish_phase() {
  local relative=$1 label=$2
  if ! git -C "$SOURCE" diff --quiet -- docs/research/final_run_20261019/CAMPAIGN_LOG.md || \
     ! git -C "$SOURCE" diff --cached --quiet; then
    echo 'HOLD: preserve existing staged work or campaign-log edits before publication' >&2
    exit 2
  fi
  "$PY" - "$SOURCE" "$relative" "$label" <<'PY'
import hashlib
import sys
from pathlib import Path
repo, relative, label = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sha = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
with (repo / 'docs/research/final_run_20261019/CAMPAIGN_LOG.md').open('a') as stream:
    stream.write('\n\nOutcome-blind phase publication: ' + label + '. Ledger `' + relative +
                 '`, SHA256 `' + sha + '`. Published before the dependent consumer; '
                 'no reserve/gameplay outcomes opened by publication.\n')
print(label + ' ' + sha)
PY
  git -C "$SOURCE" add "$relative" "$relative.sha256" "$relative.complete" \
    docs/research/final_run_20261019/CAMPAIGN_LOG.md
  git -C "$SOURCE" diff --cached --check
  git -C "$SOURCE" commit -m "Publish $label phase ledger"
  git -C "$SOURCE" fetch origin
  git -C "$SOURCE" merge origin/main
  git -C "$SOURCE" push origin HEAD:main
  git -C "$SOURCE" push origin HEAD
  git -C "$SOURCE" fetch origin
  "$PY" - "$SOURCE" "$relative" <<'PY'
import subprocess
import sys
from pathlib import Path
repo, relative = Path(sys.argv[1]), sys.argv[2]
published = subprocess.check_output(['git', '-C', str(repo), 'show', 'origin/main:' + relative])
if published != (repo / relative).read_bytes():
    raise SystemExit('HOLD: local phase ledger differs from origin/main')
print('LOCAL == origin/main ' + relative)
PY
}

case "${1:-}" in
  preflight)
    "$Q/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
      bl-f4-d256-preflight-02 "$FIT_PIN" -- \
      /usr/bin/env F4_PREFLIGHT_OUT="$F4/d256-preflight-02.json" \
      /bin/bash research/bl_residual/RUN_D256_FIT_PREFLIGHT.sh
    ;;
  prepare)
    "$Q/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
      bl-f4-d256-prepare-02 "$FIT_PIN" -- "$PY" -m research.bl_residual.f4_d256_package \
      --out "$PACK" --commit "$FIT_PIN" \
      --factory-ledger "$ART/bl-labels-f0/d256-union-final-01/verified-ledger.json" \
      --preflight-receipt "$F4/d256-preflight-02.json" \
      --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
      --judgment "$F4/judgment.json"
    ;;
  canary)
    "$Q/tools/ops/dispatch.sh" bl-f4-d256-canary-02 "$PACK/canary.json" \
      --broker "$Q" --cwd "$Q" --state-dir "$F4/d256-canary-state-02"
    ;;
  seal-main)
    "$PY" -m research.bl_residual.f4_d256_package \
      --seal-main-template "$PACK/main-template.json" \
      --canary-receipt "$PACK/canary-output-l4-01/receipt.json" \
      --main-manifest-out "$PACK/main.json"
    ;;
  fit)
    "$Q/tools/ops/dispatch.sh" bl-f4-d256-fit-02 "$PACK/main.json" \
      --broker "$Q" --cwd "$Q" --state-dir "$F4/d256-fit-state-02"
    ;;
  publish-fit)
    "$PY" -m research.bl_residual.f4_d256_package \
      --publish-fit "$PACK" --out "$SOURCE/$FIT_PHASE"
    publish_phase "$FIT_PHASE" 'F4 D256 fit'
    ;;
  stage-d1024)
    "$PY" -m research.bl_residual.f4_d256_package --stage-d1024 "$PACK" \
      --fit-publication "$SOURCE/$FIT_PHASE" --out "$F4_INPUTS"
    ;;
  speed)
    "$Q/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
      bl-f0-speed-equiv-r27 "$SPEED_PIN" -- /bin/bash research/bl_labels/RUN_SPEED_EQUIV.sh
    ;;
  publish-speed)
    "$PY" -m research.bl_labels.speed_equiv publication --output "$SPEED/equiv-01" \
      --out "$SOURCE/$SPEED_PHASE"
    publish_phase "$SPEED_PHASE" 'round-27 speed equivalence'
    ;;
  analyze-speed)
    "$Q/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
      bl-f0-speed-analyze-r27 "$SPEED_PIN" -- "$PY" -m research.bl_labels.speed_equiv analyze \
      --output "$SPEED/equiv-01" --publication-repo "$SOURCE" --publication-path "$SPEED_PHASE"
    ;;
  record-speed)
    "$PY" - "$SOURCE" "$SPEED" "$ART/bl-labels-f0/w12-admission-01/receipt.json" <<'PY'
import json
import sys
from pathlib import Path
from research.bl_labels.exports import _verify, publish_json
repo, speed, calibration = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
paths = {'equivalence': speed / 'equiv-01/receipt.json',
         'timing_table': speed / 'equiv-01/timing-table.json',
         'calibration_admission': calibration}
value = {'schema': 'bl-labels-f0-round27-operational-receipts/v1', 'receipts': {}}
for name, path in paths.items():
    sha = _verify(path)
    payload = json.loads(path.read_text())
    value['receipts'][name] = {'path': str(path), 'sha256': sha,
        'status': payload.get('status') if isinstance(payload, dict) else None}
publish_json(repo / 'research/bl_labels/ledger/ROUND27_SPEED_RECEIPTS_20261023.json', value)
PY
    publish_phase research/bl_labels/ledger/ROUND27_SPEED_RECEIPTS_20261023.json \
      'round-27 equivalence and W12 admission receipts'
    ;;
  d1024-check|d1024|d1024-check-old|d1024-old)
    args=()
    fallback=()
    name=bl-f0-d1024-r27
    if [ "$1" = d1024-check ] || [ "$1" = d1024-check-old ]; then
      args=(--check)
      name=bl-f0-d1024-check-r27
    fi
    if [ "$1" = d1024-old ] || [ "$1" = d1024-check-old ]; then
      fallback=("OLD_FACTORY_ROOT=$SPEED/equiv-pack-01/old-reference")
      name="$name-old"
    fi
    "$Q/tools/ops/dispatch.sh" --mac --repo "$SOURCE" "$name" "$OPS_PIN" -- \
      /usr/bin/env F4_INPUTS="$F4_INPUTS" F4_CONTROL="$F4_INPUTS/control.json" \
      "${fallback[@]}" \
      /bin/bash research/bl_labels/RUN_D1024.sh "${args[@]}"
    ;;
  *)
    echo 'usage: RUN_D256_NEXT_DEVICE_FIX_20261024.sh {preflight|prepare|canary|seal-main|fit|publish-fit|stage-d1024|speed|publish-speed|analyze-speed|record-speed|d1024-check|d1024|d1024-check-old|d1024-old}' >&2
    exit 2
    ;;
esac
