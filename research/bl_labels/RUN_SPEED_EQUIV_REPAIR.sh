#!/bin/bash
# Operator-only successor for the failed round-27 speed run. Never auto-chain phases.
# Usage: RUN_SPEED_EQUIV_REPAIR.sh <published-full-pin> <phase>
set -euo pipefail
SOURCE=$(cd "$(dirname "$0")/../.." && pwd)
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
SPEED="$ART/bl-labels-f0-speed"
PIN=${1:?Supply the published full repair pin}
PHASE=${2:?Supply speed, publish-speed, analyze-speed or record-speed}
SPEED_PHASE=research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES_02.json
RECEIPTS=research/bl_labels/ledger/ROUND27_SPEED_RECEIPTS_20261024_02.json
[[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full lowercase pin required' >&2; exit 2; }
export SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$SOURCE"
if [ -e "$ART/ops/mac-timing-armed" ]; then
  echo 'HOLD: P1/timing window remains armed; only its owner may close it' >&2
  exit 2
fi

check_publication_workspace() {
  if ! git diff --cached --quiet || \
     ! git diff --quiet -- docs/research/final_run_20261019/CAMPAIGN_LOG.md; then
    echo 'HOLD: preserve existing staged work or campaign-log edits before publication' >&2
    exit 2
  fi
}

publish_phase() {
  local relative=$1 label=$2
  check_publication_workspace
  "$PY" - "$relative" "$label" <<'PY'
import hashlib
import sys
from pathlib import Path
relative, label = sys.argv[1:]
sha = hashlib.sha256(Path(relative).read_bytes()).hexdigest()
with Path('docs/research/final_run_20261019/CAMPAIGN_LOG.md').open('a') as stream:
    stream.write('\n\nRound-27 speed repair equiv-02: ' + label + '. Ledger `' + relative +
                 '`, SHA256 `' + sha + '`. equiv-01 retained unchanged.\n')
PY
  git add "$relative" "$relative.sha256" "$relative.complete" \
    docs/research/final_run_20261019/CAMPAIGN_LOG.md
  git diff --cached --check
  git commit -m "Publish $label"
  git fetch origin
  git merge origin/main
  git push origin HEAD:main
  git push origin HEAD
  git fetch origin
  git show "origin/main:$relative" | cmp - "$relative"
}

case "$PHASE" in
  speed)
    # Offline-prepared equiv-pack-02 must bind PIN. Collection validates it.
    "$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
      bl-f0-speed-equiv-r27-02 "$PIN" -- /bin/bash research/bl_labels/RUN_SPEED_EQUIV.sh
    ;;
  publish-speed)
    check_publication_workspace
    "$PY" -m research.bl_labels.speed_equiv publication --output "$SPEED/equiv-02" \
      --out "$SOURCE/$SPEED_PHASE"
    publish_phase "$SPEED_PHASE" 'round-27 speed equivalence phase ledger 02'
    ;;
  analyze-speed)
    "$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
      bl-f0-speed-analyze-r27-02 "$PIN" -- "$PY" -m research.bl_labels.speed_equiv analyze \
      --output "$SPEED/equiv-02" --publication-repo "$SOURCE" --publication-path "$SPEED_PHASE"
    ;;
  record-speed)
    check_publication_workspace
    "$PY" - "$SPEED" "$ART/bl-labels-f0/w12-admission-01/receipt.json" "$RECEIPTS" "$PIN" <<'PY'
import json
import sys
from pathlib import Path
from research.bl_labels.exports import _verify, publish_json
speed, calibration, target = map(Path, sys.argv[1:4])
pin = sys.argv[4]
paths = {'equivalence': speed / 'equiv-02/receipt.json',
         'timing_table': speed / 'equiv-02/timing-table.json',
         'calibration_admission': calibration}
value = {'schema': 'bl-labels-f0-round27-operational-receipts/v1',
         'source_pin': pin, 'receipts': {}}
for name, path in paths.items():
    sha = _verify(path)
    payload = json.loads(path.read_text())
    if name == 'equivalence' and payload.get('new', {}).get('commit') != pin:
        raise SystemExit('equivalence receipt does not bind the supplied repair pin')
    value['receipts'][name] = {'path': str(path), 'sha256': sha,
        'status': payload.get('status') if isinstance(payload, dict) else None}
publish_json(target, value)
PY
    publish_phase "$RECEIPTS" 'round-27 speed equivalence and W12 receipts 02'
    ;;
  *)
    echo 'phase must be speed, publish-speed, analyze-speed or record-speed' >&2
    exit 2
    ;;
esac
