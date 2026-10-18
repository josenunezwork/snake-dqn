#!/usr/bin/env bash
# Bootstrap only on a dedicated, owner-mounted Colab runtime. Notebook supplies QUEUE_REF.
set -euo pipefail
: "${QUEUE_REF:?Set QUEUE_REF to the full immutable queue implementation commit}"
[[ "$QUEUE_REF" =~ ^[0-9a-f]{40}$ ]] || { echo 'QUEUE_REF must be a full commit SHA' >&2; exit 2; }
BASE="https://raw.githubusercontent.com/josenunezwork/snake-dqn/$QUEUE_REF/research/colab_queue"
# Local directories only. Shared Drive directories are created by q.py enqueue on the Mac.
CODE=$(mktemp -d /content/colab-queue-code.XXXXXXXX)
for FILE in core.py worker.py; do
  curl --fail --silent --show-error --location --retry 3 --max-time 120 "$BASE/$FILE" -o "$CODE/$FILE"
done
exec "${COLAB_PYTHON:-python3}" -u "$CODE/worker.py" "$@"
