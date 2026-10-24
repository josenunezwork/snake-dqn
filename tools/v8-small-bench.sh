#!/usr/bin/env bash
# Local operator sibling. No runtime/broker edits and no automatic dispatch.
set -euo pipefail
cd /Users/josenunez/Projects/ml/snake-dqn-v8-metadata-qualified-02.noindex
export PYTHONPATH=/Users/josenunez/Projects/ml/snake-dqn-v8-metadata-qualified-02.noindex:/Users/josenunez/Projects/ml/snake-dqn-gpu-v8
export PYTHONDONTWRITEBYTECODE=1
exec /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m tools.v8_small_bench "$@"
