#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
exec "$PY" -m research.watch6_search_efficacy_v3.dispatch "$@"
