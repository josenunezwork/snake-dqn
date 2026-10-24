#!/usr/bin/env bash
# Start, inspect, or stop the single persistent factory companion monitor.
set -euo pipefail

usage() {
  echo "usage: $0 [--repo REPO] start|status|stop" >&2
  exit 2
}

HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/../.." && pwd -P)"
POLICY=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_FACTORY_CONCURRENCY_20261023.md
POLICY_SHA=af117cd98d51f428cd83d6e33138fcd13def0b4d7c7cf63a8b2e2b83db99105f

if [ "${1:-}" = --repo ]; then
  [ $# -eq 3 ] || usage
  REPO=$2
  shift 2
fi
[ $# -eq 1 ] || usage
ACTION=$1
case "$ACTION" in start|status|stop) ;; *) usage ;; esac

[ -f "$POLICY" ] || { echo "REFUSE: concurrency ruling is unavailable" >&2; exit 2; }
[ "$(shasum -a 256 "$POLICY" | awk '{print $1}')" = "$POLICY_SHA" ] || {
  echo "REFUSE: concurrency ruling SHA does not match the bound policy" >&2; exit 2
}
REPO="$(cd "$REPO" && pwd -P)"
git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "REFUSE: --repo is not a Git worktree: $REPO" >&2; exit 2
}
PYTHON=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
[ -x "$PYTHON" ] || { echo "REFUSE: supervisor Python is not executable: $PYTHON" >&2; exit 2; }
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON" \
  "$REPO/tools/ops/factory_companion_monitor.py" --repo "$REPO" "$ACTION"
