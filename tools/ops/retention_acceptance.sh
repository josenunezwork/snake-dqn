#!/usr/bin/env bash
# Execute the separately reviewed retention acceptance with Mac safety guards.
# This is not the qualified supervisor route and must never update its registry.
set -euo pipefail
BROKER=${1:?broker path required}
shift
CURRENT_NICE=$(/bin/ps -o nice= -p "$$")
if ! [[ "$CURRENT_NICE" =~ ^[[:space:]]*-?[0-9]+[[:space:]]*$ ]]; then
  echo "refusing: cannot read retention acceptance nice value" >&2
  exit 2
fi
if [ "$CURRENT_NICE" -lt 15 ]; then
  exec /usr/bin/nice -n "$((15 - CURRENT_NICE))" /bin/bash "$0" "$BROKER" "$@"
fi
pmset -g batt | grep 'AC Power' >/dev/null || { echo 'refusing: Mac is on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || {
  echo 'refusing: lid is closed or its state is unknown' >&2; exit 2;
}
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
cd "$BROKER"
umask 077
exec /usr/bin/caffeinate -is "$PY" -m research.colab_cli.retention_acceptance execute "$@"
