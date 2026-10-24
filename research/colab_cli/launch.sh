#!/usr/bin/env bash
# Prospective qualified revision: do not install over a live qualified broker.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${SNAKE_DQN_PYTHON:-/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python}
ORIGINAL_ARGS=("$@")
MODULE=research.colab_cli.supervisor
CONTROL_NICE0=0
if [ "${1:-}" = --control-nice0 ]; then
  CONTROL_NICE0=1
  shift
fi
if [ "${1:-}" = acceptance ]; then
  shift
  MODULE=research.colab_cli.acceptance
fi
if [ "${1:-}" = --check ]; then
  if [ "$CONTROL_NICE0" = 1 ]; then
    echo "refusing: control mode requires its actual admitted launch chain" >&2
    exit 2
  fi
  shift
  printf '%s\n' "AC + open-lid guard; caffeinate -is $PY -m $MODULE $*"
  exit 0
fi
CURRENT_NICE=$(/bin/ps -o nice= -p "$$")
if ! [[ "$CURRENT_NICE" =~ ^[[:space:]]*-?[0-9]+[[:space:]]*$ ]]; then
  echo "refusing: cannot read Colab launcher scheduling nice value" >&2
  exit 2
fi
if [ "$CONTROL_NICE0" = 1 ]; then
  if [ "$CURRENT_NICE" -ne 0 ] || [ "$MODULE" != research.colab_cli.supervisor ]; then
    echo "refusing: admitted Colab control requires supervisor at absolute nice 0" >&2
    exit 2
  fi
  : "${SNAKE_COLAB_CONTROL_ROOT_RECORD:?refusing: missing admitted control root record}"
  : "${SNAKE_COLAB_CONTROL_HELPER:?refusing: missing pinned control verifier}"
  unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONSAFEPATH
  export PYTHONNOUSERSITE=1
  "$PY" "$SNAKE_COLAB_CONTROL_HELPER" --verify-launch "$SNAKE_COLAB_CONTROL_ROOT_RECORD" --pid "$$" -- "$@"
elif [ "$CURRENT_NICE" -lt 15 ]; then
  exec /usr/bin/nice -n "$((15 - CURRENT_NICE))" /bin/bash "$0" "${ORIGINAL_ARGS[@]}"
fi
pmset -g batt | grep 'AC Power' >/dev/null || { echo 'refusing: Mac is on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep '"AppleClamshellState" = No' >/dev/null || {
  echo 'refusing: lid is closed or its state is unknown' >&2; exit 2;
}
cd "$REPO"
umask 077
exec /usr/bin/caffeinate -is "$PY" -m "$MODULE" "$@"
