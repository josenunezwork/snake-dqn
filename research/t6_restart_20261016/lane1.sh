#!/usr/bin/env bash
# Prepared after the 200k smoke. No 8M run was started during preparation.
set -euo pipefail
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check ]]; }; then
    echo "Usage: $0 [--check]" >&2
    exit 2
fi
cd /Users/josenunez/Projects/ml/snake-dqn-t6
exec ./venv/bin/python -m research.t6_restart_20261016.launch --lane 1 "$@"
