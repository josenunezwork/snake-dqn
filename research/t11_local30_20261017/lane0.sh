#!/usr/bin/env bash
# Prepared after the MPS smoke. --check verifies and prints; neither mode starts an 8M run here.
set -euo pipefail
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check ]]; }; then
    echo "Usage: $0 [--check]" >&2
    exit 2
fi
cd /Users/josenunez/Projects/ml/snake-dqn-t11
exec ./venv/bin/python -m research.t11_local30_20261017.launch --lane 0 "$@"
