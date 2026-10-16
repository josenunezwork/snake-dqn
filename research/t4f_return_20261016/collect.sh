#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/../.."
exec ./venv/bin/python -m research.t4f_return_20261016.macro_collect "$@"
