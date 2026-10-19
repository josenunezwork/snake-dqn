#!/usr/bin/env bash
# Bootstrap invokes a repository script with no positional arguments.
set -euo pipefail
ROOT=${COLAB_CLI_ROOT:?runtime root required}
exec "$ROOT/.venv/bin/python" "$ROOT/repo/research/colab_cli/acceptance_probe.py"
