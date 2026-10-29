#!/usr/bin/env bash
# Synthetic hardware acceptance only. The qualified bootstrap selects cu126.
set -euo pipefail
ROOT=${COLAB_CLI_ROOT:?runtime root required}
exec "$ROOT/.venv/bin/python" "$ROOT/repo/research/gpu_preference/acceptance.py"
