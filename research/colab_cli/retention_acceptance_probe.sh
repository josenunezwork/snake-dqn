#!/usr/bin/env bash
set -euo pipefail
exec python -m research.colab_cli.retention_acceptance_probe "$@"
