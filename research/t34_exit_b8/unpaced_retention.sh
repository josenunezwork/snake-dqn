#!/usr/bin/env bash
# This campaign prefix activates the qualified broker's admission recheck.
# Keep all study behavior in the hash-bound Path B entrypoint.
set -euo pipefail

UNPACED_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$UNPACED_REPO_ROOT"
exec bash "$UNPACED_REPO_ROOT/research/t34_pathb/unpaced_colab.sh"
