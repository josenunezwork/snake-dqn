#!/usr/bin/env bash
# The Python adapter performs the draft guard before any substantive work.
set -euo pipefail
repo=/content/t16-snake-dqn
cd "$repo"
exec /content/t16-colab-venv/bin/python -m research.q0q1_20261017.run_queue_job
