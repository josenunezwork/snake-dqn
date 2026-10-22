#!/usr/bin/env bash
# Shell entrypoint smoke: refusal is offline and needs neither tmux nor Colab.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
bash -n "$ROOT/tools/ops/dispatch.sh"
if "$ROOT/tools/ops/dispatch.sh" > /dev/null 2>&1; then
  echo 'dispatch accepted missing arguments' >&2; exit 1
fi
if "$ROOT/tools/ops/mac_run.sh" > /dev/null 2>&1; then
  echo 'mac_run accepted missing arguments' >&2; exit 1
fi
printf 'ops shell entrypoint checks passed\n'
