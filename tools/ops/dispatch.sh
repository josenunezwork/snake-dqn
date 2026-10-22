#!/usr/bin/env bash
# Keep the public shell entrypoint; the orchestrator owns the flock descriptor.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec python3 "$HERE/dispatch.py" "$@"
