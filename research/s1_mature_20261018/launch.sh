#!/usr/bin/env bash
# Caller-owned foreground launcher.  The reviewed supervisor wrapper owns AC/lid
# guards, caffeinate, allocation, transfer, stop, and backend disappearance.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
pmset -g batt | rg 'AC Power' >/dev/null || { echo 'refusing: Mac is on battery' >&2; exit 2; }
ioreg -r -k AppleClamshellState | rg '"AppleClamshellState" = No' >/dev/null || {
  echo 'refusing: lid is closed or state is unknown' >&2; exit 2;
}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec /usr/bin/caffeinate -is "$repo/venv/bin/python" -m research.s1_mature_20261018.run "$@"
