#!/usr/bin/env bash
# Operator-only, after the exclusive P1 timing arm has been removed by its owner.
set -euo pipefail
export SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS= PYTEST_PLUGINS=
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
exec "$PY" - <<'PY'
import os
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.setpriority(os.PRIO_PROCESS, 0, max(15, os.getpriority(os.PRIO_PROCESS, 0)))

from research.compute.factory_interim_launcher import run_test

arm = Path('/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed')
if arm.exists():
    raise SystemExit('HOLD: P1/timing window remains armed; its owner must close it')
nodes = ['tests/test_bl_residual_factory_masks.py',
         'tests/test_bl_residual_f4_d256_package.py',
         'research/bl_labels/test_factory_executable.py']
pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
lines = ['schema=factory-companion-test-manifest/v2', 'source_commit=' + pin]
for node in nodes:
    lines += ['fixture=' + node + '|' + hashlib.sha256(Path(node).read_bytes()).hexdigest(),
              'node=' + node]
with tempfile.TemporaryDirectory(prefix='f4-d256-preflight-') as directory:
    manifest = Path(directory) / 'reviewed-tests.txt'
    manifest.write_text('\n'.join(lines) + '\n')
    command = [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=', '-p', 'no:xdist',
               '-p', 'research.compute.factory_companion_pytest', *nodes]
    started = time.monotonic()
    status = run_test(command, review_manifest=str(manifest),
                      review_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    if status:
        raise SystemExit(status)
    from research.bl_ppo.exports import export_json
    output = Path(os.environ.get('F4_PREFLIGHT_OUT',
        '/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d256-preflight-01.json'))
    export_json(output, {
        'schema': 'bl-residual-f4-d256-nofit-preflight/v1', 'status': 'PASS',
        'commit': pin, 'command': command, 'elapsed_seconds': time.monotonic() - started,
        'fixtures': {node: hashlib.sha256(Path(node).read_bytes()).hexdigest() for node in nodes},
        'fits_run': False, 'study_outcomes_opened': False,
    })
    print('PREFLIGHT_PASS ' + str(output))
PY
