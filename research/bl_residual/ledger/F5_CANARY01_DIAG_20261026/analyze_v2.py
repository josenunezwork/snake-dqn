"""Offline, outcome-blind metadata audit; writes a create-only analysis receipt."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from research.bl_ppo.exports import digest, export_json
from research.bl_residual import f5_environment as environment

REPO = Path('/Users/josenunez/Projects/ml/snake-dqn-f5envdiag')
BASE = REPO.parent / 'snake-dqn-artifacts/big-lever/exp'
ASTRA = BASE / 'astra-20261024'
DEST = ASTRA / 'F5_CANARY01_DIAG_20261026'
CANARY = BASE / 'f5-timing-recovery-offline-20261026/canary-01'
CAP04 = ASTRA / 'F5_W_CAPTURE_AFTER_22e86986_REPAIR_04/output-04/exports/environment.json'
CAP05 = ASTRA / 'F5_W_CAPTURE_AFTER_TRANSPORT_REQUAL_05/output-05/exports/environment.json'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def binding(path: Path) -> dict:
    return {'path': str(path), 'sha256': digest(path), 'size': path.stat().st_size}


def main() -> None:
    first, second = read(CAP04), read(CAP05)
    comparison = environment.compare_environment(first, second)
    assert comparison['expected_canonical_sha256'] == comparison['actual_canonical_sha256']
    manifest = read(CANARY / 'manifest.json')
    control = read(CANARY / 'inputs/control.json')
    assert control['environment_capture'] == second
    output = CANARY / 'output-01'
    receipt, terminal, hardware = [read(output / name) for name in
                                   ('receipt.json', 'terminal.json', 'hardware.json')]
    assert receipt['stop_verified'] and receipt['exports'] == terminal['exports'] == []
    logs = '\n'.join(read(path).get('job_log_tail', '') for path in sorted(output.glob('poll-*.json')))
    assert 'Creating virtual environment with seed packages at: /content/snake-cli/' in logs
    assert 'Using CPython 3.12.3' in logs and ' + pip==26.2.2' in logs
    sources = {}
    for name in ('f5_environment.py', 'f5_colab.sh', 'f5_job.py'):
        path = 'research/bl_residual/' + name
        blobs = {pin: subprocess.check_output(['git', 'rev-parse', f'{pin}:{path}'], cwd=REPO,
                                             text=True).strip()
                 for pin in ('e25f889b', '57a89e7d')}
        if name != 'f5_colab.sh':
            assert len(set(blobs.values())) == 1
        else:
            before, after = [subprocess.check_output(['git', 'show', f'{pin}:{path}'], cwd=REPO)
                             for pin in ('e25f889b', '57a89e7d')]
            assert before.split(b'if [ "$F5_DEVICE"')[0] == after.split(b'if [ "$F5_DEVICE"')[0]
        sources[path] = blobs
    rows = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            rows.append(binding(path))
    assert len(rows) == 64
    result = {
        'schema': 'bl-f5-canary01-offline-environment-analysis/v1',
        'status': 'HOLD_EXACT_CANARY_KEYS_UNRECOVERABLE_FROM_SAVED_EVIDENCE',
        'outcome_blind': True, 'provider_calls': 0, 'dispatches': 0,
        'saved_output_files': rows,
        'bindings': [binding(path) for path in (CAP04, CAP05, CANARY / 'manifest.json',
                                              CANARY / 'inputs/control.json', Path(__file__))],
        'capture04_capture05': {
            'canonical_equal': True,
            'canonical_sha256': comparison['expected_canonical_sha256'],
            'capture04_raw_sha256': comparison['expected_raw_sha256'],
            'capture05_raw_sha256': comparison['actual_raw_sha256'],
            'canonical_differing_keys': [],
            'canary_control_equals_capture05': True,
        },
        'runtime': {
            'source_commit': receipt['commit'], 'status': receipt['status'],
            'elapsed_seconds': receipt['elapsed_seconds'], 'stop_verified': receipt['stop_verified'],
            'cleanup_error': receipt['cleanup_error'], 'terminal': terminal,
            'hardware': hardware, 'usage_after': receipt['usage_after'],
            'qualification_compare_started_seconds': 68.154,
            'qualification_failure_seconds': 77.702,
            'fixtures_completed': 0,
            'canary_f5_python_from_log': '3.12.3', 'canary_seed_pip_from_log': '26.2.2',
        },
        'capture05_reference_fields': {key: second[key] for key in
            ('cpu', 'cpu_count', 'cpu_isa', 'platform', 'os_kernel', 'gpu_memory',
             'gpu_driver_query', 'numpy_blas_lapack_build', 'installed_versions_sha256',
             'environment_lock_sha256', 'environment_descriptor_sha256')},
        'packages': {
            'installed_count': len(second['installed_packages']),
            'locked_count': len(second['locked_wheels']),
            'outside_lock': {name: second['installed_packages'][name]['version'] for name in
                             sorted(set(second['installed_packages']) - set(second['locked_wheels']))},
            'note': 'All installed distributions are captured; source archives, uploaded inputs and '
                    'the canary source .pth are not distribution RECORD members. Seeded pip version '
                    'matches the reference in saved logs; its installed bytes are unavailable.'},
        'capture05_canary_source_blobs_environment_and_job_equal_install_prefix_equal': sources,
        'cli_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD:research/colab_cli'],
                                            cwd=REPO, text=True).strip(),
        'hypotheses': [
            {'rank_group': 1, 'fields': ['cpu_isa.model name', 'cpu_isa.flags',
                 'numpy_blas_lapack_build', 'gpu_driver_query.driver_version',
                 'gpu_driver_query_raw', 'gpu_driver_query.memory_total', 'gpu_memory'],
             'interpretation': 'Host CPU/ISA/SIMD or driver/memory variation; no saved actual values '
                               'permit ranking driver versus CPU or claiming an exact differing key.'},
            {'rank_group': 2, 'fields': ['platform', 'os_kernel.release', 'os_kernel.version',
                 'cpu_count', 'installed_packages', 'installed_versions_sha256'],
             'interpretation': 'Other host/image or installed closure drift remains possible.'},
            {'rank_group': 3, 'fields': ['source_commit', 'checkout_path', 'input_package'],
             'interpretation': 'No direct canonical dependency and no source/path bug found. '
                               'Source and input identities are separate control admission checks.'},
        ],
        'audit_harness_recovery': 'Initial analyze.py asserted all three full source files equal and stopped before writing output. f5_colab.sh differs only in later qualification mode routing; v2 verifies the shared install prefix instead. Corrected file count to64 including7 staging files; no input evidence changed.',
        'limitations': ['No actual canary environment receipt exists in saved exports or polls.',
                        'No host CPU or driver identity exists in hardware.json.',
                        'No live CUDA/provider validation was performed.'],
    }
    export_json(DEST / 'OFFLINE_ANALYSIS.json', result)
    print(json.dumps({'canonical_equal': True, 'saved_files': len(rows),
                      'canonical_sha256': comparison['expected_canonical_sha256']}))


if __name__ == '__main__':
    main()
