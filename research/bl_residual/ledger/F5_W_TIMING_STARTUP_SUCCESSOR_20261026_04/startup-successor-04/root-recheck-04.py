"""Recheck existing immutable timing04 successor; no allocation interface."""
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

from research.bl_ppo.exports import export_bytes, export_json, verify_export
from tools.ops import f5_timing_guard_successor as h

P = h.PACKAGE
E = P / 'startup-successor-04'
spec = importlib.util.spec_from_file_location('timing04_builder', E / 'build-successor-04-v3.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.fresh()
counts = b.inherited()
a = h.load(P / 'manifest-03.json')
z = h.load(P / 'manifest-04.json')
d = h.load(P / 'DIFF_03_04.json')
assert b.diff(a, z) == d['changes']
allowed = {'/job_id', '/output_dir', '/supervisor_state_dir', *('/operational_admission/live_account_preflight/' + k for k in ['path', 'sha256', 'size'])}
assert {r['path'] for r in d['changes']} == allowed
for q in [P / 'manifest-04.json', P / 'DISPATCH-04.txt', P / 'DIFF_03_04.json', P / 'cumulative-accounting-04.json']:
    verify_export(q)
for mutation in ['added_key', 'type_change', 'list_entry']:
    zz = copy.deepcopy(z)
    if mutation == 'added_key': zz['extra'] = None
    elif mutation == 'type_change': zz['timeout_minutes'] = float(zz['timeout_minutes'])
    else: zz['source_paths'][0] = 'forbidden'
    try:
        changed = b.diff(a, zz)
        assert {r['path'] for r in changed} <= allowed
    except AssertionError:
        pass
    else:
        raise AssertionError('negative diff accepted ' + mutation)
commands = []
for prior in h.load(E / 'GATES-04.json')['commands']:
    commands.append(h.run(prior['argv']))
assert commands[4]['stdout'].strip() == f'PASS_DRY_RUN {b.JOB} allocations=0 live_accounting=NOT_CHECKED'
source = h.load(P / 'source-preflight.json')['source_plan']['files']
for rel, ref in source.items():
    raw = subprocess.check_output(['git','show',z['commit'] + ':' + rel], cwd=h.ROOT, timeout=30)
    assert hashlib.sha256(raw).hexdigest() == ref['sha256']
    assert h.binding(h.BROKER / rel)['sha256'] == ref['sha256']
fail = h.load(b.F / 'FAILURE_EVIDENCE.json')
for row in fail['files'].values():
    h.check(row['source']); h.check(row['published'])
    rel = str(Path(row['published']['path']).relative_to(h.ROOT))
    raw = subprocess.check_output(['git','show','origin/main:' + rel], cwd=h.ROOT, timeout=30)
    assert hashlib.sha256(raw).hexdigest() == row['published']['sha256']
installed = h.run(['/Users/josenunez/.venvs/colab-cli/bin/python3.12', E / 'verify-installed-cli-records-v2.py'])
current = json.loads(installed['stdout'])
assert current == h.load(E / 'CLI_DEPENDENCY_INTEGRITY-04.json')
acc = h.load(P / 'cumulative-accounting-04.json')
assert acc['cumulative_session_elapsed_seconds'] == 5055.5119253752055
assert acc['prior_raw_accounting'] == h.load(P / 'cumulative-accounting-final.json')
for row in acc['appended_receipts']:
    h.check(row['binding']); assert row['raw_receipt'] == h.load(Path(row['binding']['path']))
assert acc['aggregate_effective_wall_minutes'] == 240
assert acc['per_attempt_effective_minutes'] == z['timeout_minutes'] == 30
assert z['budget']['unit_cap'] == 100 and z['budget']['reserve_units'] == 0
pins = []
for record in subprocess.check_output(['git', 'worktree', 'list', '--porcelain'],cwd=h.ROOT,text=True).split('\n\n'):
    lines = record.splitlines()
    if 'HEAD ' + z['commit'] in lines:
        path = Path(lines[0].removeprefix('worktree '))
        status = subprocess.check_output(['git','status','--porcelain'],cwd=path,text=True)
        assert not status, str(path)
        pins.append(str(path))
assert pins, 'no clean source checkout found'
b.fresh()
export_bytes(E / 'root-recheck-04.py', Path(__file__).read_bytes())
export_json(E / 'ROOT_RECHECK-04.json', {
    'schema':'bl-f5-timing04-root-recheck/v1', 'status':'PASS', 'created_utc':h.now(),
    'commands':commands, 'changed_leaves':d['changes'], 'negative_diff_probes':3,
    'inherited_counts':counts, 'source_pins_verified':len(source), 'clean_source_worktrees':pins,
    'failure_publication_verified_on_origin_main':True,
    'installed_cli_record_integrity_identical':True, 'cli_version':current['cli_version'],
    'verified_installed_hashed_files':current['verified_hashed_files'],
    'cumulative_elapsed_seconds':acc['cumulative_session_elapsed_seconds'],
    'allocations':0, 'dispatches':0, 'runtime_calls':0,
    'live_accounting':'Final locked preseal refresh and normal dispatch reconciliation required',
})
print('PASS root recheck; source pins',len(source),'inherited',counts,'clean source',pins)
