"""Create-only V5 offline verification and exact V4 machine diff; never dispatch."""
import json
import subprocess
from pathlib import Path

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import f5_engineering_seal, f5_step3_driver as driver
from research.bl_residual import f5_step3_package as package
from research.colab_cli.supervisor import Refused, validate

ROOT = Path('/Users/josenunez/Projects/ml/snake-dqn-f5rawenv')
OUTPUT = Path(__file__).resolve().parent
STAGED = OUTPUT / 'step3-package'
V4 = OUTPUT.with_name('f5-hold-corrections-v4') / 'step3-package'
SEAL = ROOT / 'research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026_V5/ENGINEERING_SEAL.json'


def diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [r for k in sorted(a.keys() | b.keys()) for r in
                diff(a.get(k), b.get(k), path + '/' + k.replace('~', '~0').replace('/', '~1'))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [r for i, (x, y) in enumerate(zip(a, b, strict=True))
                for r in diff(x, y, path + '/' + str(i))]
    return [] if a == b else [{'path': path, 'before': a, 'after': b}]


def read(p):
    return json.loads(p.read_text())


engineering = f5_engineering_seal.verify(SEAL, ROOT)
sealed = package.verify(STAGED)
result = driver.run(control_path=STAGED/'inputs/control.json',
                    fixtures=STAGED/'inputs/timing-fixtures.json', inputs=STAGED/'inputs',
                    output=STAGED/'runtime-output-attempt01',
                    ledger=STAGED/'inputs/verified-ledger.json', mode='dry-run')
export_json(STAGED/'DRIVER_DRY_RUN.json', result)
transport = {'dispatch_performed': False, 'runtime_calls': False}
for name in (package.QUALIFICATION_MANIFEST, 'manifest.json'):
    try:
        validate(read(STAGED/name), repo=ROOT)
    except Refused as error:
        assert name == 'manifest.json'
        assert str(error) == 'dispatch requires a SHA-bound family canary receipt'
        transport[name] = {'status': 'ADMISSION_REFUSED', 'reason': str(error)}
    else:
        assert name == package.QUALIFICATION_MANIFEST
        transport[name] = {'status': 'OFFLINE_SUPERVISOR_VALID'}
export_json(STAGED/'TRANSPORT_DRY_RUN.json', transport)
old_pins, new_pins = read(V4/'source-inventory.json'), read(STAGED/'source-inventory.json')
assert old_pins.keys() == new_pins.keys()
changed = {k for k in old_pins if old_pins[k] != new_pins[k]}
assert changed == {'research/bl_residual/f5_step3_driver.py',
                   'research/bl_residual/f5_step3_package.py',
                   'research/bl_residual/f5_engineering_seal.py'}
old, new = read(V4/'qualification-manifest-attempt02.json'), read(STAGED/package.QUALIFICATION_MANIFEST)
assert old['source_paths'] == new['source_paths']
assert [x['dest'] for x in old['inputs']] == [x['dest'] for x in new['inputs']]
manifest_diffs = diff(old, new)
allowed_top = {'job_id', 'commit', 'output_dir', 'supervisor_state_dir', 'inputs'}
assert all(row['path'].split('/')[1] in allowed_top for row in manifest_diffs)
changed_inputs = []
for before, after in zip(old['inputs'], new['inputs'], strict=True):
    assert before['dest'] == after['dest']
    role = before['dest']
    if before['sha256'] != after['sha256']:
        changed_inputs.append(role)
    if role not in {'control.json', 'verified-ledger.json', 'verified-ledger.json.sha256'}:
        assert before['sha256'] == after['sha256']
        assert before['size_bytes'] == after['size_bytes']
assert set(changed_inputs) == {'control.json','verified-ledger.json','verified-ledger.json.sha256'}
old_control,new_control=read(V4/'inputs/control.json'),read(STAGED/'inputs/control.json')
control_diffs=diff(old_control,new_control)
allowed_control={'namespace','source_hashes','source_sha256','source_provenance'}
assert all(r['path'].split('/')[1] in allowed_control for r in control_diffs)
assert [r for r in control_diffs if r['path'].startswith('/source_provenance/')] == [
    {'path':'/source_provenance/source_sha256','before':old_control['source_sha256'],
     'after':new_control['source_sha256']}]
ledger_diffs=diff(read(V4/'inputs/verified-ledger.json'),read(STAGED/'inputs/verified-ledger.json'))
assert {r['path'] for r in ledger_diffs} == {'/identity/namespace','/identity/source_sha256'}
assert digest(V4/'envelope.json') == digest(STAGED/'envelope.json')
# Same uploader, requirements, environment, scientific code and frozen bytes.
assert not any(p.startswith(('research/colab_cli/', 'tools/sandbox/', 'research/sandbox_')) for p in changed)
measurement_diffs=diff(read(V4/'manifest.json'),read(STAGED/'manifest.json'))
assert all(r['path'].split('/')[1] in allowed_top for r in measurement_diffs)
comparison={'schema':'f5-v5-vs-v4-machine-diff/v1','v4_qualification':package.reference(V4/'qualification-manifest-attempt02.json'),
            'v5_qualification':package.reference(STAGED/package.QUALIFICATION_MANIFEST),
            'qualification_manifest_diff':manifest_diffs,'measurement_template_diff':measurement_diffs,
            'control_diff':control_diffs,'ledger_diff':ledger_diffs,
            'source_inventory_diff':diff(old_pins,new_pins),
            'changed_payload_roles':changed_inputs,'envelope_byte_identical':True,
            'scientific_meaning_unchanged':True,'transport_source_unchanged':True,
            'runtime_calls':False,'dispatch_performed':False}
export_json(STAGED/'MACHINE_DIFF_VS_V4.json',comparison)
value={'schema':'bl-f5-hold-v5-offline-verification/v1',
       'status':'ENGINEERING_ONLY_HOLD_V5_PREPARED_NO_DISPATCH',
       'source_commit':read(STAGED/package.QUALIFICATION_MANIFEST)['commit'],
       'source_sha256':sealed['source_sha256'],'source_inventory_count':len(new_pins),
       'engineering_seal':package.reference(SEAL),
       'package':{name:package.reference(STAGED/name) for name in
                  ['PACKAGE_SEAL.json',package.QUALIFICATION_MANIFEST,'manifest.json',
                   'envelope.json','source-inventory.json','DRIVER_DRY_RUN.json',
                   'TRANSPORT_DRY_RUN.json','MACHINE_DIFF_VS_V4.json']},
       'immutable_historical_count':len(engineering['immutable_old_inventory']),
       'dry_run':result,'transport_dry_run':transport,
       'scientific_look_performed':False,'cuda_executed':False,'dispatch_performed':False,
       'independent_acceptance':'NOT_CREATED','operative_fit_acceptance':False,
       'prior_attempt02_evidence':package.reference(ROOT/'research/bl_residual/ledger/F5_STEP1_ATTEMPT02_20261026/PUBLICATION.json'),
       'offline_script':package.reference(Path(__file__).resolve())}
export_json(OUTPUT/'OFFLINE_VERIFICATION.json',value)
print(json.dumps({k:value[k] for k in ['status','source_commit','source_sha256','source_inventory_count','immutable_historical_count']}))
print(json.dumps({'engineering_seal':digest(SEAL),'package_seal':digest(STAGED/'PACKAGE_SEAL.json'),
                  'qualification_manifest':digest(STAGED/package.QUALIFICATION_MANIFEST),
                  'machine_diff':digest(STAGED/'MACHINE_DIFF_VS_V4.json')}))
