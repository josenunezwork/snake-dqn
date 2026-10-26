"""Create-only preparation/publication of Astra's single mechanical timing04 successor."""
import copy
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

from research.bl_ppo.exports import export_bytes, export_json, verify_export
from tools.ops import f5_timing_guard_successor as h
from tools.ops.dispatch import allocation_lock

P = h.PACKAGE
E = P / 'startup-successor-04'
D = h.ROOT / 'research/bl_residual/ledger/F5_W_TIMING_STARTUP_SUCCESSOR_20261026_04'
F = h.ROOT / 'research/bl_residual/ledger/F5_W_TIMING03_FAILURE_20261026'
RULING = h.BASE / 'F5_W_TIMING03_DISPOSITION_20261026.md'
JOB = 'bl-f5-w-timing-r08-sr75b14427-04'
B = h.binding
L = h.load
CHECK = h.check


def fresh():
    assert h.run(['git','rev-parse','HEAD'])['stdout'].strip() == 'ac69b65c3a2eb40a2af30ab6a9e8e4bd450471ef'
    assert not h.run(['git','status','--porcelain'])['stdout']
    for path in (P/'state-03',P/'output-03',h.OPS/'dispatch'/JOB,h.OPS/'manifests'/f'{JOB}.json',h.OPS/'logs'/f'{JOB}.log'):
        assert not path.exists(), path


def cmd():
    return [str(h.BROKER/'tools/ops/dispatch.sh'),JOB,str(P/'manifest-04.json'),'--broker',str(h.BROKER),'--cwd',str(h.BROKER),'--state-dir',str(P/'state-03'),'--dispatch-txt',str(P/'DISPATCH-04.txt')]


def account():
    with allocation_lock(h.OPS/'alloc.lock',180):
        value=h.account_preflight()
        value['serialized_ops_allocation_lock']=str(h.OPS/'alloc.lock')
        value['refuse_drift_before_dispatch']=True
        return value


def diff(a,b,p=''):
    assert type(a) is type(b), ('TYPE_CHANGE',p,type(a).__name__,type(b).__name__)
    if isinstance(a,dict):
        assert a.keys()==b.keys(), ('KEY_CHANGE',p,sorted(a.keys()^b.keys()))
        return [r for k in sorted(a) for r in diff(a[k],b[k],p+'/'+k.replace('~','~0').replace('/','~1'))]
    if isinstance(a,list):
        assert len(a)==len(b), ('LIST_LENGTH_CHANGE',p)
        return [r for i,(x,y) in enumerate(zip(a,b)) for r in diff(x,y,p+'/'+str(i))]
    return [] if a==b else [{'path':p,'type':type(a).__name__,'before':a,'after':b}]


def failure():
    fresh()
    assert not F.exists()
    r=L(P/'output-02/receipt.json')
    assert B(P/'output-02/receipt.json')['sha256']=='5f16ba02f705e6c918000392884af5dee304b9060b88e136f64c70f1522ebeab'
    assert B(P/'output-02/warmup-measurement.json')['sha256']=='bd25e43ec1821f587cf46cabf46b2e6e5c6b88c384079eb79aa4f2f3ef9d6836'
    assert r['status']=='INCOMPLETE' and r['failure_class']=='SUPERVISOR'
    assert r['stop_verified'] is True and r['cleanup_error'] is None and r['terminal'] is None
    assert all(r[k]==[] for k in ('transfers','contents_transfers','control_operations','exports'))
    assert 'kernel creation already attempted; refusing uncertain retry' in r['error']
    assert [x['command'] for x in r['cli_calls']].count('new')==1
    assert [x['command'] for x in r['cli_calls']].count('exec')==1
    assert r['usage_after']['assignments']==r['usage_after']['rate']==0
    assert B(P/'output-02/manifest.json')['sha256']==r['manifest_sha256']
    files={}
    for path in sorted((P/'output-02').rglob('*')):
        if path.is_file():
            target=F/'output-02'/path.relative_to(P/'output-02')
            export_bytes(target,path.read_bytes())
            files[str(path.relative_to(P))]={'source':B(path),'published':B(target)}
    export_bytes(F/RULING.name,RULING.read_bytes())
    export_json(F/'FAILURE_EVIDENCE.json',{'schema':'bl-f5-timing03-failure-publication/v1','status':'INCOMPLETE_STARTUP_KERNEL_CREATION_UNCERTAINTY','files':files,'receipt':B(P/'output-02/receipt.json'),'warmup':B(P/'output-02/warmup-measurement.json'),'disposition':B(RULING),'stop_verified':True,'source_staged':False,'useful_child_executed':False,'provider_root_cause':'UNKNOWN','balance_change_is_not_zero_cost_proof':True,'allocations_during_preparation':0})
    print('FAILURE_PUBLISHED',B(F/'FAILURE_EVIDENCE.json'))


def prepare():
    fresh()
    assert not E.exists()
    baseline={str(x.relative_to(P)):B(x) for x in sorted(P.rglob('*')) if x.is_file()}
    export_json(E/'predecessor-inventory.json',baseline)
    export_bytes(E/'build-successor-04.py',Path(__file__).read_bytes())
    export_bytes(E/RULING.name,RULING.read_bytes())
    export_json(E/'failure-publication-binding.json',B(F/'FAILURE_EVIDENCE.json'))
    old=L(P/'manifest-03.json')
    assert B(P/'manifest-03.json')['sha256']=='88314d5ae92948005c6d7815ac278e6406ce1090c5694d15776b36f5cc92fa39'
    prior=L(P/'cumulative-accounting-final.json')
    receipts=[L(P/f'output-{n:02d}/receipt.json') for n in (1,2)]
    elapsed=prior['cumulative_session_elapsed_seconds']+sum(r['elapsed_seconds'] for r in receipts)
    assert elapsed==5055.5119253752055
    cumulative={'schema':'bl-f5-timing04-cumulative-accounting/v1','prior_accounting':B(P/'cumulative-accounting-final.json'),'prior_raw_accounting':prior,'appended_receipts':[{'binding':B(P/f'output-{n:02d}/receipt.json'),'raw_receipt':r} for n,r in zip((1,2),receipts)],'timing_attempts_01_03':[{'attempt':1,'receipt':B(h.BASE/'F5_W_TIMING_OPERATIONAL_20261026_ROUND08_01/output-01/receipt.json'),'included_in_prior_history':True},{'attempt':2,'receipt':B(P/'output-01/receipt.json'),'preallocation_refusal':True},{'attempt':3,'receipt':B(P/'output-02/receipt.json'),'stop_verified':True}], 'cumulative_session_elapsed_seconds':elapsed,'aggregate_effective_wall_minutes':240,'remaining_aggregate_wall_seconds':240*60-elapsed,'base_hours':2,'effective_hours':4,'per_attempt_base_minutes':15,'per_attempt_effective_minutes':30,'forecast_minutes':15,'base_unit_guard':50,'effective_unit_guard':100,'base_reserve_units':0,'effective_reserve_units':0,'first_cap_wins':True,'no_redoubling_or_clock_debit_reset':True,'sum_observed_balance_decreases_units':prior['sum_observed_balance_decreases_units']+sum(r['usage_before']['balance']-r['usage_after']['balance'] for r in receipts if r['usage_before'] and r['usage_after']),'sum_full_session_priced_exposure_units':prior['sum_full_session_priced_exposure_units']+sum(r['elapsed_seconds']*2/3600 for r in receipts),'attempt03_observed_rate_not_calibrated':True,'attempt03_illustrative_registered_rate_exposure_units':receipts[1]['elapsed_seconds']*1.54/3600,'interpretation':'Conservative receipt elapsed exposure includes refusal time; balance changes, calibrated exposure and priced exposure are separate. Unchanged balance does not establish zero cost. All historical raw accounting preserved. No outcome look.'}
    export_json(P/'cumulative-accounting-04.json',cumulative)
    export_json(E/'live-account-preflight-04.json',account())
    new=copy.deepcopy(old)
    new.update(job_id=JOB,output_dir=str(P/'output-03'),supervisor_state_dir=str(P/'state-03'))
    new['operational_admission']['live_account_preflight']=B(E/'live-account-preflight-04.json')
    assert new['budget']['other_assignments']==L(E/'live-account-preflight-04.json')['other_assignments']=={}
    changes=diff(old,new)
    allowed={'/job_id','/output_dir','/supervisor_state_dir',*(f'/operational_admission/live_account_preflight/{k}' for k in ('path','sha256','size'))}
    assert {r['path'] for r in changes} <= allowed
    assert {'/job_id','/output_dir','/supervisor_state_dir','/operational_admission/live_account_preflight/path','/operational_admission/live_account_preflight/sha256'} <= {r['path'] for r in changes}
    # Negative probes prove key/type/list-entry rejection, including absent-null and bool-int.
    for mutate in ('key','type','list'):
        bad=copy.deepcopy(new)
        if mutate=='key': bad['extra']=None
        if mutate=='type': bad['timeout_minutes']=30.0
        if mutate=='list': bad['required_exports'][0]='unauthorized'
        try:
            rows=diff(old,bad)
            assert {r['path'] for r in rows} <= allowed
        except AssertionError:
            pass
        else:
            raise AssertionError('negative diff probe failed: '+mutate)
    export_json(P/'manifest-04.json',new)
    export_json(P/'DIFF_03_04.json',{'schema':'bl-f5-exhaustive-mechanical-leaf-diff/v1','status':'PASS','before':B(P/'manifest-03.json'),'after':B(P/'manifest-04.json'),'changes':changes,'permitted_leaves':sorted(allowed),'object_keys_types_list_lengths_and_every_entry_checked':True,'added_fields':[],'deleted_fields':[],'type_changes':[],'non_whitelisted_changes':[],'negative_probes':['added-null key rejected','int-to-float rejected','list item nonwhitelist rejected'],'other_assignments_unchanged_observed_empty':True,'disposition':B(RULING)})
    text=(P/'DISPATCH-03.txt').read_text()
    for a,b in ((old['job_id'],JOB),('manifest-03.json','manifest-04.json'),('DISPATCH-03.txt','DISPATCH-04.txt'),('state-02','state-03'),('output-02','output-03'),('PACKAGE_SEAL_OPERATIONAL-03.json','PACKAGE_SEAL_OPERATIONAL-04.json'),(B(P/'manifest-03.json')['sha256'],B(P/'manifest-04.json')['sha256']),(str(P/'guard-successor-03/live-account-preflight-03.json'),str(E/'live-account-preflight-04.json'))):
        text=text.replace(a,b)
    text=text.replace('PREPARED: single future operator full timing/parity/calibration dispatch requires independent review and sealed package publication to main.','PREPARED: one fresh full timing/parity/calibration attempt 04 conditionally permitted by F5_W_TIMING03_DISPOSITION_20261026.md after all listed gates and sealed package publication to main.')
    text+='\nAppend-only timing04 lineage: attempt03 remains INCOMPLETE with verified stop. Astra disposition '+str(RULING)+' SHA256 '+B(RULING)['sha256']+' permits exactly one fresh attempt04 after the exhaustive DIFF_03_04.json and all gates pass, without another review. Earlier successor prose is historical. Operative cumulative-accounting-04.json carries attempts01-03 and prior history: 5055.5119253752055 seconds against unchanged 240-minute aggregate; no cap reset or redoubling. Delta seal binds both latest receipts, disposition, diff and cumulative accounting. Reconcile the sealed account map, all pending/active reservations, positive rates, maximum3 assignments and strict balance threshold under normal serialized allocation immediately before dispatch; refuse drift. Preparation created no runtime and performed no dispatch.\n'
    export_bytes(P/'DISPATCH-04.txt',text.encode())
    print('PREPARED',B(P/'manifest-04.json'), 'CHANGES',[r['path'] for r in changes])


def inherited():
    counts={}
    for ref in L(E/'predecessor-inventory.json').values(): CHECK(ref)
    counts['predecessor_files']=len(L(E/'predecessor-inventory.json'))
    for name in ('PACKAGE_SEAL_OPERATIONAL.json','PACKAGE_SEAL_OPERATIONAL-03.json'):
        for ref in L(P/name)['files'].values(): CHECK(ref)
        counts[name]=len(L(P/name)['files'])
    for name in ('PACKAGE_PUBLICATION.json','PACKAGE_PUBLICATION-03.json'):
        pub=L(P/name)
        for ref in pub.get('original_files',{}).values(): CHECK(ref)
        for ref in pub['published_files'].values():
            CHECK(ref)
            rel=str(Path(ref['path']).relative_to(h.ROOT))
            raw=subprocess.check_output(['git','show','origin/main:'+rel],cwd=h.ROOT,timeout=30)
            assert hashlib.sha256(raw).hexdigest()==ref['sha256'],rel
        counts[name]=len(pub['published_files'])
    for ref in L(F/'FAILURE_EVIDENCE.json')['files'].values(): CHECK(ref['source']); CHECK(ref['published'])
    return counts


def gates():
    fresh()
    counts=inherited()
    m=L(P/'manifest-04.json')
    commands=[h.run(['python3',h.BROKER/'tools/ops/qualification.py','check','--repo',h.BROKER,'--record',h.REGISTRY,'--ref',ref]) for ref in ('HEAD',m['commit'])]
    prior=L(P/'qualified-consumer-preflight.json')['command']['argv']
    commands.append(h.run(prior))
    commands.append(h.run([h.BROKER/'research/colab_cli/launch.sh','--state-dir',P/'state-03','--colab','/Users/josenunez/.local/bin/colab','validate',P/'manifest-04.json']))
    commands.append(h.run(cmd()+['--dry-run']))
    assert commands[-1]['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
    source=L(P/'source-preflight.json')['source_plan']['files']
    for relative,ref in source.items():
        raw=subprocess.check_output(['git','show',m['commit']+':'+relative],cwd=h.ROOT,timeout=30)
        assert hashlib.sha256(raw).hexdigest()==ref['sha256'],relative
        assert B(h.BROKER/relative)['sha256']==ref['sha256'],relative
    assert not h.run(['git','status','--porcelain'])['stdout']
    commands.append(h.run(['git','merge-base','--is-ancestor',m['commit'],'origin/main'],cwd=h.ROOT))
    export_json(E/'GATES-04.json',{'status':'PASS','commands':commands,'inherited_verification_counts':counts,'source_pin_count':len(source),'source_commit':m['commit'],'broker_commit':h.run(['git','rev-parse','HEAD'])['stdout'].strip(),'qualified_registry':B(h.REGISTRY),'allocations':0,'dispatches':0,'runtime_calls':0,'scientific_payloads_opened':0,'dispatch_time_account_reconciliation_required':True})
    print('GATES PASS',counts,'source pins',len(source))


def seal():
    fresh(); inherited()
    assert L(E/'GATES-04.json')['status']=='PASS'
    assert L(E/'INDEPENDENT_REVIEW-04.json')['status']=='PASS'
    export_json(E/'account-recheck-before-seal-04.json',account())
    assert L(E/'account-recheck-before-seal-04.json')['other_assignments']==L(E/'live-account-preflight-04.json')['other_assignments']
    files={}
    for path in sorted(list(E.rglob('*'))+[P/n for n in ('manifest-04.json','DISPATCH-04.txt','DIFF_03_04.json','cumulative-accounting-04.json')]):
        if path.is_file() and not path.name.endswith(('.sha256','.complete')):
            verify_export(path)
            for suffix in ('','.sha256','.complete'):
                q=Path(str(path)+suffix);files[str(q.relative_to(P))]=B(q)
    export_json(P/'PACKAGE_SEAL_OPERATIONAL-04.json',{'schema':'bl-f5-mechanical-successor-delta-seal/v1','status':'SEALED_TIMING04_MECHANICAL_SUCCESSOR_NO_DISPATCH','created_utc':h.now(),'job_id':JOB,'disposition':B(RULING),'inherited_seal':B(P/'PACKAGE_SEAL_OPERATIONAL-03.json'),'inherited_publication':B(P/'PACKAGE_PUBLICATION-03.json'),'failure_publication':B(F/'FAILURE_EVIDENCE.json'),'latest_receipts':[B(P/f'output-{n:02d}/receipt.json') for n in (1,2)],'diff':B(P/'DIFF_03_04.json'),'cumulative_accounting':B(P/'cumulative-accounting-04.json'),'files':files,'state_absent':True,'output_absent':True,'science_change':False,'allocations':0,'dispatches':0,'self_exclusions':['PACKAGE_SEAL_OPERATIONAL-04.json and companions','PACKAGE_PUBLICATION-04.json and companions']})
    print('SEALED',B(P/'PACKAGE_SEAL_OPERATIONAL-04.json'))


def publish():
    fresh()
    assert not D.exists()
    rows=dict(L(P/'PACKAGE_SEAL_OPERATIONAL-04.json')['files'])
    for suffix in ('','.sha256','.complete'):
        path=P/('PACKAGE_SEAL_OPERATIONAL-04.json'+suffix);rows[str(path.relative_to(P))]=B(path)
    published={}
    for rel,ref in rows.items():
        CHECK(ref);target=D/rel;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream: stream.write(Path(ref['path']).read_bytes())
        published[rel]=B(target);assert published[rel]['sha256']==ref['sha256']
    export_json(P/'PACKAGE_PUBLICATION-04.json',{'schema':'bl-f5-mechanical-successor-delta-publication/v1','status':'PUBLISHED_TIMING04_REPRESENTATION_NO_DISPATCH','source_package':str(P),'ledger_directory':str(D),'inherited_publication':B(P/'PACKAGE_PUBLICATION-03.json'),'failure_publication':B(F/'FAILURE_EVIDENCE.json'),'seal':B(P/'PACKAGE_SEAL_OPERATIONAL-04.json'),'original_files':rows,'published_files':published,'dispatch_command':shlex.join(cmd()),'dispatches':0})
    for suffix in ('','.sha256','.complete'):
        path=P/('PACKAGE_PUBLICATION-04.json'+suffix)
        with (D/path.name).open('xb') as stream: stream.write(path.read_bytes())
    print('PUBLISHED',B(P/'PACKAGE_PUBLICATION-04.json'))


if __name__=='__main__':
    {'failure':failure,'prepare':prepare,'gates':gates,'seal':seal,'publish':publish}[sys.argv[1]]()
