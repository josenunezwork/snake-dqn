"""Create-only CAP05 packaging; no allocation or runtime execution."""
from __future__ import annotations
import copy, datetime, gzip, hashlib, json, os, pathlib, re, shlex, subprocess, sys, tarfile
R=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5-cap05')
B=R.parent/'snake-dqn-artifacts/big-lever/exp/astra-20261024'
P=B/'F5_W_CAPTURE_AFTER_TRANSPORT_REQUAL_05'
C=B/'F5_W_CAPTURE_AFTER_22e86986_REPAIR_04'
F=B/'F5_W_PREFIT_20261025/final-09'
PR=B/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02'
REG=R.parent/'snake-dqn-artifacts/ops/colab-cli-qualified.json'
PYTHON=R.parent/'snake-dqn/venv/bin/python'
PIN='e25f889b742764d29313340eecb41703e323d0dc'
IMPL='c989f6738f6aad143eb97bc47f6bd9e6687057ce'
TREE='6e8c1982b5c9ed115bfebd3145ad0720b15522fa'
JOB='bl-f5-w-capture-requal-3b32d755-05'
NAMES=['QUALIFIED_TRANSPORT_3b32d755.md','ACCEPT_TRANSPORT_REPAIR_7f5357a7.md','ACCEPT_F5_W_IMPLEMENTATION_c989f673.md','OWNER_RULING_COLAB_BUDGET_20261025.txt']
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE']})
os.environ['SNAKE_DQN_DEVICE']='cpu'
sys.path.insert(0,str(R))
from research.bl_ppo.exports import export_bytes,export_json,verify_export

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(pathlib.Path(p).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):
 p=pathlib.Path(p);assert p.is_file() and not p.is_symlink(),str(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':h,'size':p.stat().st_size}
def check(d):
 x=bind(d['path']);assert x['sha256']==d['sha256'],d['path']
 if 'size' in d:assert x['size']==d['size'],d['path']
def write(p,v):
 return export_bytes(p,v if isinstance(v,bytes) else v.encode()) if isinstance(v,(bytes,str)) else export_json(p,v)
def run(argv,timeout=55):
 v=subprocess.run([str(x) for x in argv],cwd=R,capture_output=True,text=True,timeout=timeout)
 d={'argv':[str(x) for x in argv],'finished_utc':now(),'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr};assert v.returncode==0,d;return d
def git(*args):return subprocess.check_output(['git',*args],cwd=R,timeout=30)
def command():return [str(R/'tools/ops/dispatch.sh'),JOB,str(P/'environment-capture.json'),'--broker',str(R),'--cwd',str(R),'--state-dir',str(P/'state-05'),'--dispatch-txt',str(P/'DISPATCH.txt')]
def fresh():
 for p in [P/'state-05',P/'output-05',REG.parent/'dispatch'/JOB,REG.parent/'manifests'/f'{JOB}.json',REG.parent/'logs'/f'{JOB}.log']:assert not p.exists(),str(p)
def refs(v):
 if isinstance(v,dict):
  if v.get('read') is False:return
  if {'path','sha256'}<=v.keys() and isinstance(v['path'],str) and pathlib.Path(v['path']).is_absolute():yield v
  for k,x in v.items():
   if k!='prohibited_references_not_followed':yield from refs(x)
 elif isinstance(v,list):
  for x in v:yield from refs(x)
def protected(p):return any('d4096' in x.lower() or x=='t34-round0-study-01' for x in pathlib.Path(p).parts)
phase=sys.argv[1]
if phase=='prepare':
 assert not P.exists();assert git('status','--porcelain')==b'';assert git('rev-parse','HEAD').decode().strip()==PIN
 assert git('rev-parse','origin/main').decode().strip()==PIN
 assert git('rev-parse',f'{PIN}:research/colab_cli').decode().strip()==TREE
 P.mkdir();fresh();write(P/'build-package.py',pathlib.Path(__file__).read_bytes())
 for name in NAMES:write(P/name,(B/name).read_bytes())
 write(P/'qualified-registry.json',REG.read_bytes())
 registry=load(REG);assert registry['tree_hash']==TREE and registry['qualified_commit'].startswith('3b32d755')
 result=load(registry['acceptance_result']);assert result['qualified'] is True and result['return_code']==0
 assert bind(registry['acceptance_result'])['sha256']==registry['acceptance_result_sha256']
 write(P/'qualified-acceptance-result.json',pathlib.Path(registry['acceptance_result']).read_bytes())
 from research.bl_residual.f5_package import source_inventory,source_review,governance
 hashes=source_inventory(R);old=load(F/'bootstrap-control.json')
 changed=[]
 for rel,h in hashes.items():
  raw=git('show',f'{PIN}:{rel}');assert sha(raw)==h
  before=git('show',f'{IMPL}:{rel}')
  if not rel.startswith('research/colab_cli/'):
   assert raw==before,rel
  elif raw!=before:changed.append(rel)
 f5files=git('ls-tree','-r','--name-only',IMPL,'research/bl_residual').decode().splitlines()
 f5files=[x for x in f5files if pathlib.Path(x).name.startswith('f5') and pathlib.Path(x).suffix in ['.py','.sh','.json','.lock'] and '/ledger/' not in x]
 for rel in f5files:assert (R/rel).read_bytes()==git('show',f'{IMPL}:{rel}'),rel
 write(P/'source-identity.json',{'status':'PASS','published_main_source':PIN,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'source_hashes':hashes,'f5_science_runtime_files_compared':{x:bind(R/x) for x in f5files},'nontransport_closure_byte_identical_to_c989f673':True,'transport_closure_changes':changed,'old_bootstrap_control':bind(F/'bootstrap-control.json'),'current_registry':bind(REG)})
 for name in ['timing-fixtures.json','allocation.json']:write(P/name,(F/name).read_bytes())
 assert len(load(P/'timing-fixtures.json')['fixtures'])==12
 write(P/'bootstrap-control.json',{**old,'source_hashes':hashes,'qualification_fixture_sha256':verify_export(P/'timing-fixtures.json')})
 allocation=sha(json.dumps({'namespace':'bl-residual-f5-w-20261025-v1','job_id':JOB,'source_commit':PIN,'bootstrap_control':bind(P/'bootstrap-control.json'),'frozen_allocation':bind(P/'allocation.json'),'transport_registry':bind(REG)},sort_keys=True,allow_nan=False).encode())
 write(P/'capture-allocation.json',{'schema':'bl-f5-capture-bootstrap-allocation/v1','namespace':'bl-residual-f5-w-20261025-v1','allocation':allocation,'source_commit':PIN,'bootstrap_control':bind(P/'bootstrap-control.json'),'frozen_allocation':bind(P/'allocation.json'),'job_id':JOB,'scope':'CAPTURE_ONLY_NOT_SCIENCE_ALLOCATION_OR_PLAN'})
 write(P/'verified-ledger.json',{'schema':'bl-f5-verified-ledger/v1','namespace':'bl-residual-f5-w-20261025-v1','allocation_sha256':load(F/'verified-ledger.json')['allocation_sha256'],'allocation':allocation,'status':'EMPTY_VERIFIED_LEDGER_NO_COMPLETED_SCIENCE_UNITS','units':[],'scope':'Capture-only bookkeeping binding; resume_exemption applies; ledger is not a runtime upload or scientific admission.'})
 # Preserve all capture history; copy both mechanical failures and raw accounting without inspecting outcomes.
 prior=load(C/'accounting-owner-ruling-04.json')
 receipt_paths=[pathlib.Path(x['receipt']['path']) for x in prior['prior_capture_history']['attempts']]
 receipt_paths += [C/'output-04/receipt.json',PR/'attempt01-terminal-receipt.json',PR/'timing-terminal-receipt-snapshot.json']
 rows=[]
 for i,path in enumerate(receipt_paths):
  d=load(path);assert d['stop_verified'] is True and d['cleanup_error'] is None
  name=f'history/receipt-{i+1:02d}.json';write(P/name,path.read_bytes())
  a=d['accounting'];before=d.get('usage_before',{});after=d.get('usage_after',{})
  decrease=before['balance']-after['balance'] if isinstance(before.get('balance'),(int,float)) and isinstance(after.get('balance'),(int,float)) else None
  rows.append({'original_receipt':bind(path),'receipt':bind(P/name),'job_id':d['job_id'],'status':d['status'],'failure_class':d['failure_class'],'stop_verified':True,'actual_elapsed_seconds':d['elapsed_seconds'],'raw_accounting':a,'usage_before':before,'usage_after':after,'observed_balance_decrease_units':decrease,'attributable_last_crosscheck_units':(a.get('last_crosscheck') or {}).get('attributable_account_debit_units'),'observed_rate_integrated_full_session_exposure_units':d['elapsed_seconds']*a['observed_rate_units_per_hour']/3600,'priced_full_session_exposure_units':d['elapsed_seconds']*a['priced_rate_units_per_hour']/3600})
 assert all(x['status']=='INCOMPLETE' and load(x['receipt']['path'])['exports']==[] for x in rows[-2:])
 write(P/'historical-accounting.json',{'schema':'bl-f5-capture-requal-full-historical-accounting/v1','created_utc':now(),'attempts':rows,'prior_capture_accounting':bind(C/'accounting-owner-ruling-04.json'),'prior_capture_accounting_verbatim':prior,'prior_round08_history':bind(PR/'historical-accounting.json'),'prior_round08_history_verbatim':load(PR/'historical-accounting.json'),'cumulative_session_elapsed_seconds':sum(x['actual_elapsed_seconds'] for x in rows),'sum_observed_account_balance_decreases_units':sum(x['observed_balance_decrease_units'] for x in rows if x['observed_balance_decrease_units'] is not None),'sum_full_session_observed_rate_exposure_units':sum(x['observed_rate_integrated_full_session_exposure_units'] for x in rows),'sum_full_session_priced_exposure_units':sum(x['priced_full_session_exposure_units'] for x in rows),'accounting_interpretation':'Session elapsed sum is not wall-clock. Account balance decreases are provider observations, not precise per-job attribution; rate integration/priced exposure/last crosschecks remain distinct. Missing final debit stays unknown. No prior history reset.','held_plan_attempt02_executed':False,'completed_science_units':0,'new_capture05_executed':False,'new_capture05_elapsed_seconds':0,'new_capture05_debit_units':0,'budgets_are_runaway_guards_only':True})
 write(P/'attempt04-provenance.json',{'schema':'bl-f5-old-capture-provenance-only/v1','matching_new_transport_canary':False,'raw_environment_reused':False,'seal':bind(C/'PACKAGE_SEAL_OPERATIONAL_V5.json'),'publication':bind(C/'OPERATIONAL_PUBLICATION_V5.json'),'terminal_receipt':bind(C/'output-04/receipt.json'),'post_capture_verification':bind(C/'POST_CAPTURE_VERIFICATION_04.md'),'independent_review':bind(C/'INDEPENDENT_CAPTURE_REVIEW_04.json'),'bindings':bind(C/'bindings-v5.json'),'bootstrap_control':bind(F/'bootstrap-control.json'),'source_review_packet':bind(F/'REVIEW_PACKET.json'),'source_review_bundle':bind(F/'REVIEW_BUNDLE.json')})
 write(P/'governance-source-preflight.json',{'status':'PASS','created_utc':now(),'governance':governance(R,B),'source_semantic_review':source_review(R),'allocations':0,'outcome_looks':0})
 write(P/'predecessor-inventory.json',{str(x):bind(x) for x in sorted(C.rglob('*')) if x.is_file()})
 bindings=copy.deepcopy(load(C/'bindings-v5.json'))
 # Old exact source closure is explicitly retained only in provenance, never claimed current.
 bindings.update({'created_utc':now(),'source_commit':PIN,'implementation_commit':IMPL,'qualified_transport_tree':TREE,'transport_rulings':{n:bind(P/n) for n in NAMES[:2]},'qualified_registry':bind(REG),'qualified_registry_snapshot':bind(P/'qualified-registry.json'),'qualified_acceptance':bind(P/'qualified-acceptance-result.json'),'bootstrap_control':bind(P/'bootstrap-control.json'),'timing_fixtures':bind(P/'timing-fixtures.json'),'frozen_allocation':bind(P/'allocation.json'),'capture_allocation':bind(P/'capture-allocation.json'),'empty_verified_ledger':bind(P/'verified-ledger.json'),'source_hashes':hashes,'exact_source_closure':{x:{'sha256':hashes[x],'size':(R/x).stat().st_size} for x in hashes},'attempt04_provenance_only':bind(P/'attempt04-provenance.json'),'historical_accounting':bind(P/'historical-accounting.json'),'two_mechanical_failure_receipts':[x['receipt'] for x in rows[-2:]],'scope':load(C/'environment-capture.json')['scope']})
 write(P/'bindings-v5.json',bindings)
 print('PREPARED',P,'closure',len(hashes),'F5 files',len(f5files),'changed transport closure',changed)
if phase=='preflight':
 fresh();assert git('status','--porcelain')==b''
 calls=[run(['/Users/josenunez/.local/bin/colab','--auth','oauth2',x]) for x in ['version','sessions','usage','sessions']]
 def sessions(s):
  rows=re.findall(r'^\[([^\]]+)\]\s+(\S+)\s+\| Hardware:\s*([^|\n]+)',s,re.M)
  if not rows:assert 'No active sessions found on server.' in s
  return {endpoint:{'session':name,'hardware':hardware.strip()} for name,endpoint,hardware in rows}
 first=sessions(calls[1]['stdout']);last=sessions(calls[3]['stdout']);assert first==last
 usage={}
 for k,label in [('balance','Current balance'),('rate','Usage rate'),('assignments','Active assignments')]:
  v=re.search(rf'{label}:\s*([0-9]+(?:\.[0-9]+)?)',calls[2]['stdout']);assert v;usage[k]=float(v[1])
 assert usage['assignments']==len(last)<3
 account=pathlib.Path('/Users/josenunez/.config/colab-cli/snake');pending=list(account.glob('pending-*.json'));assert not pending
 active=[bind(x) for x in sorted(account.glob('active-*.json'))];records={load(x['path'])['endpoint']:load(x['path']) for x in active};assert set(records)==set(last)
 rates={}
 for endpoint,row in last.items():
  rec=records[endpoint];assert rec['session']==row['session'];rate=rec.get('observed_rate_units_per_hour',rec.get('rate_units_per_hour'));assert type(rate) in [int,float] and rate>0;rates[endpoint]=rate
 assert abs(sum(rates.values())-usage['rate'])<0.01
 write(P/'live-account-preflight.json',{'status':'PASS','created_utc':now(),'calls':calls,'sessions':last,'usage':usage,'other_assignments':rates,'active_broker_records':active,'pending_broker_records':[],'registry':bind(REG),'account_assignment_cap':3,'allocation_calls':0,'own_rate_freshly_measured':False,'own_rate_rule':'Actual new L4 rate calibrated after allocation; historical 1.54 and priced ceiling 2 remain distinct.'})
 m=copy.deepcopy(load(C/'environment-capture.json'));m.update({'commit':PIN,'job_id':JOB,'output_dir':str(P/'output-05'),'supervisor_state_dir':str(P/'state-05')})
 m['source_paths']=sorted(load(P/'source-identity.json')['source_hashes'])+[str(x.relative_to(R)) for x in [R/'research/bl_residual/f5_adoption_environment_20261026_v6/requirements-f5-l4-cu126.lock',R/'research/bl_residual/f5_adoption_environment_20261026_v6/f5-common-environment-v6.json']]
 for d in m['inputs']:
  if d['dest'] in ['bootstrap-control.json','timing-fixtures.json']:d['path']=str(P/d['dest']);d['sha256']=verify_export(P/d['dest'])
 rows=[]
 for d in m['inputs']:
  x=bind(d['path']);assert x['sha256']==d['sha256'];rows.append({**d,'size':x['size'],'transport':'upload'})
 total=sum(x['size'] for x in rows);assert total<=45397643
 write(P/'inventory-metadata.json',{'schema':'bl-f5-complete-input-inventory/v1','rows':rows,'private_upload_bytes':total,'https_bytes':0,'total_input_bytes':total,'private_upload_cap_bytes':45397643,'private_cap_pass':True,'inspection':'Opaque file size/hash only; no checkpoint deserialization or selected training payload.'})
 m['f5'].update({'known_private_input_bytes':total,'remaining_private_bytes_for_plan_and_gate_metadata':45397643-total});m['known_private_input_bytes']=total
 m['budget']['other_assignments']=rates
 m['capture_accounting']={'historical_accounting':bind(P/'historical-accounting.json'),'allowance_reset':False,'admission_blocker':False}
 m['prior_capture_history']=load(P/'historical-accounting.json')
 m['operational_admission'].update({'bindings':bind(P/'bindings-v5.json'),'accounting':bind(P/'historical-accounting.json'),'live_account_preflight':bind(P/'live-account-preflight.json'),'transport_qualification_ruling':bind(P/NAMES[0]),'transport_repair_ruling':bind(P/NAMES[1]),'qualified_registry':bind(REG),'capture_allocation':bind(P/'capture-allocation.json'),'empty_verified_ledger':bind(P/'verified-ledger.json'),'attempt04_provenance_only':bind(P/'attempt04-provenance.json')})
 m['operational_admission'].pop('cumulative_envelope',None)
 write(P/'environment-capture.json',m)
 from research.colab_cli.supervisor import validate
 from research.colab_cli.source import prepare
 validate(m,R);plan=prepare(R,m,P/'source-preflight-artifacts')
 for x in sorted((P/'source-preflight-artifacts').iterdir()):
  if x.name.endswith(('.sha256','.complete')):continue
  d=bind(x)
  for suffix,data in [('.sha256',(d['sha256']+'  '+x.name+'\n').encode()),('.complete',b'complete\n')]:
   q=pathlib.Path(str(x)+suffix)
   if q.exists():assert q.read_bytes()==data
   else:
    with q.open('xb') as f:f.write(data)
 for rel,h in load(P/'source-identity.json')['source_hashes'].items():assert plan['files'][rel]['sha256']==h
 with tarfile.open(P/'source-preflight-artifacts/source.tar.gz') as archive:
  members=archive.getmembers();assert sorted(x.name for x in members)==sorted(plan['files'])
  for x in members:
   raw=archive.extractfile(x).read();assert sha(raw)==plan['files'][x.name]['sha256'];assert raw==git('show',f'{PIN}:{x.name}')==(R/x.name).read_bytes()
 write(P/'source-preflight.json',{'status':'PASS','created_utc':now(),'manifest':bind(P/'environment-capture.json'),'source_commit':PIN,'qualified_transport_tree':TREE,'source_plan':plan,'source_pin_count':len(load(P/'source-identity.json')['source_hashes']),'source_paths_count':len(m['source_paths']),'archive_members':len(members),'all_git_worktree_archive_bytes_match':True,'allocations':0})
 cmds=[run([PYTHON,R/'tools/ops/qualification.py','check','--repo',R,'--record',REG,'--ref',ref]) for ref in ['HEAD',PIN]]
 cmds.append(run([PYTHON,'-m','research.colab_cli.acceptance','--repo',R,'--verify-qualified-record',REG,'--required-inventory-sha256','b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea','--reject-tree','c148431277f4e33c3f589251df892ea158407e25']))
 cmds.append(run([R/'research/colab_cli/launch.sh','--state-dir',P/'state-05','--colab','/Users/josenunez/.local/bin/colab','validate',P/'environment-capture.json']))
 write(P/'broker-validation.json',{'status':'PASS','created_utc':now(),'commands':cmds,'broker':str(R),'cwd':str(R),'broker_commit':PIN,'clean':git('status','--porcelain')==b'','manifest':bind(P/'environment-capture.json'),'registry':bind(REG),'allocations':0})
 text=f'''Prepared for independent operative review and one future manual operator capture dispatch after publication to main.
Scope: environment capture plus twelve sealed disjoint source-native CUDA fixtures/observer/parity/transport only; v5 schema. No scientific fit, numerical outcome look, serving or release authority.
Job: {JOB}
Published main source: {PIN}; accepted F5 implementation: {IMPL}; qualified CLI subtree: {TREE}.
Manifest: {P/'environment-capture.json'} SHA256 {bind(P/'environment-capture.json')['sha256']}
Seal: {P/'PACKAGE_SEAL_OPERATIONAL_V5.json'} and SHA/completion sidecars; publication envelope separately binds seal.
Broker/cwd: clean detached checkout {R}; state {P/'state-05'}; output {P/'output-05'}; all must remain unused.
Transport authority: {P/NAMES[0]} and {P/NAMES[1]}; registry {REG} SHA256 {bind(REG)['sha256']}.
Attempt04 is historical provenance only, never a new-transport canary. All six prior session receipts/accounting retained, including both mechanical plan/timing failures; held plan attempt02 has not executed. No completed science units or allowance reset.
30-minute timeout/exemption: base15 x2, already effective; no redoubling. Estimate5 remains inherited forecast. Technical cap2/reserve1, base1/.5 x2, rate ceiling2u/h; owner ruling says budget isnt important pretendit doesnt exist for colab. These are runaway guards only.
One exact L4 >=6vCPU, cuda:0, Python3.12.3, same69-wheel/cu126 lock, --no-compile, deterministic fp32 and numerical threads1; OAuth2, account-wide maximum3 assignments, fresh calibration after allocation, lease180s, max export2GiB.
Exactly three uploads (B8, timing-fixtures.json, newly source-bound bootstrap-control.json), ten required exports. No selected-training payload. Capture-only allocation and empty ledger are bookkeeping bindings under short-canary exemption, not additional uploads or science plan.
Retain transfer<=150s and total staging<=300s, private bytes<=45397643, scientific N/namespace/selected training/ordered2000x256 CPU-generator/RNG contract and one-look discipline. No allocation/job/completion/automatic retry.
Account snapshot is timestamped preflight evidence only; real wrapper refreshes assignments/rates under lock. Recheck unused identity/output/state and registry before dispatch; changed declarations require a fresh successor. Independent operative review is required before operator dispatch.
After capture require fresh successful raw v5 environment, genuine native CUDA evidence matching this source/transport, immutable Mac-verified exports, terminal receipt and independent stop verification. Subsequent plan/timing and final science-inventory canary plus separate ACCEPT_F5_FIT remain required.
Exact operator command:
{shlex.join(command())}
'''
 write(P/'DISPATCH.txt',text)
 dr=run(command()+['--dry-run']);assert dr['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED';fresh()
 write(P/'dispatch-dry-run.json',{'status':'PASS','created_utc':now(),'command':dr,'manifest':bind(P/'environment-capture.json'),'dispatch_instructions':bind(P/'DISPATCH.txt'),'allocations':0,'dispatches':0,'state_absent':True,'output_absent':True})
 print(json.dumps({'status':'PASS','manifest':bind(P/'environment-capture.json'),'private_upload_bytes':total,'archive_members':len(members),'account':usage,'dry_run':dr['stdout'].strip()}))
