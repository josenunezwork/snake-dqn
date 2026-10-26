"""Create-only round08 plan package. Never allocates or dispatches a runtime."""
from __future__ import annotations
import copy, datetime, gzip, hashlib, json, os, pathlib, re, shlex, subprocess, sys, tarfile, time
R=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5w')
B=R.parent/'snake-dqn-artifacts/big-lever/exp/astra-20261024'
D=B/'F5_W_NEXT_AFTER_ATTEMPT04_DRAFT_20261026_v2'
P=B/'F5_W_TIMING_OPERATIONAL_20261026_ROUND08_01'
PLAN=B/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_01'
T=B/'F5_W_TIMING_DRAFT_20261026_ROUND08_v3'
C=B/'F5_W_CAPTURE_AFTER_22e86986_REPAIR_04'
F=B/'F5_W_PREFIT_20261025/final-09'
PIN='c989f6738f6aad143eb97bc47f6bd9e6687057ce'
JOB='bl-f5-w-timing-round08-c989f673-01'
PYTHON=R.parent/'snake-dqn/venv/bin/python'
REG=R.parent/'snake-dqn-artifacts/ops/colab-cli-qualified.json'
RULINGS={'acceptance':'ACCEPT_F5_W_NEXT_PACKAGES_20261026_ROUND08.md','capture_qualification':'F5_W_CAPTURE_TRANSPORT_QUALIFIED_20261026.md','parallel_clarification':'F5_W_ROUND08_PARALLEL_SEALING_CLARIFICATION_20261026.md','owner_ruling':'OWNER_RULING_COLAB_BUDGET_20261025.txt'}
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE']})
os.environ['SNAKE_DQN_DEVICE']='cpu'
sys.path.insert(0,str(R))
from research.bl_ppo.exports import export_bytes, export_json, verify_export

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(pathlib.Path(p).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):
 p=pathlib.Path(p);assert p.is_file() and not p.is_symlink(),str(p)
 with p.open('rb') as s:h=hashlib.file_digest(s,'sha256').hexdigest()
 return {'path':str(p),'sha256':h,'size':p.stat().st_size}
def check(d):
 x=bind(d['path']);assert x['sha256']==d['sha256'],d['path']
 if 'size' in d:assert x['size']==d['size'],d['path']
def write(p,v):return export_bytes(p,v if isinstance(v,bytes) else v.encode()) if isinstance(v,(str,bytes)) else export_json(p,v)
def run(argv,timeout=55):
 start=now();v=subprocess.run([str(x) for x in argv],cwd=R,capture_output=True,text=True,timeout=timeout)
 row={'argv':[str(x) for x in argv],'started_utc':start,'finished_utc':now(),'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr}
 assert v.returncode==0,row
 return row
def git(*a):return subprocess.check_output(['git',*a],cwd=R,timeout=30)
def command():return [str(R/'tools/ops/dispatch.sh'),JOB,str(P/'manifest.json'),'--broker',str(R),'--cwd',str(R),'--state-dir',str(P/'state-01'),'--dispatch-txt',str(P/'DISPATCH.txt')]
def fresh():
 for p in [P/'state-01',P/'output-01',REG.parent/'dispatch'/JOB,REG.parent/'manifests'/f'{JOB}.json',REG.parent/'logs'/f'{JOB}.log']:assert not p.exists(),str(p)
def inv(m):
 rows=[]
 for d in m['inputs']:
  if 'url' in d:rows.append({**d,'transport':'https','size':d['size_bytes']})
  else:
   x=bind(d['path']);assert x['sha256']==d['sha256'];rows.append({**d,'size':x['size'],'transport':'upload'})
 private=sum(x['size'] for x in rows if x['transport']=='upload');public=sum(x['size'] for x in rows if x['transport']=='https')
 assert private<=45397643
 return {'schema':'bl-f5-complete-input-inventory/v1','rows':rows,'private_upload_bytes':private,'https_bytes':public,'total_input_bytes':private+public,'private_upload_cap_bytes':45397643,'private_cap_pass':True,'inspection':'Opaque hashes/sizes only; no label/checkpoint deserialization.'}
def update_counts(m,i):
 m['known_private_input_bytes']=i['private_upload_bytes']
 m['f5'].update({'known_private_input_bytes':i['private_upload_bytes'],'https_input_bytes':i['https_bytes'],'total_input_bytes':i['total_input_bytes'],'remaining_private_bytes_for_plan_and_gate_metadata':45397643-i['private_upload_bytes']})
def refs(v):
 if isinstance(v,dict):
  if v.get('read') is False:return
  if {'path','sha256'}<=v.keys() and isinstance(v['path'],str) and pathlib.Path(v['path']).is_absolute():yield v
  for k,x in v.items():
   if k!='prohibited_references_not_followed':yield from refs(x)
 elif isinstance(v,list):
  for x in v:yield from refs(x)
def protected(p):return any('d4096' in x.lower() or x=='t34-round0-study-01' for x in pathlib.Path(p).parts)
def write_once(p,v):
 if p.exists():verify_export(p)
 else:write(p,v)
phase=sys.argv[1]
if phase=='prepare':
 assert not P.exists();P.mkdir();fresh()
 write(P/'build-package.py',pathlib.Path(__file__).read_bytes())
 write(P/'predecessor-inventory.json',{str(x):bind(x) for root in [T,D] for x in sorted(root.rglob('*')) if x.is_file()})
 for name in RULINGS.values():write(P/name,(B/name).read_bytes())
 c=load(T/'control.json');c.pop('draft_name');c['status']='ACCEPTED_FULL_TIMING_PARITY_AND_CALIBRATION_ONLY_NO_SCIENTIFIC_FIT';c['operative_package']=str(P);c['scientific_admission']=False;c['operative_authority']={k:bind(P/v) for k,v in RULINGS.items()}
 c['allocation']=sha(json.dumps({k:v for k,v in c.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 l={'schema':'bl-f5-verified-ledger/v1','namespace':c['namespace'],'allocation_sha256':c['identity']['allocation_sha256'],'allocation':c['allocation'],'status':'EMPTY_VERIFIED_LEDGER_NO_COMPLETED_UNITS','units':[]}
 write(P/'verified-ledger.json',l);c['ledger_sha256']=verify_export(P/'verified-ledger.json');write(P/'control.json',c)
 write(P/'historical-accounting.json',load(D/'historical-accounting.json'))
 bindings=load(PLAN/'bindings.json');bindings['schema']='bl-f5-timing-operative-bindings/v1';bindings['rulings']={k:bind(P/v) for k,v in RULINGS.items()};bindings['timing_predecessor_manifest']=bind(T/'manifest.json');bindings['timing_predecessor_publication']=bind(T/'PACKAGE_PUBLICATION.json');bindings['plan_package_publication']=bind(PLAN/'PACKAGE_PUBLICATION.json');bindings['plan_published_commit']='fd7361088dc8445ddc0a4a15f019931a0122171d'
 write(P/'bindings.json',bindings)
 from research.bl_residual.f5_package import governance,source_review
 write(P/'governance-source-preflight.json',{'schema':'bl-f5-governance-source-preflight/v1','status':'PASS','created_utc':now(),'governance':governance(R,B),'source_semantic_review':source_review(R),'no_protected_payload_read':True,'no_scientific_outcome_read':True,'allocations':0})
 print('PREPARED',P)
if phase=='preflight':
 fresh()
 from research.bl_residual.f5_environment import validate_environment,compare_environment
 from research.bl_residual.f5_job import _control,_source_admission
 c=_control(P);_source_admission(c);actual=load(C/'output-04/exports/environment.json');assert c['environment_capture']==actual
 validate_environment(actual);comparison=compare_environment(actual,c['environment_capture'],mode='cross-assignment')
 write_once(P/'environment-reference-preflight.json',{'status':'PASS','created_utc':now(),'control':bind(P/'control.json'),'actual_attempt04_environment':bind(C/'output-04/exports/environment.json'),'exact_raw_object_equality':True,'comparison':comparison,'boundary_implementation':bind(R/'research/bl_residual/f5_job.py'),'new_runtime_environment':'Future operator must capture and validate cross-assignment plus before/after boundaries; not observed here.'})
 m=load(T/'manifest.json');m.pop('draft_only');m['job_id']=JOB;m['output_dir']=str(P/'output-01');m['supervisor_state_dir']=str(P/'state-01');m['lease_seconds']=180
 m['inventory_status']='FINAL_COMPLETE_METADATA_RECONCILED_HTTPS_B8_VERIFIED'
 m['f5']['technical_budget']='Owner ruling: no unit budget gate. Generous technical schema placeholders only; base/effective unit allowance 1/2, reserve0/0. Runaway wall timeout base15/effective30; forecast15 unchanged.'
 m['f5']['technical_guards']={'base_unit_cap':1.0,'effective_unit_cap':2.0,'base_reserve_units':0.0,'effective_reserve_units':0.0,'base_timeout_minutes':15,'effective_timeout_minutes':30,'units_are_scientific_budget_gate':False,'historical_aggregate_reservations_are_admission_gate':False}
 m['f5']['public_b8_transport']='Pinned release HTTPS fetch verified in public-b8-preflight.json; no local B8 substitution.'
 m['f5']['final_fit_inventory_gate']='PENDING_SEPARATE_ACCEPT_F5_FIT_FULL_INVENTORY_CANARY_AND_LOCAL_UPLOAD_RESOLUTION'
 m['timing_calibration']['scope']='Full twelve source-native timing/parity workloads plus timing-only optimizer/resume calibration; no allocation/job/completion/automatic retry.'
 m['resume_exemption']['reason']='Accepted full source timing/parity and timing-only optimizer/resume calibration; no scientific fit or outcome read; one attempt, no retry.'
 for row in m['inputs']:
  if row['dest'] in ['control.json','verified-ledger.json']:row['path']=str(P/row['dest']);row['sha256']=verify_export(P/row['dest'])
 from research.colab_cli.supervisor import input_preflight,validate
 before=now();public=input_preflight(m);assert len(public)==1 and public[0]['size_bytes']==6562005
 write_once(P/'public-b8-preflight.json',{'schema':'bl-f5-public-b8-https-preflight/v1','status':'PASS','started_utc':before,'finished_utc':now(),'verified_fetches':public,'public_release_provenance':bind(R/'research/t34_exit_b8/PUBLIC_INPUTS.json'),'fetch_implementation':bind(R/'research/colab_cli/inputs.py'),'transfer_policy':bind(R/'research/colab_cli/transfer_policy.py'),'payload_saved':False,'allocations':0})
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
 account=pathlib.Path('/Users/josenunez/.config/colab-cli/snake');active=[bind(x) for x in sorted(account.glob('active-*.json'))];pending=[bind(x) for x in sorted(account.glob('pending-*.json'))];assert not pending
 records={load(x['path'])['endpoint']:load(x['path']) for x in active};assert set(records)==set(last)
 rates={}
 for endpoint,row in last.items():
  rec=records[endpoint];assert rec['session']==row['session'];rate=rec.get('observed_rate_units_per_hour',rec.get('rate_units_per_hour'));assert isinstance(rate,(float,int)) and not isinstance(rate,bool) and rate>0
  expected={'L4':1.54,'CPU':0.08}.get(row['hardware'].upper());assert expected is None or rate==expected;assert row['hardware'].upper() in ['L4','CPU','A100'];rates[endpoint]=rate
 assert abs(sum(rates.values())-usage['rate'])<0.01
 # Bind only immutable accounting or terminal evidence from the separate plan operation.
 calibration=PLAN/'output-01/rate-calibration.json';check(bind(calibration));write(P/'plan-rate-calibration-snapshot.json',calibration.read_bytes());cal=load(calibration);assert cal['status']=='PASS'
 plan_endpoint=next(iter(cal['samples'][-1]['sessions']));plan_rate=cal['observed_rate_units_per_hour']
 plan_binding={'job_id':'bl-f5-w-plan-round08-c989f673-01','publication':bind(PLAN/'PACKAGE_PUBLICATION.json'),'rate_calibration':bind(calibration),'actual_endpoint':plan_endpoint,'actual_rate_units_per_hour':plan_rate,'read_only':True}
 terminal=PLAN/'output-01/receipt.json'
 if plan_endpoint in last:
  assert rates[plan_endpoint]==plan_rate;plan_binding['state']='ACTIVE_ACCOUNTED';assert last[plan_endpoint]['session']==cal['samples'][-1]['sessions'][plan_endpoint]
 else:
  check(bind(terminal));receipt=load(terminal);assert receipt['stop_verified'] is True;assert receipt['endpoint']==plan_endpoint;write(P/'plan-terminal-receipt-snapshot.json',terminal.read_bytes());plan_binding.update({'state':'ENDED_TERMINAL_RECEIPT_BOUND','terminal_receipt':bind(terminal),'terminal_status':receipt['status'],'endpoint_absent_in_both_fresh_sessions':True,'stop_verified':True,'receipt_snapshot':bind(P/'plan-terminal-receipt-snapshot.json')})
 write(P/'plan-operation-binding.json',plan_binding)
 write(P/'live-account-preflight.json',{'schema':'bl-f5-live-account-preflight/v5','status':'PASS','observed_utc':now(),'calls':calls,'sessions':last,'usage':usage,'other_assignments':rates,'plan_operation':bind(P/'plan-operation-binding.json'),'account_assignment_cap':3,'active_broker_records':active,'pending_broker_records':pending,'qualified_registry':bind(REG),'auth':'oauth2','allocation_calls':0,'registered_L4_rate_units_per_hour':1.54,'own_rate_freshly_measured':False,'own_rate_rule':'New assignment rate must be calibrated after allocation; priced ceiling2.0 is distinct.','changed_assignment_declaration':'Changed sealed other_assignments requires fresh published successor.'})
 m['budget']={'rate_units_per_hour':2.0,'unit_cap':2.0,'reserve_units':0.0,'other_assignments':rates}
 m['operational_admission']={'schema':'bl-f5-timing-operational-admission/v1','status':'ACCEPTED_FULL_TIMING_PARITY_AND_CALIBRATION_ONLY','scientific_fits_authorized':False,'numerical_looks_authorized':False,'release_authorized':False,'rulings':{k:bind(P/v) for k,v in RULINGS.items()},'bindings':bind(P/'bindings.json'),'historical_accounting':bind(P/'historical-accounting.json'),'live_account_preflight':bind(P/'live-account-preflight.json'),'public_b8_preflight':bind(P/'public-b8-preflight.json'),'accounting_declaration_must_match_seal':True}
 inventory=inv(m);update_counts(m,inventory);write(P/'inventory-metadata.json',inventory);write(P/'manifest.json',m);validate(m,R)
 from research.colab_cli.source import prepare
 plan=prepare(R,m,P/'source-preflight-artifacts')
 for relative,h in c['source_hashes'].items():assert plan['files'][relative]['sha256']==h
 for relative,x in plan['files'].items():
  raw=git('show',f'{PIN}:{relative}');assert sha(raw)==x['sha256'] and len(raw)==x['size'];assert (R/relative).read_bytes()==raw
 with tarfile.open(P/'source-preflight-artifacts/source.tar.gz') as archive:
  members=archive.getmembers();assert sorted(x.name for x in members)==sorted(plan['files'])
  for x in members:assert sha(archive.extractfile(x).read())==plan['files'][x.name]['sha256']
 for p in sorted((P/'source-preflight-artifacts').iterdir()):
  if p.name.endswith(('.sha256','.complete')):continue
  for suffix,data in [('.sha256',(bind(p)['sha256']+'  '+p.name+'\n').encode()),('.complete',b'complete\n')]:
   x=pathlib.Path(str(p)+suffix)
   if not x.exists():
    with x.open('xb') as stream:stream.write(data)
 write(P/'source-preflight.json',{'schema':'bl-f5-source-preflight/v5','status':'PASS','created_utc':now(),'manifest':bind(P/'manifest.json'),'implementation_commit':PIN,'source_pin_count':len(c['source_hashes']),'source_paths_count':len(m['source_paths']),'archive_members':len(members),'source_plan':plan,'all_git_and_working_source_pins_match':True,'allocation_calls':0,'full_qualification_entrypoint':True,'boundary_checks_retained':True,'no_scientific_payload_in_source_archive':True})
 print(json.dumps({'package':str(P),'private_bytes':inventory['private_upload_bytes'],'account_usage':usage,'other_assignments':rates,'plan_binding':plan_binding},indent=2))
if phase=='validate':
 fresh();assert not (P/'broker-validation.json').exists()
 cmds=[run(['python3',R/'tools/ops/qualification.py','check','--repo',R,'--record',REG,'--ref',v]) for v in ['HEAD',PIN]]
 cmds.append(run([R/'research/colab_cli/launch.sh','--state-dir',P/'state-01','--colab','/Users/josenunez/.local/bin/colab','validate',P/'manifest.json']))
 write(P/'broker-validation.json',{'status':'PASS','created_utc':now(),'commands':cmds,'registry':bind(REG),'qualified_acceptance_result':bind(load(REG)['acceptance_result']),'broker_wrapper':bind(R/'tools/ops/dispatch.py'),'manifest':bind(P/'manifest.json'),'allocations':0})
 text=f'''ADMITTED: single future operator full timing/parity/calibration dispatch after sealed package publication to main.
Scope: F5_MODE=qualification. Run all twelve source-native timing/parity workloads and timing-only optimizer/resume calibration; no transport-only substitute. Report explicit pending-Mac status. No actual common plan is consumed by this job.
Scientific admission remains false. Scientific fits, numerical outcome looks, serving and release remain unauthorized. Calibration artifacts cannot be used as scientific fits.
Job: {JOB}
Implementation: {PIN}
Manifest: {P/'manifest.json'} SHA256 {bind(P/'manifest.json')['sha256']}
State: {P/'state-01'}
Output: {P/'output-01'}
Seal: {P/'PACKAGE_SEAL_OPERATIONAL.json'}; verify SHA sidecar and main publication before use.
One separate L4 >=6vCPU; cuda:0; CPython3.12.3; exact69-wheel/cu126 lock; --no-compile, deterministic fp32, numeric threads1. Forecast15 minutes; base15/effective30-minute timeout for manifest and short-canary exemption.
Owner ruling forbids budget-only refusal: units and aggregate reservations are not budget gates. Schema unit placeholders base1/effective2, reserve0/0. Historical accounting retained.
Nine exact inputs: five pinned opponent checkpoint uploads plus fixtures/control/empty matching ledger and release HTTPS B8 (6562005 bytes). Seven required exports: qualification/qualification.json plus the six paired producer paths. Private upload cap45397643 bytes; transfer<=150s; staging<=300s; export<=2GiB.
No allocation/job/completion/automatic retry. Preserve mechanical failure diagnostics.
Plan operation reference: {P/'plan-operation-binding.json'}; live account/rates: {P/'live-account-preflight.json'}.
Recheck unused name/output/state immediately before dispatch. Refresh read-only account/assignment/rate accounting under serialized supervisor allocation; maximum3 total account assignments. Changed sealed endpoint/rates require a fresh published successor; preserve this package.
After execution Mac-verify immutable exports, actual v5 cross-assignment environment and before/after boundaries, terminal success and independently saved release. Join only both verified branches with Mac timing/calibration, honest forecasts, actual common plan and initial paired ledger, sole prospective stage claim/look token, complete final science inventory/current-source canary, local-upload final-fit resolution and separate ACCEPT_F5_FIT.
Exact operator command:
{shlex.join(command())}
'''
 write(P/'DISPATCH.txt',text)
 dr=run(command()+['--dry-run']);fresh();assert dr['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 write(P/'dispatch-dry-run.json',{'status':'PASS','created_utc':now(),'command':dr,'wrapper':bind(R/'tools/ops/dispatch.py'),'manifest':bind(P/'manifest.json'),'dispatch_instructions':bind(P/'DISPATCH.txt'),'separate_live_account_preflight':bind(P/'live-account-preflight.json'),'allocations':0,'state_absent':True,'output_absent':True})
 print(dr['stdout'])
