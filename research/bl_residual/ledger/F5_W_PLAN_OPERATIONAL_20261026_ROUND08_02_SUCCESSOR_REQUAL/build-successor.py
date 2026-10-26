"""Create-only round08 plan package. Never allocates or dispatches a runtime."""
from __future__ import annotations
import copy, datetime, gzip, hashlib, json, os, pathlib, re, shlex, subprocess, sys, tarfile, time
R=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5-plan02-requal')
B=R.parent/'snake-dqn-artifacts/big-lever/exp/astra-20261024'
D=B/'F5_W_NEXT_AFTER_ATTEMPT04_DRAFT_20261026_v2'
P=B/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02_SUCCESSOR_REQUAL'
OLD=B/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_01'
TIMING=B/'F5_W_TIMING_OPERATIONAL_20261026_ROUND08_01'
DISPOSITION='F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026.md'
T=B/'F5_W_TIMING_DRAFT_20261026_ROUND08_v3'
C=B/'F5_W_CAPTURE_AFTER_TRANSPORT_REQUAL_05'
F=B/'F5_W_PREFIT_20261025/final-09'
PIN='75b1442768f37ce83188a2990c9ef0026ee806b5'
JOB='bl-f5-w-plan-r08-sr75b14427-01'
PYTHON=R.parent/'snake-dqn/venv/bin/python'
REG=R.parent/'snake-dqn-artifacts/ops/colab-cli-qualified.json'
RULINGS={'acceptance':'ACCEPT_F5_W_NEXT_PACKAGES_20261026_ROUND08.md','capture_qualification':'F5_W_CAPTURE_TRANSPORT_QUALIFIED_20261026.md','parallel_clarification':'F5_W_ROUND08_PARALLEL_SEALING_CLARIFICATION_20261026.md','owner_ruling':'OWNER_RULING_COLAB_BUDGET_20261025.txt','transport_repair':'ACCEPT_TRANSPORT_REPAIR_7f5357a7.md'}
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
phase=sys.argv[1]
if phase=='prepare':
 assert all(x.name in {'build-successor.py','seal-successor.py'} for x in P.iterdir())
 predecessor={str(x):bind(x) for x in sorted(OLD.rglob('*')) if x.is_file()}
 write(P/'predecessor-inventory.json',predecessor)
 for name in [*RULINGS.values(),DISPOSITION]:write(P/name,(B/name).read_bytes())
 write(P/'capture05-post-publication-verification.json',(C/'POST_CAPTURE_VERIFICATION_05.json').read_bytes())
 write(P/'capture05-latency-summary.json',(C/'POST_CAPTURE_LATENCY_SUMMARY_05.json').read_bytes())
 write(P/'capture05-environment.json',(C/'output-05/exports/environment.json').read_bytes())
 write(P/'build-package.py',pathlib.Path(__file__).read_bytes())
 write(P/'attempt01-terminal-receipt.json',(OLD/'output-01/receipt.json').read_bytes())
 write(P/'attempt01-accounting-baseline.json',(OLD/'output-01/accounting-baseline.json').read_bytes())
 write(P/'attempt01-rate-calibration.json',(OLD/'output-01/rate-calibration.json').read_bytes())
 pub=R/'research/bl_residual/ledger'/OLD.name/'OUTPUT_01_PUBLICATION.json'
 write(P/'attempt01-output-publication.json',pub.read_bytes())
 previous=load(OLD/'historical-accounting.json');receipt=load(P/'attempt01-terminal-receipt.json')
 assert receipt['status']=='INCOMPLETE' and receipt['stop_verified'] is True and receipt['exports']==[]
 assert receipt['usage_before']['balance']==1066.28 and receipt['usage_after']['balance']==1066.25
 hist={**previous,'schema':'bl-f5-cumulative-accounting-plan02/v1','prior_history':bind(OLD/'historical-accounting.json'),'prior_history_verbatim':previous,'attempt01_receipt':bind(P/'attempt01-terminal-receipt.json'),'attempt01_baseline':bind(P/'attempt01-accounting-baseline.json'),'attempt01_calibration':bind(P/'attempt01-rate-calibration.json'),'attempt01_output_publication':bind(P/'attempt01-output-publication.json'),'attempt01_publication_commit':git('rev-parse','HEAD').decode().strip(),'attempt01_elapsed_seconds':receipt['elapsed_seconds'],'attempt01_observed_balance_before':1066.28,'attempt01_observed_balance_after':1066.25,'attempt01_observed_balance_decrease_units':0.03,'observed_debit_attribution':'Observed balance decrease only; not a precision claim about workload attribution.','capture_and_plan01_cumulative_elapsed_seconds':previous['cumulative_elapsed_seconds']+receipt['elapsed_seconds'],'capture_and_plan01_cumulative_observed_balance_decreases_units':previous['attempts_01_to_03_observed_attributable_units']+previous['attempt04_observed_balance_debit_units']+0.03,'cumulative_envelope':bind(C/'cumulative-envelope-owner-ruling-04.json'),'cumulative_accounting_not_reset':True,'attempt02_elapsed_seconds':0,'attempt02_debit_units':0,'attempt02_not_dispatched':True,'overlapping_timing_accounting':'Separate live runtime rate bound by fresh account preflight. Timing debit is never erased or attributed to this plan.'}
 write(P/'historical-accounting.json',hist)
 control=copy.deepcopy(load(OLD/'control.json'));control['operative_package']=str(P)
 from research.bl_residual.f5_package import source_inventory
 control['source_hashes']=source_inventory(R)
 control['identity']['source_sha256']=sha(json.dumps(control['source_hashes'],sort_keys=True).encode())
 control['environment_capture']=load(P/'capture05-environment.json')
 control['attempt05_environment_reference']=bind(P/'capture05-environment.json')
 control['operative_authority']={k:bind(P/v) for k,v in RULINGS.items()}
 control['attempt02_authority']={'disposition':bind(P/DISPOSITION),'ended_attempt01':bind(P/'attempt01-terminal-receipt.json'),'historical_accounting':bind(P/'historical-accounting.json'),'manual_attempt_maximum':1,'automatic_retries':False}
 control['allocation']=sha(json.dumps({k:v for k,v in control.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 ledger={'schema':'bl-f5-verified-ledger/v1','namespace':control['namespace'],'allocation_sha256':control['identity']['allocation_sha256'],'allocation':control['allocation'],'status':'EMPTY_VERIFIED_LEDGER_NO_COMPLETED_UNITS','units':[]}
 write(P/'verified-ledger.json',ledger);control['ledger_sha256']=verify_export(P/'verified-ledger.json');write(P/'control.json',control)
 bindings=load(OLD/'bindings.json');bindings['rulings']={k:bind(P/v) for k,v in RULINGS.items()};bindings['attempt02_disposition']=bind(P/DISPOSITION);bindings['attempt01_terminal']=bind(P/'attempt01-terminal-receipt.json');bindings['attempt01_seal']=bind(OLD/'PACKAGE_SEAL_OPERATIONAL.json');bindings['attempt01_package_publication']=bind(OLD/'PACKAGE_PUBLICATION.json');bindings['attempt01_output_publication']=bind(P/'attempt01-output-publication.json');bindings['historical_accounting']=bind(P/'historical-accounting.json');bindings['capture05_postpublication_verification']=bind(P/'capture05-post-publication-verification.json');bindings['capture05_latency_summary']=bind(P/'capture05-latency-summary.json');bindings['capture05_environment']=bind(P/'capture05-environment.json')
 write(P/'bindings.json',bindings)
 from research.bl_residual.f5_package import governance,source_review
 write(P/'governance-source-preflight.json',{'schema':'bl-f5-governance-source-preflight/v1','status':'PASS','created_utc':now(),'governance':governance(R,B),'source_semantic_review':source_review(R),'no_protected_payload_read':True,'no_scientific_outcome_read':True,'allocations':0})
 print('PREPARED_CREATE_ONLY',P)
if phase=='preflight':
 fresh()
 from research.bl_residual.f5_environment import validate_environment, compare_environment
 from research.bl_residual.f5_job import _control,_source_admission
 control=_control(P);_source_admission(control);actual=load(P/'capture05-environment.json');assert control['environment_capture']==actual
 validate_environment(actual);comparison=compare_environment(actual,control['environment_capture'],mode='cross-assignment')
 write(P/'environment-reference-preflight.json',{'status':'PASS','created_utc':now(),'control':bind(P/'control.json'),'actual_attempt05_environment':bind(P/'capture05-environment.json'),'postpublication_verification':bind(P/'capture05-post-publication-verification.json'),'exact_raw_object_equality':True,'comparison':comparison,'boundary_implementation':bind(R/'research/bl_residual/f5_job.py'),'new_runtime_environment':'Must be captured and cross-assignment validated after future operator dispatch; not observed here.'})
 m=load(OLD/'manifest.json');m['job_id']=JOB;m['commit']=PIN;m['output_dir']=str(P/'output-01');m['supervisor_state_dir']=str(P/'state-01');m['lease_seconds']=180
 m['inventory_status']='FINAL_COMPLETE_METADATA_RECONCILED_HTTPS_B8_VERIFIED'
 m['f5']['technical_budget']='Finite generous technical guards only; owner ruling forbids budget-only refusal. Base cap/reserve1/.5 doubled2/1, base timeout15 doubled30; forecast15 unchanged.'
 m['f5']['technical_guards']={'base_unit_cap':1.0,'effective_unit_cap':2.0,'base_reserve_units':0.5,'effective_reserve_units':1.0,'base_timeout_minutes':15,'effective_timeout_minutes':30,'units_are_scientific_budget_gate':False,'historical_aggregate_reservations_are_admission_gate':False}
 m['f5']['public_b8_transport']='Pinned release HTTPS fetch verified in public-b8-preflight.json; no local B8 substitution.'
 m['f5']['final_fit_inventory_gate']='PENDING_SEPARATE_ACCEPT_F5_FIT_FULL_INVENTORY_CANARY_AND_LOCAL_UPLOAD_RESOLUTION'
 m['timing_calibration']['scope']='Single accepted plan materialization and disjoint CUDA transport; no allocation/job/completion/automatic retry.'
 m['resume_exemption']['reason']='Astra F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026 explicitly permits exactly one fresh manually dispatched plan attempt02 after mechanical attempt01 INCOMPLETE; identical materialization/disjoint transport scope only; no scientific fit, outcome look, allocation/job/completion/automatic retry.'
 for row in m['inputs']:
  if row['dest'] in ['control.json','verified-ledger.json']:row['path']=str(P/row['dest']);row['sha256']=verify_export(P/row['dest'])
 from research.colab_cli.supervisor import input_preflight,validate
 before=now();public=input_preflight(m)
 assert len(public)==1 and public[0]['size_bytes']==6562005
 write(P/'public-b8-preflight.json',{'schema':'bl-f5-public-b8-https-preflight/v1','status':'PASS','started_utc':before,'finished_utc':now(),'verified_fetches':public,'public_release_provenance':bind(R/'research/t34_exit_b8/PUBLIC_INPUTS.json'),'fetch_implementation':bind(R/'research/colab_cli/inputs.py'),'transfer_policy':bind(R/'research/colab_cli/transfer_policy.py'),'payload_saved':False,'allocations':0})
 calls=load(C/'live-account-preflight.json')['calls']
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
  rec=records[endpoint];assert row['session'] in [rec['session'],'?'];assert rec['session'].startswith('snake-bl-f5-w-timing-round08-c989f673-01-');rate=rec.get('observed_rate_units_per_hour',rec.get('rate_units_per_hour'));assert isinstance(rate,(float,int)) and not isinstance(rate,bool) and rate>0
  expected={'L4':1.54,'CPU':0.08}.get(row['hardware'].upper());assert expected is None or rate==expected;assert row['hardware'].upper() in ['L4','CPU','A100'];rates[endpoint]=rate
 assert abs(sum(rates.values())-usage['rate'])<0.01
 assert not last and not records and usage['assignments']==0.0 and usage['rate']==0.0
 write(P/'saved-account-snapshot.json',{'schema':'bl-f5-saved-account-snapshot/v1','source':bind(C/'live-account-preflight.json'),'saved_observed_utc':load(C/'live-account-preflight.json')['created_utc'],'sessions':last,'usage':usage,'other_assignments':rates,'active_broker_records':active,'pending_broker_records':pending,'current_availability_claimed':False,'label':'SAVED_HISTORICAL_NOT_LIVE; dispatch wrapper refresh owns current accounting.','allocation_calls':0})
 m['budget']={'rate_units_per_hour':2.0,'unit_cap':2.0,'reserve_units':1.0,'other_assignments':rates}
 m['operational_admission']={'schema':'bl-f5-plan-operational-admission/v2','status':'ACCEPTED_PLAN_MATERIALIZATION_AND_TRANSPORT_ONLY','scientific_fits_authorized':False,'numerical_looks_authorized':False,'release_authorized':False,'rulings':{k:bind(P/v) for k,v in RULINGS.items() if k!='transport_repair'},'bindings':bind(P/'bindings.json'),'historical_accounting':bind(P/'historical-accounting.json'),'saved_account_snapshot':bind(P/'saved-account-snapshot.json'),'public_b8_preflight':bind(P/'public-b8-preflight.json'),'transport_repair_ruling':bind(P/'ACCEPT_TRANSPORT_REPAIR_7f5357a7.md'),'capture05_postpublication_verification':bind(P/'capture05-post-publication-verification.json'),'accounting_declaration_must_match_seal':True,'attempt02_disposition':bind(P/DISPOSITION),'attempt01_terminal':bind(P/'attempt01-terminal-receipt.json'),'fresh_external_review_required_before_manual_dispatch':True}
 inventory=inv(m);update_counts(m,inventory);write(P/'inventory-metadata.json',inventory);write(P/'manifest.json',m)
 validate(m,R)
 from research.colab_cli.source import prepare
 plan=prepare(R,m,P/'source-preflight-artifacts')
 for x in sorted((P/'source-preflight-artifacts').iterdir()):
  export_bytes(P/'source-preflight-companions'/x.name,x.read_bytes())
 for relative,h in control['source_hashes'].items():assert plan['files'][relative]['sha256']==h
 for relative,x in plan['files'].items():
  raw=git('show',f'{PIN}:{relative}');assert sha(raw)==x['sha256'] and len(raw)==x['size'];assert (R/relative).read_bytes()==raw
 with tarfile.open(P/'source-preflight-artifacts/source.tar.gz') as archive:
  members=archive.getmembers();assert sorted(x.name for x in members)==sorted(plan['files'])
  for x in members:assert sha(archive.extractfile(x).read())==plan['files'][x.name]['sha256']
 write(P/'source-preflight.json',{'schema':'bl-f5-source-preflight/v4','status':'PASS','created_utc':now(),'manifest':bind(P/'manifest.json'),'implementation_commit':PIN,'source_pin_count':len(control['source_hashes']),'source_paths_count':len(m['source_paths']),'archive_members':len(members),'source_plan':plan,'all_git_and_working_source_pins_match':True,'allocation_calls':0,'boundary_checks_retained':True,'no_scientific_payload_in_source_archive':True})
 cmds=[run(['python3',R/'tools/ops/qualification.py','check','--repo',R,'--record',REG,'--ref',v]) for v in ['HEAD',PIN]]
 # launch.sh validate is read-only; use a distinct temporary state argument so target stays unused.
 cmds.append(run([R/'research/colab_cli/launch.sh','--state-dir',P/'state-01','--colab','/Users/josenunez/.local/bin/colab','validate',P/'manifest.json']))
 write(P/'broker-validation.json',{'status':'PASS','created_utc':now(),'commands':cmds,'registry':bind(REG),'qualified_acceptance_result':bind(load(REG)['acceptance_result']),'broker_wrapper':bind(R/'tools/ops/dispatch.py'),'manifest':bind(P/'manifest.json'),'allocations':0})
 text=f'''ADMITTED: single future operator plan-materialization dispatch after sealed package publication to main.
Authority: F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026.md; attempt01 INCOMPLETE terminal and all historical debit/elapsed retained without reset.
Scope: F5_MODE=prepare-qualification; ordered L4-materialized2000x256 common plan (128 doom then128 ordinary), seed3403, exact row ordering and initial/final CPU-generator/global RNG boundaries; disjoint genuine CUDA transport only.
Scientific admission remains false; scientific fitting, numerical outcome reads, serving and release remain unauthorized. Paired exports are transport/calibration artifacts unusable as scientific fits.
Job: {JOB}
Implementation: {PIN}
Manifest: {P/'manifest.json'} SHA256 {bind(P/'manifest.json')['sha256']}
State: {P/'state-01'}
Output: {P/'output-01'}
Seal: {P/'PACKAGE_SEAL_OPERATIONAL.json'}; verify SHA sidecar and main publication before use.
One L4 >=6vCPU; cuda:0; Python3.12.3; exact69-wheel/cu126 lock, --no-compile, deterministic fp32, numerical threads1; timeout15base/30effective; estimate15 forecast unchanged.
Technical cap1base/2effective and reserve.5base/1effective; time/unit/aggregate reservations are runaway guards only per owner ruling. No budget-only refusal authority. Historical accounting retained without reset.
All nine required exports and all seven exact inputs are fixed by manifest; B8 uses release HTTPS (6562005 bytes), not private upload. Inclusive transfer<=150s, staging<=300s, private bytes<=45397643, export<=2GiB.
Exactly one fresh manual attempt02 is authorized by F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026.md. No allocation/job/completion/automatic retry. If failure repeats, preserve both receipts and return for transport diagnosis/repair/requalification before any further plan attempt.
Recheck state/output/name unused immediately before dispatch. Refresh account evidence under serialized supervisor allocation, <=3 account-wide assignments; changed endpoint/rate declarations require fresh successor sealing/publication. The sealed timing endpoint/rate is an actual current account observation, not a future availability claim.
After operation verify immutable exports, actual v5 cross-assignment environment, plan512000-row receipt and actual plan/RNG digests, terminal success and saved independent release before any join.
Timing job bl-f5-w-timing-round08-c989f673-01 is running on its separate L4. Its actual live endpoint/rate is bound in timing-operation-binding.json and live-account-preflight.json. Do not mutate timing state/output or co-locate workloads.
Before science require both verified branches, Mac timing/calibration and honest forecasts, actual plan, paired ledger, sole prospective stage claim/look token, complete final science inventory/current-source canary, local-upload final-fit resolution and separate ACCEPT_F5_FIT.
Exact operator command:
{shlex.join(command())}
'''
 write(P/'DISPATCH.txt',text)
 dr=run(command()+['--dry-run']);fresh();assert dr['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 write(P/'dispatch-dry-run.json',{'status':'PASS','created_utc':now(),'command':dr,'manifest':bind(P/'manifest.json'),'dispatch_instructions':bind(P/'DISPATCH.txt'),'separate_live_account_preflight':bind(P/'live-account-preflight.json'),'allocations':0,'state_absent':True,'output_absent':True})
 print(json.dumps({'package':str(P),'manifest':bind(P/'manifest.json'),'private_bytes':inventory['private_upload_bytes'],'source_pins':len(control['source_hashes']),'archive_members':len(members),'account_usage':usage,'dry_run_output':dr['stdout'].strip()},indent=2))
