"""Create-only round08 plan package. Never allocates or dispatches a runtime."""
from __future__ import annotations
import copy, datetime, gzip, hashlib, json, os, pathlib, re, shlex, subprocess, sys, tarfile, time
R=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5w')
B=R.parent/'snake-dqn-artifacts/big-lever/exp/astra-20261024'
D=B/'F5_W_NEXT_AFTER_ATTEMPT04_DRAFT_20261026_v2'
P=B/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_01'
T=B/'F5_W_TIMING_DRAFT_20261026_ROUND08_v3'
C=B/'F5_W_CAPTURE_AFTER_22e86986_REPAIR_04'
F=B/'F5_W_PREFIT_20261025/final-09'
PIN='c989f6738f6aad143eb97bc47f6bd9e6687057ce'
JOB='bl-f5-w-plan-round08-c989f673-01'
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
phase=sys.argv[1]
if phase=='prepare':
 assert not P.exists() and not T.exists();P.mkdir();T.mkdir();fresh()
 predecessor={str(x):bind(x) for x in sorted(D.rglob('*')) if x.is_file()}
 write(P/'predecessor-inventory.json',predecessor)
 for name in RULINGS.values():write(P/name,(B/name).read_bytes())
 write(P/'build-package.py',pathlib.Path(__file__).read_bytes())
 old=load(D/'plan-seal/control.json');control=copy.deepcopy(old)
 control.pop('draft_name');control['status']='ACCEPTED_PLAN_MATERIALIZATION_AND_TRANSPORT_ONLY_NO_SCIENTIFIC_FIT'
 control['operative_package']=str(P)
 control['operative_authority']={k:bind(P/v) for k,v in RULINGS.items()}
 control['scientific_admission']=False
 control['allocation']=sha(json.dumps({k:v for k,v in control.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 ledger={'schema':'bl-f5-verified-ledger/v1','namespace':control['namespace'],'allocation_sha256':control['identity']['allocation_sha256'],'allocation':control['allocation'],'status':'EMPTY_VERIFIED_LEDGER_NO_COMPLETED_UNITS','units':[]}
 write(P/'verified-ledger.json',ledger);control['ledger_sha256']=verify_export(P/'verified-ledger.json');write(P/'control.json',control)
 write(P/'historical-accounting.json',load(D/'historical-accounting.json'))
 bindings=load(D/'draft-bindings.json');bindings['schema']='bl-f5-plan-operative-bindings/v1';bindings['rulings']={k:bind(P/v) for k,v in RULINGS.items()}
 bindings['predecessor_seal']=bind(D/'PACKAGE_SEAL.json');bindings['source_commit']=PIN
 bindings['accepted_source_packet']=bind(F/'REVIEW_PACKET.json');bindings['accepted_source_bundle']=bind(F/'REVIEW_BUNDLE.json')
 bindings['capture_bindings']=bind(C/'bindings-v5.json');bindings['prior_accounting']=bind(C/'accounting-owner-ruling-04.json');bindings['prior_cumulative_envelope']=bind(C/'cumulative-envelope-owner-ruling-04.json')
 bindings['capture_terminal']=bind(C/'output-04/receipt.json');bindings['capture_publication_commit']='969efe7cf782af049612055cb8f7150d23c7ee52'
 bindings['fixture_descriptor']=bind(F/'timing-fixtures.json');bindings['frozen_allocation']=bind(F/'allocation.json')
 for name in ['requirements-f5-l4-cu126.lock','f5-common-environment-v6.json']:
  bindings[name]=bind(R/'research/bl_residual/f5_adoption_environment_20261026_v6'/name)
 bindings['frozen_v5_schema']=bind(R/'research/bl_residual/f5_environment_schema_v5.json')
 bindings['transfer_policy']=bind(R/'research/colab_cli/transfer_policy.py');bindings['public_release_provenance']=bind(R/'research/t34_exit_b8/PUBLIC_INPUTS.json')
 write(P/'bindings.json',bindings)
 # Timing bookkeeping successor only; no operative seal, preflight or dispatch instruction.
 for name in ['control.json','verified-ledger.json']:write(T/name,(D/'timing-qualification'/name).read_bytes())
 tm=load(D/'timing-qualification/manifest.json');tm['job_id']='bl-f5-w-timing-round08-draft-v3';tm['output_dir']=str(T/'output-pending');tm['supervisor_state_dir']=str(T/'state-pending')
 for row in tm['inputs']:
  if row['dest'] in ['control.json','verified-ledger.json']:row['path']=str(T/row['dest']);row['sha256']=verify_export(T/row['dest'])
 ti=inv(tm);update_counts(tm,ti);tm['draft_only']['status']='DRAFT_ONLY_NOT_DISPATCHABLE_AWAIT_PLAN_LIVE_ENDPOINT_RATE';tm['draft_only']['parallel_clarification']=bind(P/RULINGS['parallel_clarification'])
 write(T/'inventory-metadata.json',ti);write(T/'manifest.json',tm)
 write(T/'SUCCESSOR_NOTE.json',{'schema':'bl-f5-timing-draft-bookkeeping-successor/v1','status':'DRAFT_ONLY_UNSEALED_NOT_DISPATCHABLE','predecessor_manifest':bind(D/'timing-qualification/manifest.json'),'correction':'Regenerated top-level known_private_input_bytes from final successor inputs; nested count and inventory reconciled.','old_top_level_bytes':30722179,'correct_top_level_bytes':ti['private_upload_bytes'],'required_next_step':'After plan dispatch bind live endpoint/rate and fresh account evidence, finalize successor controls/allocation/ledger and inventories, validate, seal and publish fresh operative timing successor. If plan already released bind actual then-current state.','parallel_clarification':bind(P/RULINGS['parallel_clarification']),'timing_sealed':False,'dispatches':0})
 print('PREPARED',P,'TIMING_DRAFT',T,'control',bind(P/'control.json'),'timing_bytes',ti['private_upload_bytes'])
if phase=='preflight':
 fresh()
 from research.bl_residual.f5_environment import validate_environment, compare_environment
 from research.bl_residual.f5_job import _control,_source_admission
 control=_control(P);_source_admission(control);actual=load(C/'output-04/exports/environment.json');assert control['environment_capture']==actual
 validate_environment(actual);comparison=compare_environment(actual,control['environment_capture'],mode='cross-assignment')
 write(P/'environment-reference-preflight.json',{'status':'PASS','created_utc':now(),'control':bind(P/'control.json'),'actual_attempt04_environment':bind(C/'output-04/exports/environment.json'),'exact_raw_object_equality':True,'comparison':comparison,'boundary_implementation':bind(R/'research/bl_residual/f5_job.py'),'new_runtime_environment':'Must be captured and cross-assignment validated after future operator dispatch; not observed here.'})
 m=load(D/'plan-seal/manifest.json');m.pop('draft_only');m['job_id']=JOB;m['output_dir']=str(P/'output-01');m['supervisor_state_dir']=str(P/'state-01');m['lease_seconds']=180
 m['inventory_status']='FINAL_COMPLETE_METADATA_RECONCILED_HTTPS_B8_VERIFIED'
 m['f5']['technical_budget']='Finite generous technical guards only; owner ruling forbids budget-only refusal. Base cap/reserve1/.5 doubled2/1, base timeout15 doubled30; forecast15 unchanged.'
 m['f5']['technical_guards']={'base_unit_cap':1.0,'effective_unit_cap':2.0,'base_reserve_units':0.5,'effective_reserve_units':1.0,'base_timeout_minutes':15,'effective_timeout_minutes':30,'units_are_scientific_budget_gate':False,'historical_aggregate_reservations_are_admission_gate':False}
 m['f5']['public_b8_transport']='Pinned release HTTPS fetch verified in public-b8-preflight.json; no local B8 substitution.'
 m['f5']['final_fit_inventory_gate']='PENDING_SEPARATE_ACCEPT_F5_FIT_FULL_INVENTORY_CANARY_AND_LOCAL_UPLOAD_RESOLUTION'
 m['timing_calibration']['scope']='Single accepted plan materialization and disjoint CUDA transport; no allocation/job/completion/automatic retry.'
 m['resume_exemption']['reason']='Accepted plan materialization and disjoint transport only; no scientific fit or outcome read; one attempt, no retry.'
 for row in m['inputs']:
  if row['dest'] in ['control.json','verified-ledger.json']:row['path']=str(P/row['dest']);row['sha256']=verify_export(P/row['dest'])
 from research.colab_cli.supervisor import input_preflight,validate
 before=now();public=input_preflight(m)
 assert len(public)==1 and public[0]['size_bytes']==6562005
 write(P/'public-b8-preflight.json',{'schema':'bl-f5-public-b8-https-preflight/v1','status':'PASS','started_utc':before,'finished_utc':now(),'verified_fetches':public,'public_release_provenance':bind(R/'research/t34_exit_b8/PUBLIC_INPUTS.json'),'fetch_implementation':bind(R/'research/colab_cli/inputs.py'),'transfer_policy':bind(R/'research/colab_cli/transfer_policy.py'),'payload_saved':False,'allocations':0})
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
 write(P/'live-account-preflight.json',{'schema':'bl-f5-live-account-preflight/v4','status':'PASS','observed_utc':now(),'calls':calls,'sessions':last,'usage':usage,'other_assignments':rates,'account_assignment_cap':3,'active_broker_records':active,'pending_broker_records':pending,'qualified_registry':bind(REG),'auth':'oauth2','allocation_calls':0,'registered_L4_rate_units_per_hour':1.54,'own_rate_freshly_measured':False,'own_rate_rule':'Actual new assignment rate must be calibrated after allocation; historical1.54 and priced ceiling2.0 remain distinct.','changed_assignment_declaration':'Wrapper must refuse changed sealed other_assignments and require a fresh published successor.'})
 m['budget']={'rate_units_per_hour':2.0,'unit_cap':2.0,'reserve_units':1.0,'other_assignments':rates}
 m['operational_admission']={'schema':'bl-f5-plan-operational-admission/v1','status':'ACCEPTED_PLAN_MATERIALIZATION_AND_TRANSPORT_ONLY','scientific_fits_authorized':False,'numerical_looks_authorized':False,'release_authorized':False,**{k:bind(P/v) for k,v in RULINGS.items()},'bindings':bind(P/'bindings.json'),'historical_accounting':bind(P/'historical-accounting.json'),'live_account_preflight':bind(P/'live-account-preflight.json'),'public_b8_preflight':bind(P/'public-b8-preflight.json'),'accounting_declaration_must_match_seal':True}
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
No allocation/job/completion/automatic retry. Preserve mechanical failure diagnostics.
Recheck state/output/name unused immediately before dispatch. Refresh account evidence under serialized supervisor allocation, <=3 account-wide assignments; changed endpoint/rate declarations require fresh successor sealing/publication. Historical zero assignments is not future availability.
After operation verify immutable exports, actual v5 cross-assignment environment, plan512000-row receipt and actual plan/RNG digests, terminal success and saved independent release before any join.
Timing draft remains unsealed. After plan dispatch bind live endpoint/rate with fresh account evidence in a separate operative timing successor before sealing/publication.
Before science require both verified branches, Mac timing/calibration and honest forecasts, actual plan, paired ledger, sole prospective stage claim/look token, complete final science inventory/current-source canary, local-upload final-fit resolution and separate ACCEPT_F5_FIT.
Exact operator command:
{shlex.join(command())}
'''
 write(P/'DISPATCH.txt',text)
 dr=run(command()+['--dry-run']);fresh();assert dr['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 write(P/'dispatch-dry-run.json',{'status':'PASS','created_utc':now(),'command':dr,'manifest':bind(P/'manifest.json'),'dispatch_instructions':bind(P/'DISPATCH.txt'),'separate_live_account_preflight':bind(P/'live-account-preflight.json'),'allocations':0,'state_absent':True,'output_absent':True})
 print(json.dumps({'package':str(P),'manifest':bind(P/'manifest.json'),'private_bytes':inventory['private_upload_bytes'],'source_pins':len(control['source_hashes']),'archive_members':len(members),'account_usage':usage,'dry_run_output':dr['stdout'].strip()},indent=2))
