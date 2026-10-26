"""Create-only named successor preparation; only offline validation/dry-run, never dispatch."""
import pathlib,json,hashlib,subprocess,datetime,copy,sys,os,tarfile,gzip,shlex
R=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5-plan02-requal')
W=R.parent/'snake-dqn-f5w';A=R.parent/'snake-dqn-artifacts/big-lever/exp/astra-20261024'
P=A/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02_SUCCESSOR_REQUAL';H=A/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02'
C=A/'F5_W_CAPTURE_AFTER_TRANSPORT_REQUAL_05';F=A/'F5_W_PREFIT_20261025/final-09'
PIN='75b1442768f37ce83188a2990c9ef0026ee806b5';BROKER='97f066ce106000ae0264c27f88da38e1b894a860'
IMPL='c989f6738f6aad143eb97bc47f6bd9e6687057ce';TREE='6e8c1982b5c9ed115bfebd3145ad0720b15522fa'
JOB='bl-f5-w-plan-r08-sr75b14427-01';REG=R.parent/'snake-dqn-artifacts/ops/colab-cli-qualified.json'
PY=R.parent/'snake-dqn/venv/bin/python';OPS=REG.parent
NAMES={'acceptance':'ACCEPT_F5_W_NEXT_PACKAGES_20261026_ROUND08.md','capture_qualification':'F5_W_CAPTURE_TRANSPORT_QUALIFIED_20261026.md','parallel_clarification':'F5_W_ROUND08_PARALLEL_SEALING_CLARIFICATION_20261026.md','owner_ruling':'OWNER_RULING_COLAB_BUDGET_20261025.txt'}
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE']});os.environ['SNAKE_DQN_DEVICE']='cpu';sys.path.insert(0,str(R))
from research.bl_ppo.exports import export_bytes,export_json,verify_export,digest
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def bind(p):
 p=pathlib.Path(p);assert p.is_file() and not p.is_symlink(),str(p)
 return {'path':str(p),'sha256':digest(p),'size':p.stat().st_size}
def write(p,v):return export_bytes(p,v if isinstance(v,bytes) else v.encode()) if isinstance(v,(str,bytes)) else export_json(p,v)
def check(d):
 b=bind(d['path']);assert b['sha256']==d['sha256'],d['path']
 if 'size' in d:assert b['size']==d['size'],d['path']
def git(*a):return subprocess.check_output(['git',*a],cwd=R,timeout=30)
def run(argv,timeout=60):
 start=now();p=subprocess.run([str(x) for x in argv],cwd=R,capture_output=True,text=True,timeout=timeout);d={'argv':[str(x) for x in argv],'started_utc':start,'finished_utc':now(),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0,d;return d
def command():return [str(R/'tools/ops/dispatch.sh'),JOB,str(P/'manifest.json'),'--broker',str(R),'--cwd',str(R),'--state-dir',str(P/'state-01'),'--dispatch-txt',str(P/'DISPATCH.txt')]
def fresh():
 assert git('rev-parse','HEAD').decode().strip()==BROKER and git('status','--porcelain')==b''
 for p in [P/'state-01',P/'output-01',OPS/'dispatch'/JOB,OPS/'logs'/f'{JOB}.log',OPS/'manifests'/f'{JOB}.json']:assert not p.exists(),str(p)
def protected(p):return any('d4096' in s.lower() or s=='t34-round0-study-01' for s in pathlib.Path(p).parts)
def refs(v):
 if isinstance(v,dict):
  if v.get('read') is False:return
  if {'path','sha256'}<=v.keys() and isinstance(v['path'],str) and pathlib.Path(v['path']).is_absolute():yield v
  for k,x in v.items():
   if k not in ['prohibited_references_not_followed','protected_references_not_followed']:yield from refs(x)
 elif isinstance(v,list):
  for x in v:yield from refs(x)
def companion(p):
 p=pathlib.Path(p)
 for s,raw in [('.sha256',(digest(p)+'  '+p.name+'\n').encode()),('.complete',b'complete\n')]:
  q=pathlib.Path(str(p)+s)
  if q.exists():assert q.read_bytes()==raw,str(q)
  else:
   with q.open('xb') as f:f.write(raw)
def inventory(m):
 rows=[]
 for x in m['inputs']:
  if 'url' in x:rows.append({**x,'transport':'https','size':x['size_bytes']})
  else:
   b=bind(x['path']);assert b['sha256']==x['sha256'];rows.append({**x,'transport':'upload','size':b['size']})
 private=sum(x['size'] for x in rows if x['transport']=='upload');public=sum(x['size'] for x in rows if x['transport']=='https');assert private<=45397643
 return {'schema':'bl-f5-complete-input-inventory/v2','rows':rows,'private_upload_bytes':private,'https_bytes':public,'total_input_bytes':private+public,'private_upload_cap_bytes':45397643,'private_cap_pass':True,'inspection':'Opaque SHA/size only; no selected-training/checkpoint deserialization.'}
phase=sys.argv[1]
if phase=='prepare':
 fresh();write(P/'operative-builder-v2.py',pathlib.Path(__file__).read_bytes())
 write(P/'held-plan02-inventory.json',{str(x):bind(x) for x in sorted(H.rglob('*')) if x.is_file()})
 # Retain all partial helper bytes as NONOPERATIVE provenance; never execute them.
 write(P/'preparation-history-v2.json',{'status':'RECOVERED_LOCAL_METADATA_PREPARATION_ONLY','created_utc':now(),'prior_trace':bind(P/'PREPARATION_RECOVERY_01.md'),'nonoperative_helpers':[bind(P/x) for x in ['build-successor.py','seal-successor.py','build-package.py']],'rule':'Only operative-builder-v2.py is the current preparation/verification program. All draft bytes remain unchanged as provenance.','runtime_attempt_consumed':False,'colab_calls':0,'dispatches':0})
 prior=load(C/'historical-accounting.json');paths=[pathlib.Path(x['receipt']['path']) for x in prior['attempts']]+[C/'output-05/receipt.json'];rows=[]
 for i,path in enumerate(paths,1):
  r=load(path);assert r['stop_verified'] is True
  dest=P/'history'/f'receipt-{i:02d}.json';write(dest,path.read_bytes())
  accounting=r['accounting'];priced=accounting.get('priced_rate_units_per_hour',accounting['own_rate_units_per_hour']);observed=accounting.get('observed_rate_units_per_hour')
  rows.append({'original_receipt':bind(path),'receipt':bind(dest),'job_id':r['job_id'],'status':r['status'],'stop_verified':r['stop_verified'],'elapsed_seconds':r['elapsed_seconds'],'raw_accounting':accounting,'usage_before':r.get('usage_before'),'usage_after':r.get('usage_after'),'observed_balance_decrease_units':r['usage_before']['balance']-r['usage_after']['balance'],'full_priced_exposure_units':r['elapsed_seconds']*priced/3600,'full_observed_rate_exposure_units':None if observed is None else r['elapsed_seconds']*observed/3600})
 assert len(rows)==7 and rows[4]['job_id']=='bl-f5-w-plan-round08-c989f673-01' and rows[5]['job_id']=='bl-f5-w-timing-round08-c989f673-01'
 h={'schema':'bl-f5-successor-accounting/v2','created_utc':now(),'attempts':rows,'prior_six_session_history':bind(C/'historical-accounting.json'),'prior_six_session_history_verbatim':prior,'cumulative_session_elapsed_seconds':sum(x['elapsed_seconds'] for x in rows),'sum_observed_balance_decreases_units':sum(x['observed_balance_decrease_units'] for x in rows),'sum_full_session_priced_exposure_units':sum(x['full_priced_exposure_units'] for x in rows),'sum_full_session_observed_rate_exposure_units':sum(x['full_observed_rate_exposure_units'] for x in rows if x['full_observed_rate_exposure_units'] is not None),'accounting_interpretation':'Full saved raw accounting and crosschecks retained. Observed balances, last crosscheck debit, and integrated elapsed/rate exposure are distinct; none is claimed as precise final provider billing. No historical clock/debit/envelope reset.','held_plan02_executed':False,'successor_execution_count':0,'completed_science_units':0,'cumulative_envelope':load(H/'bindings.json')['prior_cumulative_envelope'],'base_timeout_minutes':15,'effective_timeout_minutes':30,'budgets_are_runaway_guards_only':True,'limits_redoubled':False}
 assert h['cumulative_session_elapsed_seconds']==3782.8593037500978
 write(P/'historical-accounting.json',h)
 write(P/'qualified-registry.json',REG.read_bytes());result=pathlib.Path(load(REG)['acceptance_result']);write(P/'qualified-acceptance-result.json',result.read_bytes());assert digest(result)==load(REG)['acceptance_result_sha256'] and load(result)['qualified'] is True
 from research.bl_residual.f5_package import source_inventory,governance,source_review
 pins=source_inventory(R);old=load(H/'control.json');c=copy.deepcopy(old)
 c['source_hashes']=pins;c['identity']['source_sha256']=sha(json.dumps(pins,sort_keys=True,allow_nan=False).encode());c['operative_package']=str(P)
 c.pop('attempt04_environment_reference',None);c['environment_capture']=load(P/'capture05-environment.json');c['attempt05_environment_reference']=bind(P/'capture05-environment.json');c['cross_assignment_comparison']='v5 required against embedded admitted attempt-05 raw object'
 c['operative_authority']={k:bind(P/v) for k,v in NAMES.items()};c['transport_repair_ruling']=bind(P/'ACCEPT_TRANSPORT_REPAIR_7f5357a7.md');c['qualified_registry']=bind(P/'qualified-registry.json');c['qualified_acceptance']=bind(P/'qualified-acceptance-result.json')
 c['attempt02_authority']={'disposition':bind(P/'F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026.md'),'held_unexecuted_plan02_seal':bind(H/'PACKAGE_SEAL_OPERATIONAL.json'),'two_mechanical_failure_receipts':[bind(P/'history/receipt-05.json'),bind(P/'history/receipt-06.json')],'historical_accounting':bind(P/'historical-accounting.json'),'manual_attempt_maximum':1,'automatic_retries':False,'named_successor_not_a_third_execution':True}
 c['source_commit']=PIN;c['accepted_f5_implementation']=IMPL;c['allocation']=sha(json.dumps({k:v for k,v in c.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 ledger={'schema':'bl-f5-verified-ledger/v1','namespace':c['namespace'],'allocation_sha256':c['identity']['allocation_sha256'],'allocation':c['allocation'],'status':'EMPTY_VERIFIED_LEDGER_NO_COMPLETED_UNITS','units':[]};write(P/'verified-ledger.json',ledger);c['ledger_sha256']=verify_export(P/'verified-ledger.json');write(P/'control.json',c)
 changed={}
 for rel,hsh in pins.items():
  raw=(R/rel).read_bytes();assert sha(raw)==hsh and raw==git('show',f'{PIN}:{rel}')
  if rel.startswith('research/colab_cli/'):
   if raw!=git('show',f'{IMPL}:{rel}'):changed[rel]={'old_sha256':sha(git('show',f'{IMPL}:{rel}')),'new_sha256':hsh}
  else:assert raw==git('show',f'{IMPL}:{rel}'),rel
 assert git('rev-parse',f'{PIN}:research/colab_cli').decode().strip()==TREE
 write(P/'source-identity.json',{'status':'PASS','source_commit':PIN,'broker_commit':BROKER,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'source_hashes':pins,'changed_transport_closure':changed,'nontransport_closure_equals_c989f673':True})
 b={'schema':'bl-f5-successor-bindings/v2','source_commit':PIN,'broker_commit':BROKER,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'held_plan02_historical_bindings':bind(H/'bindings.json'),'held_plan02_seal':bind(H/'PACKAGE_SEAL_OPERATIONAL.json'),'held_plan02_inventory':bind(P/'held-plan02-inventory.json'),'original_rulings':{k:bind(P/v) for k,v in NAMES.items()},'transport_repair_ruling':bind(P/'ACCEPT_TRANSPORT_REPAIR_7f5357a7.md'),'attempt01_disposition':bind(P/'F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026.md'),'source_identity':bind(P/'source-identity.json'),'control':bind(P/'control.json'),'empty_verified_ledger':bind(P/'verified-ledger.json'),'historical_accounting':bind(P/'historical-accounting.json'),'capture05_environment':bind(P/'capture05-environment.json'),'capture05_postpublication_verification':bind(P/'capture05-post-publication-verification.json'),'capture05_latency_summary':bind(P/'capture05-latency-summary.json'),'capture05_published_outputs':bind(C/'OUTPUT_PUBLICATION_05.json'),'capture05_evidence_publication_commit':PIN,'qualified_registry':bind(P/'qualified-registry.json'),'qualified_acceptance':bind(P/'qualified-acceptance-result.json'),'both_mechanical_failure_receipts':[bind(P/'history/receipt-05.json'),bind(P/'history/receipt-06.json')],'source_acceptance':load(H/'bindings.json')['references'],'common_identity':c['identity'],'frozen_allocation':bind(F/'allocation.json'),'fixture_descriptor':bind(F/'timing-fixtures.json'),'science_admission':False}
 write(P/'bindings.json',b)
 write(P/'governance-source-preflight.json',{'status':'PASS','created_utc':now(),'governance':governance(R,A),'source_semantic_review':source_review(R),'science_payloads_opened':0,'allocations':0})
 print('PREPARED',len(pins),'source pins; seven historical sessions')
if phase=='preflight':
 fresh()
 from research.bl_residual.f5_job import _control,_source_admission
 from research.bl_residual.f5_environment import validate_environment,compare_environment
 from research.bl_residual.f5_qualification import load_fixtures
 c=_control(P);_source_admission(c);env=load(P/'capture05-environment.json');assert c['environment_capture']==env;validate_environment(env);comparison=compare_environment(env,env,mode='cross-assignment')
 fixtures,fh=load_fixtures(F/'timing-fixtures.json');assert len(fixtures)==12 and fh==c['qualification_fixture_sha256']
 write(P/'environment-reference-preflight.json',{'status':'PASS','created_utc':now(),'environment':bind(P/'capture05-environment.json'),'exact_capture05_raw_bytes':(P/'capture05-environment.json').read_bytes()==(C/'output-05/exports/environment.json').read_bytes(),'comparison':comparison,'control_source_admission':'PASS','fixture_count':len(fixtures),'fixture_sha256':fh,'new_runtime_environment':'Future runtime must independently capture and pass v5 cross-assignment comparison.'})
 receipt=load(C/'output-05/receipt.json');saved=C/'state-05'/receipt['job_id']/'sessions.json';assert load(saved)=={} and receipt['stop_verified'] is True and receipt['usage_after']['assignments']==receipt['usage_after']['rate']==0
 account=R.parent.parent.parent/'.config/colab-cli/snake' # use explicit owner account path below
 account=pathlib.Path('/Users/josenunez/.config/colab-cli/snake');active=[bind(x) for x in sorted(account.glob('active-*.json'))];pending=[bind(x) for x in sorted(account.glob('pending-*.json'))]
 write(P/'saved-account-snapshot.json',{'status':'SAVED_POSTRELEASE_NOT_LIVE','created_utc':now(),'receipt':bind(C/'output-05/receipt.json'),'saved_sessions':bind(saved),'sessions':{},'usage':receipt['usage_after'],'other_assignments':{},'local_active_broker_records':active,'local_pending_broker_records':pending,'local_records_match_saved_empty_snapshot':not active and not pending,'current_account_availability_claimed':False,'colab_calls':0,'dispatch_rule':'Wrapper must refresh actual endpoint/rates under allocation lock and refuse changes to the sealed empty map; no current availability inferred.'})
 assert not active and not pending,'local account records require reconciliation before sealing'
 m=copy.deepcopy(load(H/'manifest.json'));m.update({'commit':PIN,'job_id':JOB,'output_dir':str(P/'output-01'),'supervisor_state_dir':str(P/'state-01')})
 for x in m['inputs']:
  if x['dest'] in ['control.json','verified-ledger.json']:x.update({'path':str(P/x['dest']),'sha256':verify_export(P/x['dest'])})
 m['resume_exemption']['reason']='Named successor of held unexecuted plan attempt02 under ACCEPT_TRANSPORT_REPAIR_7f5357a7 and F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026; one manual identically scoped plan materialization/disjoint CUDA execution only; no science fit/look or automatic allocation/job/completion retry.'
 m['budget']['other_assignments']={}
 from research.colab_cli.supervisor import input_preflight,validate
 start=now();public=input_preflight(m);assert len(public)==1 and public[0]['sha256']==c['qualification_b8_sha256'] and public[0]['size_bytes']==6562005
 write(P/'public-b8-preflight.json',{'status':'PASS','started_utc':start,'finished_utc':now(),'verified_fetches':public,'public_release_provenance':bind(R/'research/t34_exit_b8/PUBLIC_INPUTS.json'),'payload_saved':False,'checkpoint_deserialized':False,'colab_calls':0})
 m['operational_admission']={'schema':'bl-f5-plan-operational-admission/v1','status':'ACCEPTED_PLAN_MATERIALIZATION_AND_TRANSPORT_ONLY','scientific_fits_authorized':False,'numerical_looks_authorized':False,'release_authorized':False,'rulings':{k:bind(P/v) for k,v in NAMES.items()},'transport_repair_ruling':bind(P/'ACCEPT_TRANSPORT_REPAIR_7f5357a7.md'),'capture05_postpublication_verification':bind(P/'capture05-post-publication-verification.json'),'bindings':bind(P/'bindings.json'),'historical_accounting':bind(P/'historical-accounting.json'),'saved_account_snapshot':bind(P/'saved-account-snapshot.json'),'public_b8_preflight':bind(P/'public-b8-preflight.json'),'accounting_declaration_must_match_seal':True,'independent_operative_review_required':True}
 inv=inventory(m);m['known_private_input_bytes']=inv['private_upload_bytes'];m['f5'].update({'known_private_input_bytes':inv['private_upload_bytes'],'https_input_bytes':inv['https_bytes'],'total_input_bytes':inv['total_input_bytes'],'remaining_private_bytes_for_plan_and_gate_metadata':45397643-inv['private_upload_bytes']});write(P/'inventory-metadata.json',inv);write(P/'manifest.json',m);validate(m,R)
 from research.colab_cli.source import prepare
 plan=prepare(R,m,P/'source-preflight-artifacts')
 for p in sorted((P/'source-preflight-artifacts').iterdir()):companion(p)
 for rel,row in plan['files'].items():
  raw=git('show',f'{PIN}:{rel}');assert sha(raw)==row['sha256'] and len(raw)==row['size'] and raw==(R/rel).read_bytes()
 for rel,hsh in c['source_hashes'].items():assert plan['files'][rel]['sha256']==hsh
 with tarfile.open(P/'source-preflight-artifacts/source.tar.gz') as tar:
  members=tar.getmembers();assert sorted(x.name for x in members)==sorted(plan['files'])
  for x in members:assert sha(tar.extractfile(x).read())==plan['files'][x.name]['sha256']
 write(P/'source-preflight.json',{'status':'PASS','source_pin_count':len(c['source_hashes']),'source_paths_count':len(m['source_paths']),'archive_members':len(members),'source_plan':plan,'all_git_worktree_archive_bytes_match':True,'source_commit':PIN,'broker_commit':BROKER})
 cmds=[run([PY,R/'tools/ops/qualification.py','check','--repo',R,'--record',REG,'--ref',v]) for v in ['HEAD',PIN]]
 cmds.append(run([PY,'-m','research.colab_cli.acceptance','--repo',R,'--verify-qualified-record',REG,'--required-inventory-sha256','b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea','--reject-tree','c148431277f4e33c3f589251df892ea158407e25']))
 cmds.append(run([R/'research/colab_cli/launch.sh','--state-dir',P/'state-01','--colab','/Users/josenunez/.local/bin/colab','validate',P/'manifest.json']))
 write(P/'broker-validation.json',{'status':'PASS','created_utc':now(),'commands':cmds,'registry':bind(REG),'broker_commit':BROKER,'source_commit':PIN,'manifest':bind(P/'manifest.json'),'colab_calls':0})
 text=f'''F5 plan attempt02 named successor after transport requalification. Prepared; no dispatch performed.
Source commit {PIN}; accepted F5 bytes {IMPL}; CLI tree {TREE}; clean broker commit {BROKER}.
Manifest SHA256 {bind(P/'manifest.json')['sha256']}.
Scope: F5_MODE=prepare-qualification; shared ordered2000x256 CPU-generator plan, seed3403,128 doom then128 ordinary, exact rows/initial-final RNG contract; disjoint source-native CUDA transport. No scientific fit/look, serving or release authority.
This succeeds held unexecuted ROUND08_02, not a third execution. Hold package and both mechanical receipts preserved. All seven historical sessions/debit retained. Exactly one manual dispatch following independent concrete operative review and main publication. No automatic allocation/job/completion retry.
Base15/effective30minute runaway timeout retained, estimate15 unchanged; base1/effective2 unit cap, base.5/effective1 reserve retained without redoubling. Owner ruling forbids budget-only refusal. Original science caps retained.
Seven inputs, nine exports, private bytes{inv['private_upload_bytes']} <=45397643, verified transfer<=150seconds, staging<=300seconds, max export2GiB. L4>=6vCPU,cuda:0, exact69-wheel/cu126/Python3.12.3 fp32 controls, threads1.
Capture05 raw v5 environment is the reference. Future runtime must independently capture and pass cross-assignment comparison. Capture bootstrap is not final science family-canary qualification.
Account evidence is SAVED_POSTRELEASE_NOT_LIVE (capture05). No account/Colab/provider calls made here. Wrapper must freshly reconcile endpoints/rates under lock, account cap3; any change from sealed other_assignments={{}} refuses and requires a new successor. Saved zero assignments is not current availability.
Recheck registry, source/broker pins, all SHA/markers, main publication and unused job/state/output immediately before dispatch. Verify PACKAGE_SEAL_OPERATIONAL.json and PACKAGE_PUBLICATION.json.
Timing is not sealed by this preparation. Before science require separately verified plan/timing, Mac evidence, exact eventual upload family-canary through f5_package.bind_qualification_inventory, and separate ACCEPT_F5_FIT.
Exact operator command:
{shlex.join(command())}
'''
 write(P/'DISPATCH.txt',text);write(P/'README.md',text+'\nDraft helpers build-successor.py,seal-successor.py,build-package.py are retained NONOPERATIVE provenance. Use operative-builder-v2.py for preparation/verification; the only future runtime command is above.\n')
 dr=run(command()+['--dry-run']);fresh();assert dr['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 write(P/'dispatch-dry-run.json',{'status':'PASS','created_utc':now(),'command':dr,'manifest':bind(P/'manifest.json'),'live_accounting':'NOT_CHECKED','colab_calls':0,'dispatches':0,'state_absent':True,'output_absent':True});print(dr['stdout'].strip());print('private bytes',inv['private_upload_bytes'])
if phase=='verify':
 fresh();m=load(P/'manifest.json');c=load(P/'control.json');l=load(P/'verified-ledger.json');old=load(H/'control.json');om=load(H/'manifest.json')
 assert c['allocation']==sha(json.dumps({k:v for k,v in c.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 assert l['allocation']==c['allocation']!=old['allocation'] and l['namespace']==c['namespace']==old['namespace'] and l['units']==[] and l['allocation_sha256']==old['identity']['allocation_sha256'];assert c['ledger_sha256']==verify_export(P/'verified-ledger.json')
 for k in ['input_hashes','environment_policy','qualification_b8_sha256','qualification_fixture_sha256','namespace','scientific_admission','status','schema']:assert c[k]==old[k],k
 for k in ['allocation_sha256','corpus_sha256','environment_sha256','loss_sha256']:assert c['identity'][k]==old['identity'][k],k
 assert c['environment_capture']==load(C/'output-05/exports/environment.json');assert c['source_hashes']==load(C/'source-identity.json')['source_hashes']
 for k in ['script','env','requirements','timeout_minutes','lease_seconds','max_export_bytes','source_paths','required_exports','substeps','device','canary','scientific_admission','timing_calibration','budget']:assert m[k]==om[k],k
 for k in ['base_timeout_minutes','effective_timeout_minutes','automatic_retries','technical_guards','transfer_bar_seconds','staging_bar_seconds','private_upload_cap_bytes','private_upload_bytes_max']:assert m['f5'][k]==om['f5'][k],k
 assert m['timeout_minutes']==m['resume_exemption']['timeout_minutes']==30 and m['commit']==PIN and m['job_id']==JOB
 assert inventory(m)==load(P/'inventory-metadata.json') and len(m['inputs'])==7 and len(m['required_exports'])==9
 assert load(REG)==load(P/'qualified-registry.json')
 for d in load(P/'held-plan02-inventory.json').values():check(d)
 for d in load(P/'predecessor-inventory.json').values():check(d)
 dr=load(P/'dispatch-dry-run.json');assert dr['command']['argv']==command()+['--dry-run'] and dr['command']['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED';assert shlex.join(command()) in (P/'DISPATCH.txt').read_text()
 hist=load(P/'historical-accounting.json');assert len(hist['attempts'])==7 and hist['cumulative_session_elapsed_seconds']==sum(load(x['receipt']['path'])['elapsed_seconds'] for x in hist['attempts'])
 for x in hist['attempts']:check(x['receipt']);check(x['original_receipt']);assert pathlib.Path(x['receipt']['path']).read_bytes()==pathlib.Path(x['original_receipt']['path']).read_bytes()
 external={};skipped=[]
 roots=[load(p) for p in sorted(P.rglob('*.json'))]+[load(C/'bindings-v5.json'),load(H/'bindings.json'),load(F/'REVIEW_PACKET.json'),load(F/'REVIEW_BUNDLE.json')]
 for value in roots:
  for d in refs(value):
   if protected(d['path']):skipped.append({'path':d['path'],'read':False});continue
   check(d)
   if not pathlib.Path(d['path']).is_relative_to(P):external[d['path']]=bind(d['path'])
 for rel in m['source_paths']+['tools/ops/dispatch.py','tools/ops/dispatch.sh','tools/ops/qualification.py']:
  p=R/rel;external[str(p)]=bind(p)
 historical_sidecars=[]
 for d in list(external.values()):
  p=pathlib.Path(d['path'])
  if p.name.endswith(('.sha256','.complete')):continue
  for suffix in ['.sha256','.complete']:
   q=pathlib.Path(str(p)+suffix)
   if not q.is_file():continue
   external[str(q)]=bind(q)
   if suffix=='.sha256' and q.read_text().split()[0]!=d['sha256']:
    assert p.is_relative_to(R) and q.is_relative_to(R),str(q);rel=str(p.relative_to(R));qr=str(q.relative_to(R));assert p.read_bytes()==git('show',f'{IMPL}:{rel}') and q.read_bytes()==git('show',f'{IMPL}:{qr}');historical_sidecars.append({'source':d,'sidecar':bind(q),'status':'PREEXISTING_NONOPERATIVE_TRACKED_SIDECAR; operative Git/current source digest verified'})
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):companion(p);verify_export(p)
 report={'schema':'bl-f5-successor-operative-verification/v2','status':'PASS','created_utc':now(),'source_commit':PIN,'broker_commit':BROKER,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'manifest':bind(P/'manifest.json'),'control':bind(P/'control.json'),'ledger':bind(P/'verified-ledger.json'),'checks':{'seven_inputs_nine_exports':True,'source_192pins_194paths_200archive_members':load(P/'source-preflight.json')['archive_members']==200,'nontransport_F5_bytes_equal_c989f673':True,'capture05_exact_environment_v5':True,'unchanged_scope_and_science_identities':True,'recomputed_allocation_and_empty_ledger':True,'original_base_effective_caps_not_redoubled':True,'seven_sessions_raw_accounting_and_two_mechanical_receipts':True,'held_unexecuted_plan02_unchanged':True,'qualified_consumers_and_broker_validation':True,'dryrun_and_dispatch_agree':True,'saved_not_live_account_evidence':True,'state_output_unused_broker_clean':True,'all_package_sidecars_and_markers':True},'external_files_verified':len(external),'external_files':external,'protected_references_not_followed':skipped,'preexisting_nonoperative_source_sidecars':historical_sidecars,'colab_calls':0,'dispatches':0,'science_payloads_opened':0,'draft_helpers':'Retained as NONOPERATIVE provenance; all bytes hashed, never executed for preflights or dispatch.'}
 write(P/'operational-verification-v2.json',report);print('VERIFICATION_PASS',len(external),'external files')
if phase=='seal':
 fresh();v=load(P/'operational-verification-v2.json');review=load(P/'independent-review.json');assert review['status']=='NO_BLOCKING_FINDINGS'
 for d in v['external_files'].values():check(d)
 for d in load(P/'held-plan02-inventory.json').values():check(d)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 files={str(p.relative_to(P)):bind(p) for p in sorted(P.rglob('*')) if p.is_file()}
 seal={'schema':'bl-f5-plan-successor-operative-seal/v2','status':'SEALED_ACCEPTED_PLAN02_SUCCESSOR_AFTER_TRANSPORT_REQUALIFICATION','created_utc':now(),'package_path':str(P),'source_commit':PIN,'broker_commit':BROKER,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'job_id':JOB,'manifest':bind(P/'manifest.json'),'control':bind(P/'control.json'),'ledger':bind(P/'verified-ledger.json'),'bindings':bind(P/'bindings.json'),'verification':bind(P/'operational-verification-v2.json'),'independent_review':bind(P/'independent-review.json'),'files':files,'external_files':v['external_files'],'dispatch_command':shlex.join(command()),'dry_run_output':load(P/'dispatch-dry-run.json')['command']['stdout'].strip(),'state_dir':str(P/'state-01'),'output_dir':str(P/'output-01'),'state_absent':True,'output_absent':True,'scientific_admission':False,'timing_sealed':False,'colab_calls':0,'dispatches':0,'science_payloads_opened':0,'held_plan02_unchanged':True,'account_evidence':'Saved postrelease only; real wrapper must refresh under lock and reject drift.','self_reference_exclusion':'Seal/companions are bound by later create-only PACKAGE_PUBLICATION.json; publication envelope is the only subsequent package addition.'};write(P/'PACKAGE_SEAL_OPERATIONAL.json',seal);print(json.dumps(bind(P/'PACKAGE_SEAL_OPERATIONAL.json')))
