"""Create-only round08 plan package. Never allocates or dispatches a runtime."""
from __future__ import annotations
import copy, datetime, gzip, hashlib, json, os, pathlib, re, shlex, subprocess, sys, tarfile, time
R=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5w')
B=R.parent/'snake-dqn-artifacts/big-lever/exp/astra-20261024'
D=B/'F5_W_NEXT_AFTER_ATTEMPT04_DRAFT_20261026_v2'
P=B/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02'
OLD=B/'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_01'
TIMING=B/'F5_W_TIMING_OPERATIONAL_20261026_ROUND08_01'
DISPOSITION='F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026.md'
T=B/'F5_W_TIMING_DRAFT_20261026_ROUND08_v3'
C=B/'F5_W_CAPTURE_AFTER_22e86986_REPAIR_04'
F=B/'F5_W_PREFIT_20261025/final-09'
PIN='c989f6738f6aad143eb97bc47f6bd9e6687057ce'
JOB='bl-f5-w-plan-round08-c989f673-02'
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
if phase=='document':
 fresh()
 write(P/'seal-publication-builder.py',pathlib.Path(__file__).read_bytes())
 write(P/'README.md',f'''# F5 round08 operative plan attempt02

Status: accepted plan materialization/disjoint CUDA transport only; scientific_admission=false.
Implementation {PIN}; job {JOB}. Exactly one future manual dispatch, no automatic allocation/job/completion retry. No runtime dispatched by this preparation.

Authority is F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026.md plus the four original accepted rulings. Attempt01 remains INCOMPLETE, saved stop verified. Its output-01 was published first at commit a9829745c5c024adcef938f0f1b7046b7ea78c60; the publication envelope and terminal/calibration/accounting are bound here. No completed units exist to import. Own namespace/frozen allocation semantics are retained; operative allocation and empty ledger recomputed for this package.

Timing initially occupied actual endpoint gpu-l4-s-ass1a0-2ptzg6d08d12q at observed1.54u/h (priced2.0). It ended independently during preparation: INCOMPLETE, saved stop_verified=true; fresh account preflight observes0 assignments/0u/h, balance1066.22. Following the disposition's sequential refresh-after-release clause, other_assignments={{}}. Timing terminal/rate snapshots are read-only; no timing state/output was modified. Initial builder and its stopped pre-manifest preparation are retained, followed by create-only preflight-builder-v2.py; no seal was altered or job attempted.

All original scientific identities, selected-training input, ordered2000x256 plan and RNG contract, source192 pins/194paths/200archive members, exact69-wheel/cu126 environment and raw admitted environment object, seven inputs, nine required exports and four no-retry fences are unchanged. Numeric threads1; deterministic fp32; no compile; L4>=6vCPU/cuda:0; forecast15min unchanged; base15/effective30min manifest/exemption, base1/effective2 units and base.5/effective1 reserve retained without redoubling. Transport control30/recovery60s; inclusive transfer150s; staging300s; private cap45397643 bytes; export2GiB. Owner ruling forbids budget-only refusal. Full failed elapsed/debit and cumulative envelopes are retained; cumulative-accounting-final.json also includes the separately ended timing attempt.

Fresh HTTPS/source/environment/governance/account/broker preflights and wrapper dry-run are actual saved evidence. Dry-run does not check live accounting. Recheck account/name/state/output immediately before single operator dispatch; a changed sealed endpoint/rate map requires a fresh published successor rather than mutation. A second occurrence of this plan failure stops unchanged repetition and requires diagnosis/repair/requalification before any further plan attempt.

See DISPATCH.txt for exact future operator command. Verify PACKAGE_SEAL_OPERATIONAL.json and PACKAGE_PUBLICATION.json sidecars plus local/origin publication first. Final science waits for both verified output branches, Mac timing/calibration, final matching canary and separate ACCEPT_F5_FIT. Serving/release still require owner authority.
''')
 calls=[run(['/Users/josenunez/.local/bin/colab','--auth','oauth2',x]) for x in ['sessions','usage','sessions']]
 assert all('No active sessions found on server.' in calls[i]['stdout'] for i in [0,2])
 assert 'Active assignments: 0' in calls[1]['stdout'] and 'Usage rate: 0.00/hr' in calls[1]['stdout']
 write(P/'account-recheck-before-seal.json',{'status':'PASS','created_utc':now(),'calls':calls,'sealed_other_assignments':load(P/'manifest.json')['budget']['other_assignments'],'unchanged_since_manifest_preflight':True,'allocations':0,'timing_state_output_modified':False})
 print('DOCUMENTED_AND_ACCOUNT_RECHECKED')
if phase=='verify':
 fresh();m=load(P/'manifest.json');c=load(P/'control.json');l=load(P/'verified-ledger.json');old=load(OLD/'control.json');om=load(OLD/'manifest.json')
 assert m['job_id']==JOB and m['commit']==PIN and m['scientific_admission'] is False
 assert m['output_dir']==str(P/'output-01') and m['supervisor_state_dir']==str(P/'state-01')
 assert c['allocation']==sha(json.dumps({k:v for k,v in c.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 assert l['allocation']==c['allocation']!=old['allocation'] and l['namespace']==c['namespace']==old['namespace'] and l['units']==[] and l['allocation_sha256']==c['identity']['allocation_sha256']
 assert c['ledger_sha256']==verify_export(P/'verified-ledger.json')
 ignored={'allocation','ledger_sha256','operative_package','operative_authority','attempt02_authority'}
 assert {k:v for k,v in c.items() if k not in ignored}=={k:v for k,v in old.items() if k not in ignored}
 assert c['input_hashes']=={'selected-training.tar.xz':'2c033750b0d25568b19ba81aec7bd0e9b0583bdd32a994a9a1984a409e45c46c'}
 assert c['environment_capture']==load(C/'output-04/exports/environment.json')
 unchanged=['commit','script','env','requirements','timeout_minutes','lease_seconds','max_export_bytes','source_paths','required_exports','substeps','device','canary','scientific_admission']
 for k in unchanged:assert m[k]==om[k],k
 assert m['timeout_minutes']==m['resume_exemption']['timeout_minutes']==30
 assert 'F5_W_PLAN_ATTEMPT01_DISPOSITION_20261026' in m['resume_exemption']['reason']
 assert set(m['operational_admission']['rulings'])==set(RULINGS)
 for k in ['no_automatic_retry','no_allocation_retry','no_job_retry','no_completion_retry']:assert m['timing_calibration'][k] is True
 assert m['budget']==om['budget']=={'rate_units_per_hour':2.0,'unit_cap':2.0,'reserve_units':1.0,'other_assignments':{}}
 for k in ['transfer_bar_seconds','staging_bar_seconds','private_upload_cap_bytes','private_upload_bytes_max','base_timeout_minutes','effective_timeout_minutes','automatic_retries','technical_guards']:assert m['f5'][k]==om['f5'][k],k
 for n in ['cli_guard.py','supervisor.py','policy.py','transfers.py']:
  p=R/'research/colab_cli'/n;assert p.read_bytes()==git('show',f'{PIN}:research/colab_cli/{n}')
 inventory=inv(m);assert inventory==load(P/'inventory-metadata.json')
 assert len(inventory['rows'])==7 and len(m['required_exports'])==9
 assert m['known_private_input_bytes']==m['f5']['known_private_input_bytes']==inventory['private_upload_bytes']
 assert m['budget']['other_assignments']==load(P/'live-account-preflight.json')['other_assignments']=={}
 tr=load(P/'timing-terminal-receipt-snapshot.json');assert tr['stop_verified'] is True
 hist=load(P/'historical-accounting.json');final=load(P/'cumulative-accounting-final.json')
 assert hist['attempt01_elapsed_seconds']==load(P/'attempt01-terminal-receipt.json')['elapsed_seconds']
 assert hist['attempt01_observed_balance_decrease_units']==0.03 and hist['cumulative_accounting_not_reset'] is True
 assert final['cumulative_elapsed_seconds_including_capture_plan01_timing']==hist['capture_and_plan01_cumulative_elapsed_seconds']+tr['elapsed_seconds']
 assert load(P/'dispatch-dry-run.json')['command']['argv']==command()+['--dry-run']
 assert load(P/'dispatch-dry-run.json')['command']['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 assert shlex.join(command()) in (P/'DISPATCH.txt').read_text()
 for d in load(P/'predecessor-inventory.json').values():check(d)
 # Verify all structured references in every operative JSON without reading outcome payloads.
 external={};skipped=[]
 roots=[load(p) for p in sorted(P.glob('*.json'))]
 roots.extend([load(C/'bindings-v5.json'),load(C/'accounting-owner-ruling-04.json'),load(C/'cumulative-envelope-owner-ruling-04.json'),load(F/'REVIEW_PACKET.json'),load(F/'REVIEW_BUNDLE.json')])
 for v in roots:
  for d in refs(v):
   if protected(d['path']):skipped.append({'path':d['path'],'read':False});continue
   check(d);external[d['path']]=bind(d['path'])
 for relative,h in c['source_hashes'].items():
  p=R/relative;d=bind(p);assert d['sha256']==h and p.read_bytes()==git('show',f'{PIN}:{relative}');external[str(p)]=d
 for relative in m['source_paths']:
  p=R/relative;external[str(p)]=bind(p)
 for name in ['tools/ops/dispatch.py','tools/ops/dispatch.sh','tools/ops/qualification.py']:
  p=R/name;external[str(p)]=bind(p)
 for d in list(external.values()):
  p=pathlib.Path(d['path'])
  if p.name.endswith(('.sha256','.complete')):continue
  for suffix in ['.sha256','.complete']:
   x=pathlib.Path(str(p)+suffix)
   if x.is_file():
    external[str(x)]=bind(x)
    if suffix=='.sha256':assert x.read_text().split()[0]==d['sha256'],str(x)
 for p in sorted((P/'source-preflight-artifacts').iterdir()):
  if p.name.endswith(('.sha256','.complete')):continue
  for suffix,data in [('.sha256',(bind(p)['sha256']+'  '+p.name+'\n').encode()),('.complete',b'complete\n')]:
   x=pathlib.Path(str(p)+suffix)
   if not x.exists():
    with x.open('xb') as stream:stream.write(data)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 assert load(P/'operational-tests.json')['status']=='PASS'
 report={'schema':'bl-f5-plan02-operative-consistency/v1','status':'PASS','created_utc':now(),'manifest':bind(P/'manifest.json'),'checks':{'exact_control_allocation_ledger_linkage':True,'identical_science_implementation_scope_limits':True,'exact_admitted_attempt04_raw_object':True,'selected_training_only_bound_inputs':True,'source_192pins_git_and_worktree_match':True,'source_paths_194_and_archive_200':True,'exact_seven_inputs_nine_exports':True,'all_inventories_counts_recomputed':True,'actual_https_source_environment_governance_broker_account_preflights':True,'scientific_admission_false_retained':True,'four_no_retry_fences_retained':True,'attempt02_explicit_one_manual_exception':True,'base_effective_limits_not_redoubled':True,'timing_ended_independently_fresh_postrelease_account':True,'timing_actual_endpoint_rate_receipt_bound_readonly':True,'dryrun_command_agrees_dispatch_instructions':True,'predecessor_files_rehashed_unchanged':True,'historical_failed_debit_elapsed_and_cumulative_envelope_retained':True,'all_payload_sidecars_markers_verified':True,'unused_state_output_name_rechecked':True},'external_files_verified':len(external),'protected_references_not_followed':skipped,'opaque_input_handling':'Hashes/sizes only, no label/checkpoint deserialization/outcome inspection.','allocations':0,'dispatches':0,'fits':0,'outcome_looks':0,'external_files':external}
 write(P/'internal-consistency.json',report)
 print(json.dumps({'status':'PASS','external_refs':len(external),'private_bytes':inventory['private_upload_bytes'],'predecessors':len(load(P/'predecessor-inventory.json'))}))
if phase=='seal':
 fresh();consistency=load(P/'internal-consistency.json');assert consistency['status']=='PASS'
 assert load(P/'independent-review.json')['status']=='PASS'
 for d in consistency['external_files'].values():check(d)
 for d in load(P/'predecessor-inventory.json').values():check(d)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 files={str(p.relative_to(P)):bind(p) for p in sorted(P.rglob('*')) if p.is_file()}
 write(P/'PACKAGE_SEAL_OPERATIONAL.json',{'schema':'bl-f5-plan-operative-package-seal/v1','status':'SEALED_ACCEPTED_PLAN_MATERIALIZATION_AND_TRANSPORT_ONLY','created_utc':now(),'package_path':str(P),'job_id':JOB,'implementation_commit':PIN,'manifest':bind(P/'manifest.json'),'control':bind(P/'control.json'),'ledger':bind(P/'verified-ledger.json'),'bindings':bind(P/'bindings.json'),'rulings':{k:bind(P/v) for k,v in RULINGS.items()},'attempt02_disposition':bind(P/DISPOSITION),'attempt01_terminal':bind(P/'attempt01-terminal-receipt.json'),'cumulative_accounting':bind(P/'cumulative-accounting-final.json'),'timing_operation':bind(P/'timing-operation-binding.json'),'files':files,'external_files':consistency['external_files'],'internal_consistency':bind(P/'internal-consistency.json'),'independent_review':bind(P/'independent-review.json'),'dispatch_command':shlex.join(command()),'dry_run_output':load(P/'dispatch-dry-run.json')['command']['stdout'].strip(),'state_dir':str(P/'state-01'),'output_dir':str(P/'output-01'),'state_absent':True,'output_absent':True,'scientific_admission':False,'scientific_fits_authorized':False,'numerical_looks_authorized':False,'release_authorized':False,'publication_required_before_operator_dispatch':True,'seal_self_reference':'Seal and companions are bound by subsequent create-only publication envelope.','freshness':'Current post-release snapshot only. Changed endpoint/rate map requires fresh successor sealing/publication; state/output/name recheck before dispatch.','allocations':0,'dispatches':0})
 print(json.dumps({'seal':bind(P/'PACKAGE_SEAL_OPERATIONAL.json'),'manifest':bind(P/'manifest.json'),'files':len(files),'external_files':len(consistency['external_files']),'dispatch':shlex.join(command())},indent=2))
if phase=='publish':
 fresh();seal=load(P/'PACKAGE_SEAL_OPERATIONAL.json')
 for d in seal['files'].values():check(d)
 for d in seal['external_files'].values():check(d)
 dest=R/'research/bl_residual/ledger'/P.name;assert not dest.exists();dest.mkdir()
 def put(p,b):
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as stream:stream.write(b)
 rows=[]
 for p in sorted(P.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(P);raw=p.read_bytes();q=dest/rel;row={'source':bind(p),'relative_path':str(rel)}
  if len(raw)>1900000:
   compressed=gzip.compress(raw,compresslevel=9,mtime=0);parts=[]
   for i,start in enumerate(range(0,len(compressed),1500000)):
    part=pathlib.Path(str(q)+f'.gz.part-{i:03d}');payload=compressed[start:start+1500000];write(part,payload);parts.append({'path':str(part.relative_to(dest)),'sha256':sha(payload),'size':len(payload)})
   index=pathlib.Path(str(q)+'.COMPRESSED_INDEX.json');write(index,{'original':bind(p),'encoding':'gzip-concatenated-parts','compressed_sha256':sha(compressed),'parts':parts});row['representation']={'index':str(index.relative_to(dest))}
   assert gzip.decompress(b''.join((dest[x['path']]).read_bytes() for x in parts))==raw
  else:put(q,raw);row['representation']={'path':str(rel)}
  rows.append(row)
 for p in sorted(dest.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):
   for suffix,raw in [('.sha256',(bind(p)['sha256']+'  '+p.name+'\n').encode()),('.complete',b'complete\n')]:
    x=pathlib.Path(str(p)+suffix)
    if x.exists():assert x.read_bytes()==raw
    else:put(x,raw)
 published={str(p.relative_to(dest)):bind(p) for p in sorted(dest.rglob('*')) if p.is_file()}
 envelope={'schema':'bl-f5-round08-complete-package-publication/v1','status':'SEALED_PLAN_ATTEMPT02_MATERIALIZATION_ONLY','created_utc':now(),'source_package':str(P),'ledger_directory':str(dest),'original_files':rows,'published_files':published,'lossless_large_metadata_reconstruction_verified':True,'science_payloads_copied':False,'dispatches':0,'operative_seal':bind(P/'PACKAGE_SEAL_OPERATIONAL.json'),'timing_state_output_modified':False}
 write(P/'PACKAGE_PUBLICATION.json',envelope)
 for suffix in ['', '.sha256','.complete']:put(dest/('PACKAGE_PUBLICATION.json'+suffix),(P/('PACKAGE_PUBLICATION.json'+suffix)).read_bytes())
 print(json.dumps({'package':str(P),'publication':bind(P/'PACKAGE_PUBLICATION.json'),'original_files':len(rows),'published_files':len(published)+3}))
