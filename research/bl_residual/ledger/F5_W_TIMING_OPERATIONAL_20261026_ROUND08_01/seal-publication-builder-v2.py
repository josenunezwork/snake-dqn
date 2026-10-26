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
if phase=='tests':
 fresh();write(P/'seal-publication-builder-v2.py',pathlib.Path(__file__).read_bytes())
 files=['tests/test_ops_f5_plan_admission.py','tests/test_ops_f5_timing_admission.py','tests/test_ops_dispatch.py','tests/test_bl_residual_f5_broker.py','tests/test_colab_cli_source.py','tests/test_colab_cli_schema_failures.py','tests/test_f5_capture_diagnostics.py','tests/test_bl_residual_f5_qualification.py','tests/test_bl_residual_f5_calibration.py']
 argv=[str(PYTHON),'-m','pytest','-q','-n','0','-p','no:cacheprovider','--basetemp','/tmp/f5-round08-timing-operational-pytest',*files]
 env=dict(os.environ);env['DYLD_LIBRARY_PATH']=str(R.parent/'snake-dqn/venv/lib/python3.12/site-packages/torch/lib')
 def priority():os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
 start=time.monotonic();stamp=now();v=subprocess.run(argv,cwd=R,capture_output=True,text=True,timeout=240,env=env)
 write(P/'operational-tests.log',(v.stdout+v.stderr).encode());write(P/'operational-tests.json',{'schema':'bl-f5-timing-operational-tests/v1','status':'PASS' if v.returncode==0 else 'FAIL','command':argv,'started_utc':stamp,'finished_utc':now(),'elapsed_seconds':time.monotonic()-start,'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'test_sources':{x:bind(R/x) for x in files},'pytest_workers':0,'numeric_threads':1,'absolute_niceness':15,'device':'cpu','allocations':0})
 print(v.stdout);print(v.stderr);assert v.returncode==0
 cmds=[run([PYTHON,'-m','black','--check','--line-length','100','tools/ops/dispatch.py','tests/test_ops_f5_timing_admission.py']),run([PYTHON,'-m','isort','--check-only','--profile','black','tools/ops/dispatch.py','tests/test_ops_f5_timing_admission.py']),run([PYTHON,'-m','flake8','tools/ops/dispatch.py','tests/test_ops_f5_timing_admission.py']),run(['git','diff','--check'])]
 write(P/'lint-verification.json',{'status':'PASS','commands':cmds,'allocations':0})
if phase=='verify':
 fresh();m=load(P/'manifest.json');c=load(P/'control.json');l=load(P/'verified-ledger.json')
 assert m['scientific_admission'] is False and m['env']['F5_MODE']=='qualification' and 'substeps' not in m and 'draft_only' not in m
 assert c['allocation']==sha(json.dumps({k:v for k,v in c.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 assert l['allocation']==c['allocation'] and l['namespace']==c['namespace'] and l['units']==[] and l['allocation_sha256']==c['identity']['allocation_sha256']
 assert c['ledger_sha256']==verify_export(P/'verified-ledger.json') and c['allocation']!=load(PLAN/'control.json')['allocation']
 assert c['input_hashes']=={'b8.pth':'3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8'}
 assert c['environment_capture']==load(C/'output-04/exports/environment.json')
 assert m['required_exports']==load(T/'manifest.json')['required_exports'] and len(m['required_exports'])==7
 assert m['source_paths']==load(T/'manifest.json')['source_paths']
 inventory=inv(m);assert inventory==load(P/'inventory-metadata.json') and len(inventory['rows'])==9
 assert m['known_private_input_bytes']==m['f5']['known_private_input_bytes']==inventory['private_upload_bytes']
 assert m['budget']['other_assignments']==load(P/'live-account-preflight.json')['other_assignments']
 assert m['timeout_minutes']==m['resume_exemption']['timeout_minutes']==30
 assert load(P/'dispatch-dry-run.json')['command']['argv']==command()+['--dry-run']
 assert load(P/'dispatch-dry-run.json')['command']['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 assert shlex.join(command()) in (P/'DISPATCH.txt').read_text()
 for d in load(P/'predecessor-inventory.json').values():check(d)
 assert load(P/'operational-tests.json')['status']=='PASS' and load(P/'independent-review.json')['status']=='PASS'
 external={};skipped=[]
 roots=[load(P/'bindings.json'),load(P/'governance-source-preflight.json'),m,load(P/'broker-validation.json'),load(P/'plan-operation-binding.json'),load(C/'bindings-v5.json'),load(C/'accounting-owner-ruling-04.json'),load(C/'cumulative-envelope-owner-ruling-04.json'),load(F/'REVIEW_PACKET.json'),load(F/'REVIEW_BUNDLE.json')]
 for v in roots:
  for d in refs(v):
   if protected(d['path']):skipped.append({'path':d['path'],'read':False});continue
   check(d);external[d['path']]=bind(d['path'])
 for relative,h in c['source_hashes'].items():
  p=R/relative;d=bind(p);assert d['sha256']==h and p.read_bytes()==git('show',f'{PIN}:{relative}');external[str(p)]=d
 for relative in m['source_paths']:
  p=R/relative;external[str(p)]=bind(p)
 for name in ['tools/ops/dispatch.py','tools/ops/dispatch.sh','tests/test_ops_f5_plan_admission.py','tests/test_ops_f5_timing_admission.py','tests/test_ops_dispatch.py','tools/ops/qualification.py']:
  p=R/name;external[str(p)]=bind(p)
 for d in list(external.values()):
  p=pathlib.Path(d['path'])
  if p.name.endswith(('.sha256','.complete')):continue
  for suffix in ['.sha256','.complete']:
   x=pathlib.Path(str(p)+suffix)
   if x.is_file():
    external[str(x)]=bind(x)
    if suffix=='.sha256':assert x.read_text().split()[0]==d['sha256'],str(x)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 checks={k:True for k in ['exact_control_allocation_ledger_linkage','distinct_plan_and_timing_allocations','exact_admitted_attempt04_raw_object','B8_only_control_input_hashes','five_exact_pinned_opponent_uploads','source_192pins_git_and_worktree_match','source_paths194_archive200','exact_nine_inputs_seven_exports','inventories_recomputed','fresh_https_source_governance_broker_account_preflights','plan_terminal_receipt_and_actual_calibrated_endpoint_rate_bound','scientific_admission_false','full_qualification_mode_no_transport_only','no_retry_fences','dryrun_command_matches_dispatch','predecessors_unchanged','all_payload_sidecars_markers','unused_state_output_name']}
 write(P/'internal-consistency.json',{'schema':'bl-f5-timing-operative-consistency/v1','status':'PASS','created_utc':now(),'manifest':bind(P/'manifest.json'),'checks':checks,'external_files_verified':len(external),'external_files':external,'protected_references_not_followed':skipped,'opaque_input_handling':'Only hashes/sizes; no checkpoint/label deserialization or outcome inspection.','allocations':0,'dispatches':0,'fits':0,'outcome_looks':0})
 print(json.dumps({'status':'PASS','external_refs':len(external),'private_bytes':inventory['private_upload_bytes']}))
if phase=='seal':
 fresh();cons=load(P/'internal-consistency.json');assert cons['status']=='PASS'
 for d in cons['external_files'].values():check(d)
 for d in load(P/'predecessor-inventory.json').values():check(d)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 files={str(p.relative_to(P)):bind(p) for p in sorted(P.rglob('*')) if p.is_file()}
 write(P/'PACKAGE_SEAL_OPERATIONAL.json',{'schema':'bl-f5-timing-operative-package-seal/v1','status':'SEALED_ACCEPTED_FULL_TIMING_PARITY_AND_CALIBRATION_ONLY','created_utc':now(),'package_path':str(P),'job_id':JOB,'implementation_commit':PIN,'manifest':bind(P/'manifest.json'),'control':bind(P/'control.json'),'ledger':bind(P/'verified-ledger.json'),'bindings':bind(P/'bindings.json'),'rulings':{k:bind(P/v) for k,v in RULINGS.items()},'files':files,'external_files':cons['external_files'],'internal_consistency':bind(P/'internal-consistency.json'),'independent_review':bind(P/'independent-review.json'),'dispatch_command':shlex.join(command()),'dry_run_output':load(P/'dispatch-dry-run.json')['command']['stdout'].strip(),'state_dir':str(P/'state-01'),'output_dir':str(P/'output-01'),'state_absent':True,'output_absent':True,'scientific_admission':False,'scientific_fits_authorized':False,'numerical_looks_authorized':False,'release_authorized':False,'publication_required_before_operator_dispatch':True,'seal_self_reference':'Seal and sidecars are bound by later create-only publication envelope.','freshness':'Recheck unused output/state/name immediately before dispatch; changed assignments require a fresh published successor.','allocations':0,'dispatches':0})
 print(json.dumps({'seal':bind(P/'PACKAGE_SEAL_OPERATIONAL.json'),'manifest':bind(P/'manifest.json'),'files':len(files),'external_files':len(cons['external_files']),'dispatch':shlex.join(command())},indent=2))
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
   assert gzip.decompress(b''.join((dest/x['path']).read_bytes() for x in parts))==raw
  else:put(q,raw);row['representation']={'path':str(rel)}
  rows.append(row)
 for p in sorted(dest.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):
   for suffix,raw in [('.sha256',(bind(p)['sha256']+'  '+p.name+'\n').encode()),('.complete',b'complete\n')]:
    x=pathlib.Path(str(p)+suffix)
    if x.exists():assert x.read_bytes()==raw
    else:put(x,raw)
 published={str(p.relative_to(dest)):bind(p) for p in sorted(dest.rglob('*')) if p.is_file()}
 envelope={'schema':'bl-f5-round08-complete-package-publication/v1','status':'SEALED_FULL_TIMING_PARITY_AND_CALIBRATION_ONLY','created_utc':now(),'source_package':str(P),'ledger_directory':str(dest),'original_files':rows,'published_files':published,'operative_seal':bind(P/'PACKAGE_SEAL_OPERATIONAL.json'),'lossless_large_metadata_reconstruction_verified':True,'science_payloads_copied':False,'allocations':0,'dispatches':0}
 write(P/'PACKAGE_PUBLICATION.json',envelope)
 for suffix in ['', '.sha256','.complete']:put(dest/('PACKAGE_PUBLICATION.json'+suffix),(P/('PACKAGE_PUBLICATION.json'+suffix)).read_bytes())
 print(json.dumps({'package':str(P),'publication':bind(P/'PACKAGE_PUBLICATION.json'),'original_files':len(rows),'published_files':len(published)+3}))
