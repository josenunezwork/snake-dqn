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
 for name in ['F5_W_PLAN_ATTEMPT02_PUBLICATION_CONTINUATION_20261026_v2.py','F5_W_PLAN_ATTEMPT02_PUBLICATION_NOTE_20261026_v2.json']:
  for suffix in ['', '.sha256','.complete']:put(dest/('publication-tooling/'+name+suffix),(B/(name+suffix)).read_bytes())
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
phase='publish'
if phase=='publish':
 fresh();seal=load(P/'PACKAGE_SEAL_OPERATIONAL.json')
 for d in seal['files'].values():check(d)
 for d in seal['external_files'].values():check(d)
 dest=R/'research/bl_residual/ledger'/P.name;dest.mkdir(exist_ok=True)
 original_write=write
 def write(p,v):
  if p.exists():
   expected=v if isinstance(v,bytes) else v.encode() if isinstance(v,str) else (json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()
   assert p.read_bytes()==expected,str(p);verify_export(p)
  else:original_write(p,v)
 def put(p,b):
  p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==b,str(p)
  else:
   with p.open('xb') as stream:stream.write(b)
 for name in ['F5_W_PLAN_ATTEMPT02_PUBLICATION_CONTINUATION_20261026_v2.py','F5_W_PLAN_ATTEMPT02_PUBLICATION_NOTE_20261026_v2.json']:
  for suffix in ['', '.sha256','.complete']:put(dest/('publication-tooling/'+name+suffix),(B/(name+suffix)).read_bytes())
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
 envelope={'schema':'bl-f5-round08-complete-package-publication/v1','status':'SEALED_PLAN_ATTEMPT02_MATERIALIZATION_ONLY','created_utc':now(),'source_package':str(P),'ledger_directory':str(dest),'original_files':rows,'published_files':published,'lossless_large_metadata_reconstruction_verified':True,'science_payloads_copied':False,'dispatches':0,'operative_seal':bind(P/'PACKAGE_SEAL_OPERATIONAL.json'),'timing_state_output_modified':False,'publication_continuation':bind(B/'F5_W_PLAN_ATTEMPT02_PUBLICATION_CONTINUATION_20261026_v2.py'),'publication_note':bind(B/'F5_W_PLAN_ATTEMPT02_PUBLICATION_NOTE_20261026_v2.json')}
 write(P/'PACKAGE_PUBLICATION.json',envelope)
 for suffix in ['', '.sha256','.complete']:put(dest/('PACKAGE_PUBLICATION.json'+suffix),(P/('PACKAGE_PUBLICATION.json'+suffix)).read_bytes())
 print(json.dumps({'package':str(P),'publication':bind(P/'PACKAGE_PUBLICATION.json'),'original_files':len(rows),'published_files':len(published)+3}))
