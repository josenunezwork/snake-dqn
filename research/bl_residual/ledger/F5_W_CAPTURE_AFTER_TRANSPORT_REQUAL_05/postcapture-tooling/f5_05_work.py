import pathlib,json,hashlib,gzip,subprocess,datetime
R=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5w')
A=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024')
P=A/'F5_W_CAPTURE_AFTER_TRANSPORT_REQUAL_05'
D=R/'research/bl_residual/ledger'/P.name
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def bind(p):
 p=pathlib.Path(p);return {'path':str(p),'sha256':sha(p.read_bytes()),'size':p.stat().st_size}
def put(p,raw):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
def write(p,v):
 p=pathlib.Path(p);raw=v if isinstance(v,bytes) else (json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()
 put(p,raw);put(str(p)+'.sha256',(sha(raw)+'  '+p.name+'\n').encode());put(str(p)+'.complete',b'complete\n');return bind(p)
def publish(files,dest):
 rows=[]
 for p,rel in files:
  raw=p.read_bytes();q=dest/rel
  if len(raw)>1900000:
   z=gzip.compress(raw,compresslevel=9,mtime=0);parts=[]
   for i,start in enumerate(range(0,len(z),1500000)):
    part=pathlib.Path(str(q)+f'.gz.part-{i:03d}');payload=z[start:start+1500000];write(part,payload);parts.append({'path':str(part.relative_to(dest)),'sha256':sha(payload),'size':len(payload)})
   index=pathlib.Path(str(q)+'.COMPRESSED_INDEX.json');write(index,{'original':bind(p),'encoding':'gzip-concatenated-parts','compressed_sha256':sha(z),'parts':parts});rep={'index':str(index.relative_to(dest))}
   assert gzip.decompress(b''.join((dest/x['path']).read_bytes() for x in parts))==raw
  else:
   put(q,raw);rep={'path':str(rel)}
  rows.append({'source':bind(p),'relative_path':str(rel),'representation':rep})
 for p in sorted(dest.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):
   for suffix,raw in [('.sha256',(sha(p.read_bytes())+'  '+p.name+'\n').encode()),('.complete',b'complete\n')]:
    q=pathlib.Path(str(p)+suffix)
    if q.exists():assert q.read_bytes()==raw
    else:put(q,raw)
 return rows
if __name__=='__main__':
 out=P/'output-05';dest=D/'output-05';assert not dest.exists()
 rows=publish([(p,p.relative_to(out)) for p in sorted(out.rglob('*')) if p.is_file()],dest)
 state=P/'state-05/bl-f5-w-capture-requal-3b32d755-05'
 state_rows=publish([(p,p.relative_to(state)) for p in sorted(state.rglob('*')) if p.is_file()],D/'release-state-05')
 inventory={str(p.relative_to(D)):bind(p) for base in [dest,D/'release-state-05'] for p in sorted(base.rglob('*')) if p.is_file()}
 report={'schema':'bl-f5-capture-output-publication/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':load(out/'receipt.json')['job_id'],'source_files':rows,'saved_release_state':state_rows,'published_files':inventory,'byte_identical_or_losslessly_reconstructed':True,'scope':'Complete saved operational output and release state; post-publication verification follows origin/main publication. No payload deserialization.','dispatches':0,'colab_calls':0,'science_payloads_opened':0}
 write(P/'OUTPUT_PUBLICATION_05.json',report)
 for suffix in ['', '.sha256','.complete']:put(D/('OUTPUT_PUBLICATION_05.json'+suffix),(P/('OUTPUT_PUBLICATION_05.json'+suffix)).read_bytes())
 print(json.dumps({'source_files':len(rows),'published_files':len(inventory),'inventory':bind(P/'OUTPUT_PUBLICATION_05.json')}))
