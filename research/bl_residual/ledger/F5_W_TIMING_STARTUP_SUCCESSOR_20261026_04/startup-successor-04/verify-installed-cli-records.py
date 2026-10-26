"""Offline installed-distribution RECORD integrity; no CLI/provider invocation."""
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

rows=[]
failures=[]
for dist in sorted(metadata.distributions(), key=lambda d:d.metadata['Name'].lower()):
    verified=[]
    unhashed=[]
    for entry in dist.files or []:
        path=Path(dist.locate_file(entry))
        if entry.hash:
            if not path.is_file():
                failures.append({'distribution':dist.metadata['Name'],'path':str(path),'error':'missing'})
                continue
            raw=path.read_bytes()
            digest=hashlib.new(entry.hash.mode,raw).digest()
            expected=base64.urlsafe_b64decode(entry.hash.value+'='*((-len(entry.hash.value))%4))
            if digest!=expected or (entry.size is not None and len(raw)!=entry.size):
                failures.append({'distribution':dist.metadata['Name'],'path':str(path),'error':'RECORD mismatch'})
            verified.append({'relative_path':str(entry),'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)})
        else:
            unhashed.append(str(entry))
    rows.append({'distribution':dist.metadata['Name'],'version':dist.version,'verified_hashed_files':len(verified),'installed_file_inventory_sha256':hashlib.sha256(json.dumps(verified,sort_keys=True).encode()).hexdigest(),'unhashed_record_entries':unhashed,'record_path':str(Path(dist.locate_file(next(x for x in dist.files if str(x).endswith('.dist-info/RECORD')))))})
assert metadata.version('google-colab-cli')=='0.7.4'
keys={}
for name,rel in [('google-colab-cli','colab_cli/runtime.py'),('jupyter_kernel_client','jupyter_kernel_client/manager.py'),('jupyter_kernel_client','jupyter_kernel_client/constants.py')]:
    path=Path(metadata.distribution(name).locate_file(rel))
    keys[rel]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size}
value={'schema':'bl-f5-local-cli-record-integrity/v1','status':'PASS' if not failures else 'FAIL','cli_version':metadata.version('google-colab-cli'),'distributions':rows,'failures':failures,'key_startup_files':keys,'verified_hashed_files':sum(r['verified_hashed_files'] for r in rows),'no_dependency_upgrade':True,'allocation_calls':0,'runtime_calls':0,'scope':'Current installed CLI/dependency bytes verified against installed distribution RECORD SHA/size. Historical acceptance records prove CLI0.7.4 and repository transport tree; they contain no Mac installed-dependency snapshot. Do not claim a historical byte-for-byte comparison that was not recorded.'}
print(json.dumps(value,sort_keys=True))
assert not failures, failures
