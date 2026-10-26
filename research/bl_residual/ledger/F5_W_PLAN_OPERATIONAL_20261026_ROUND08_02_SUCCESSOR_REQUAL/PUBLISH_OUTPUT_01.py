"""Create-only opaque publication of the exact F5 plan successor output."""
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/Users/josenunez/Projects/ml/snake-dqn-f5planpub')
BASE = Path('/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024')
NAME = 'F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02_SUCCESSOR_REQUAL'
PACKAGE = BASE / NAME
DEST = ROOT / 'research/bl_residual/ledger' / NAME

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def encode(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def create(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw)

def publish(path, raw):
    create(path, raw)
    create(Path(str(path)+'.sha256'), f'{sha(raw)}  {path.name}\n'.encode())
    create(Path(str(path)+'.complete'), b'complete\n')


def main():
    source = PACKAGE / 'output-01'
    receipt = json.loads((source/'receipt.json').read_bytes())
    assert receipt['status']=='PASS' and receipt['stop_verified'] is True
    assert receipt['cleanup_error'] is None and len(receipt['exports'])==9
    entries=[]
    # Every original file is copied as opaque bytes; checkpoints are never deserialized.
    originals=sorted(x for x in source.rglob('*') if x.is_file())
    for p in originals:
        assert not p.is_symlink()
        raw=p.read_bytes()
        if p.name.endswith('.sha256') or p.name.endswith('.complete'):
            continue
        for suffix in ('.sha256', '.complete'):
            if Path(str(p)+suffix).exists():
                value=Path(str(p)+suffix).read_bytes()
                assert value==(f'{sha(raw)}  {p.name}\n'.encode() if suffix=='.sha256' else b'complete\n')
        rel=p.relative_to(source); q=DEST/'output-01'/rel
        entry={'relative_path':str(rel),'source_path':str(p),'size':len(raw),'sha256':sha(raw)}
        if len(raw)<=1_500_000:
            publish(q,raw); entry['representation']={'path':str(q.relative_to(DEST))}
        else:
            compressed=gzip.compress(raw,mtime=0); parts=[]
            for i,start in enumerate(range(0,len(compressed),1_500_000)):
                payload=compressed[start:start+1_500_000]
                target=Path(str(q)+f'.gz.part-{i:03d}')
                publish(target,payload)
                parts.append({'path':str(target.relative_to(DEST/'output-01')),'size':len(payload),'sha256':sha(payload)})
            index={'original':{'path':str(p),'sha256':sha(raw),'size':len(raw)},'encoding':'gzip-concatenated-parts','compressed_sha256':sha(compressed),'parts':parts}
            publish(Path(str(q)+'.COMPRESSED_INDEX.json'),encode(index))
            # Preserve original sidecars even where the payload has a compressed representation.
            create(Path(str(q)+'.sha256'),f'{sha(raw)}  {q.name}\n'.encode())
            create(Path(str(q)+'.complete'),b'complete\n')
            assert gzip.decompress(b''.join((DEST/'output-01'/v['path']).read_bytes() for v in parts))==raw
            entry['representation']={'index':str(q.relative_to(DEST))+'.COMPRESSED_INDEX.json'}
        entries.append(entry)
    # Saved broker release state, not a new provider call.
    state=PACKAGE/'state-01'/receipt['job_id']
    for name in ['sessions.json','sessions.assignment.json','assignment.json','sessions.json.lock','cli-hook/sitecustomize.py']:
        p=state/name
        publish(DEST/'release-state-01'/name,p.read_bytes())
    acceptance=BASE/'ACCEPT_F5_W_PLAN_SUCCESSOR_REQUAL.md'
    publish(DEST/acceptance.name,acceptance.read_bytes())
    publish(DEST/'PUBLISH_OUTPUT_01.py',Path(__file__).read_bytes())
    report={'schema':'bl-f5-plan-output-publication/v1','created_utc':datetime.now(timezone.utc).isoformat(),'job_id':receipt['job_id'],'source_commit':receipt['commit'],'status':receipt['status'],'stop_verified':receipt['stop_verified'],'elapsed_seconds':receipt['elapsed_seconds'],'all_original_files':len(originals),'payload_files':len(entries),'entries':entries,'colab_calls':0,'dispatches':0,'scientific_fit':False,'payloads_deserialized':False,'opaque_bytes_only':True}
    files=sorted(x for x in DEST.rglob('*') if x.is_file() and ('output-01' in x.parts or 'release-state-01' in x.parts or x.name.startswith(('PUBLISH_OUTPUT_01.py','ACCEPT_F5_W_PLAN_SUCCESSOR_REQUAL.md'))))
    report['published_files']={str(x.relative_to(DEST)):{'size':x.stat().st_size,'sha256':sha(x.read_bytes())} for x in files}
    publish(DEST/'OUTPUT_PUBLICATION_01.json',encode(report))
    print(json.dumps({'payload_files':len(entries),'original_files':len(originals),'published_files':len(files),'elapsed_seconds':receipt['elapsed_seconds'],'receipt_sha256':sha((source/'receipt.json').read_bytes())}))

if __name__=='__main__':
    main()
