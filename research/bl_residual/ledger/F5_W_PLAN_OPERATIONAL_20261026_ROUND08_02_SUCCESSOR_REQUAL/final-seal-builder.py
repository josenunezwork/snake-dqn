"""Freeze the independently reviewed successor and all create-only preparation evidence."""
import pathlib
P=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_W_PLAN_OPERATIONAL_20261026_ROUND08_02_SUCCESSOR_REQUAL')
exec((P/'operative-builder-v2.py').read_text().split('phase=sys.argv[1]')[0])
fresh();v=load(P/'operational-verification-v2.json');review=load(P/'independent-review.json');assert review['status']=='NO_BLOCKING_FINDINGS'
erratum=P/'MANIFEST_ANNOTATION_ERRATUM.json';assert bind(erratum)['sha256']=='ff9caafe2cd4bc732a5dbaa6d18b1b98cef691e08ee4c48b00c8eb17e00e2cf4'
for d in v['external_files'].values():check(d)
for name in ['held-plan02-inventory.json','predecessor-inventory.json']:
 for d in load(P/name).values():check(d)
for value in [load(erratum),load(P/'test-verification.json'),review]:
 for d in refs(value):check(d)
write(P/'final-seal-builder.py',pathlib.Path(__file__).read_bytes())
for p in sorted(P.rglob('*')):
 if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
files={str(p.relative_to(P)):bind(p) for p in sorted(P.rglob('*')) if p.is_file()}
seal={'schema':'bl-f5-plan-successor-operative-seal/v3','status':'SEALED_ACCEPTED_PLAN02_SUCCESSOR_AFTER_TRANSPORT_REQUALIFICATION','created_utc':now(),'package_path':str(P),'source_commit':PIN,'broker_commit':BROKER,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'job_id':JOB,'manifest':bind(P/'manifest.json'),'manifest_annotation_erratum':bind(erratum),'control':bind(P/'control.json'),'ledger':bind(P/'verified-ledger.json'),'bindings':bind(P/'bindings.json'),'verification':bind(P/'operational-verification-v2.json'),'independent_review':bind(P/'independent-review.json'),'tests':bind(P/'test-verification.json'),'files':files,'external_files':v['external_files'],'dispatch_command':shlex.join(command()),'dry_run_output':load(P/'dispatch-dry-run.json')['command']['stdout'].strip(),'state_dir':str(P/'state-01'),'output_dir':str(P/'output-01'),'state_absent':True,'output_absent':True,'scientific_admission':False,'timing_sealed':False,'colab_calls':0,'dispatches':0,'science_payloads_opened':0,'held_plan02_unchanged':True,'account_evidence':'Saved postrelease only; actual wrapper must refresh under lock and reject drift.','nonoperative_provenance':['build-successor.py','build-package.py','seal-successor.py','PREPARATION_RECOVERY_01.md'],'self_reference_exclusion':'Seal and companions are bound by subsequent create-only PACKAGE_PUBLICATION.json; publication envelope is the only subsequent package addition.'}
write(P/'PACKAGE_SEAL_OPERATIONAL.json',seal);print(json.dumps({'seal':bind(P/'PACKAGE_SEAL_OPERATIONAL.json'),'files':len(files),'manifest':bind(P/'manifest.json'),'control':bind(P/'control.json'),'ledger':bind(P/'verified-ledger.json')}))
