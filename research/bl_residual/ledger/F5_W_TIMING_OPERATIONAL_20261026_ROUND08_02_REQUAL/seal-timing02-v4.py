"""Verify, seal and publish timing02; never allocate or dispatch."""
import pathlib, sys, json, hashlib, gzip, tarfile, shlex
P=pathlib.Path(__file__).parent
exec((P/'operative-builder-v6.py').read_text().split("phase=sys.argv[1]")[0])
phase=sys.argv[1]
def verify_evidence(path):
 if path.name in ['operational-tests-03.log','operational-tests-03.json']:
  assert pathlib.Path(str(path)+'.sha256').read_text().split()==[bind(path)['sha256']]
  assert pathlib.Path(str(path)+'.complete').read_text()=='complete\n'
  return bind(path)['sha256']
 return verify_export(path)
def operative(): return load(P/'OPERATIVE_FILES.json')
def final_command(): return shlex.split(operative()['dispatch_command'])
if phase=='document':
 fresh()
 h=load(P/'historical-accounting.json');r=load(P/'plan-terminal-receipt-snapshot.json')
 debit=r['usage_before']['balance']-r['usage_after']['balance'];rate=r['accounting']['observed_rate_units_per_hour']
 write(P/'cumulative-accounting-final.json',{'schema':'bl-f5-timing02-cumulative-accounting/v1','historical_seven_sessions':bind(P/'historical-accounting.json'),'plan_successor_terminal':bind(P/'plan-terminal-receipt-snapshot.json'),'plan_elapsed_seconds':r['elapsed_seconds'],'plan_observed_balance_decrease_units':debit,'plan_raw_accounting':r['accounting'],'plan_usage_before':r['usage_before'],'plan_usage_after':r['usage_after'],'cumulative_session_elapsed_seconds':h['cumulative_session_elapsed_seconds']+r['elapsed_seconds'],'sum_observed_balance_decreases_units':h['sum_observed_balance_decreases_units']+debit,'sum_full_session_observed_rate_exposure_units':h['sum_full_session_observed_rate_exposure_units']+r['elapsed_seconds']*rate/3600,'sum_full_session_priced_exposure_units':h['sum_full_session_priced_exposure_units']+r['elapsed_seconds']*r['accounting']['priced_rate_units_per_hour']/3600,'accounting_interpretation':h['accounting_interpretation'],'timing_attempt02_executed':False,'timing_attempt02_elapsed_seconds':0,'timing_attempt02_debit_units':0,'budgets_are_runaway_guards_only':True})
 write(P/'README.md',f'''# F5 W timing attempt02 requalified successor

Prepared only. Independent concrete review and publication precede the operator's single manual dispatch. No timing runtime was allocated by this preparation. The operative files are manifest-02.json and DISPATCH-02.txt, identified by OPERATIVE_FILES.json. Original manifest.json and DISPATCH.txt remain preserved historical active-map predecessors: the plan independently released during preparation, so the post-release successor uses fresh zero-assignment evidence and the actual plan terminal PASS/stop_verified receipt. No old seal was changed.

Source {PIN}, unchanged F5 implementation {IMPL}, qualified CLI tree {TREE}; clean detached broker {R} at {BROKER}. Full twelve-workload timing/parity and timing-only optimizer/resume calibration, explicit pending-Mac status; no common-plan data dependency. Five pinned opponent uploads, frozen fixtures, own control allocation/empty ledger and pinned HTTPS B8. No scientific fit, numerical look, serving or release authority. All four retry fences retained.

Base15/effective30-minute runaway timeout, forecast15 minutes; numeric threads1, deterministic fp32, no compile, L4>=6vCPU/cuda:0. Technical unit placeholders9999 and reserve0 implement owner no-budget ruling; scientific caps, transfer150s/staging300s, private45397643B/export2GiB unchanged. Historical seven sessions and both mechanical failures retained; cumulative-accounting-final.json adds the independently ended plan successor without resetting exposure/debit. Observed balance changes are distinct from integrated exposure and final billing.

Partial builders, recovery records and prior preflights remain immutable. operative-builder-v6.py supersedes failed preparation helpers only. The CLI's unknown [?] session label was reconciled via exact endpoint, durable broker record and immutable calibration. That mutable active record disappeared on independent release; its historical observation is retained, while final account admission binds the terminal receipt and fresh absence. No live plan state was changed. Hash/size-only handling was used for scientific inputs.

Immediately before dispatch recheck source/broker publication, registry, seals, unused state/output/job, fresh complete endpoint/rate map under the broker lock, maximum3 assignments and serialized staging. A changed sealed endpoint/rate map requires another reviewed create-only successor. Mac companion timing must be uncontended. Fits require both verified output branches, Mac evidence, current-source full-inventory canary and separate ACCEPT_F5_FIT. No retry follows a refusal automatically.
''')
 print('DOCUMENTED')
if phase=='verify':
 fresh();o=operative();m=load(o['manifest']['path']);c=load(P/'control.json');l=load(P/'verified-ledger.json');old=load(T/'control.json');om=load(T/'manifest.json')
 for d in o.values():
  if isinstance(d,dict) and 'sha256' in d:check(d)
 assert c['allocation']==sha(json.dumps({k:v for k,v in c.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 assert l['allocation']==c['allocation']!=old['allocation'] and l['units']==[] and l['namespace']==c['namespace']==old['namespace'] and l['allocation_sha256']==c['identity']['allocation_sha256']
 assert c['ledger_sha256']==verify_export(P/'verified-ledger.json')
 assert c['environment_capture']==load(P/'capture05-environment.json')
 for k in ['namespace','input_hashes','qualification_b8_sha256','qualification_checkpoint_pool','qualification_fixture_sha256','environment_policy']:
  assert c[k]==old[k],k
 for k in c['identity']:
  if k!='source_sha256':assert c['identity'][k]==old['identity'][k],k
 assert c['input_hashes']=={'b8.pth':'3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8'}
 for k in ['script','env','requirements','timeout_minutes','lease_seconds','max_export_bytes','source_paths','required_exports','device','canary','scientific_admission']:
  assert m[k]==om[k],k
 assert m['commit']==PIN and m['timeout_minutes']==m['resume_exemption']['timeout_minutes']==30
 assert m['scientific_admission'] is False and m['budget']['other_assignments']=={}
 for k in ['no_automatic_retry','no_allocation_retry','no_job_retry','no_completion_retry']:assert m['timing_calibration'][k] is True
 inventory=inv(m);assert inventory==load(P/'inventory-metadata.json');assert len(m['inputs'])==9 and len(m['required_exports'])==7
 assert m['known_private_input_bytes']==m['f5']['known_private_input_bytes']==inventory['private_upload_bytes']
 for k in ['transfer_bar_seconds','staging_bar_seconds','private_upload_cap_bytes','private_upload_bytes_max','base_timeout_minutes','effective_timeout_minutes','automatic_retries']:assert m['f5'][k]==om['f5'][k],k
 assert git('rev-parse',f'{PIN}:research/colab_cli').decode().strip()==TREE
 source=load(P/'source-preflight-artifacts/source-plan.json');assert source['commit']==PIN
 for rel,h in c['source_hashes'].items():
  assert bind(R/rel)['sha256']==h==sha(git('show',f'{PIN}:{rel}'))
  if not rel.startswith('research/colab_cli/'):assert git('show',f'{IMPL}:{rel}')==git('show',f'{PIN}:{rel}'),rel
 with tarfile.open(P/'source-preflight-artifacts/source.tar.gz') as a:
  assert sorted(x.name for x in a.getmembers())==sorted(source['files'])
  for x in a.getmembers():assert sha(a.extractfile(x).read())==source['files'][x.name]['sha256']
 assert shlex.join(final_command()) in (P/'DISPATCH-02.txt').read_text()
 assert load(P/'dispatch-dry-run-02.json')['command']['argv']==final_command()+['--dry-run']
 assert load(P/'dispatch-dry-run-02.json')['command']['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 for d in load(P/'predecessor-inventory.json').values():check(d)
 tests=load(P/'operational-tests-03.json');assert tests['requested_f5_scope']['status']=='PASS' and tests['status']=='PASS_WITH_RECORDED_PREEXISTING_FAILURES'
 # Rehash every referenced immutable file; mutable historical account observations are explicitly recorded.
 external={};observation_only=[];prohibited=[]
 for f in sorted(P.glob('*.json')):
  for d in refs(load(f)):
   path=pathlib.Path(d['path'])
   if protected(path):prohibited.append({'path':str(path),'read':False});continue
   if str(path).startswith('/Users/josenunez/.config/colab-cli/snake/active-'):
    observation_only.append({'reference':d,'status':'HISTORICAL_ACTIVE_RECORD_REMOVED_ON_VERIFIED_RELEASE','read_now':False,'superseded_by':bind(P/'live-account-preflight-02.json')});continue
   check(d);external[str(path)]=bind(path)
 for rel in source['files']:
  path=R/rel;assert path.read_bytes()==git('show',f'{PIN}:{rel}');external[str(path)]=bind(path)
 for rel in ['tools/ops/dispatch.py','tools/ops/dispatch.sh','tools/ops/qualification.py']:
  path=R/rel;external[str(path)]=bind(path)
 for d in list(external.values()):
  path=pathlib.Path(d['path'])
  if path.name.endswith(('.sha256','.complete')):continue
  for suffix in ['.sha256','.complete']:
   part=pathlib.Path(str(path)+suffix)
   if part.is_file():
    external[str(part)]=bind(part)
    if suffix=='.sha256':assert part.read_text().split()[0]==d['sha256']
 for path in sorted(P.rglob('*')):
  if path.is_file() and not path.name.endswith(('.sha256','.complete')):verify_evidence(path)
 calls=[run(['/Users/josenunez/.local/bin/colab','--auth','oauth2',x]) for x in ['sessions','usage','sessions']]
 assert all('No active sessions found on server.' in calls[i]['stdout'] for i in [0,2])
 assert 'Active assignments: 0' in calls[1]['stdout'] and 'Usage rate: 0.00/hr' in calls[1]['stdout']
 write(P/'account-recheck-before-seal-02.json',{'status':'PASS','created_utc':now(),'calls':calls,'sealed_other_assignments':{},'unchanged_since_successor_preflight':True,'allocations':0})
 write(P/'internal-consistency-02.json',{'schema':'bl-f5-timing02-consistency/v1','status':'PASS','created_utc':now(),'operative_files':bind(P/'OPERATIVE_FILES.json'),'source_pin_count':len(c['source_hashes']),'archive_members':len(source['files']),'inputs':9,'required_exports':7,'private_bytes':inventory['private_upload_bytes'],'https_bytes':inventory['https_bytes'],'predecessor_files_unchanged':len(load(P/'predecessor-inventory.json')),'all_local_payload_sidecars_markers_verified':True,'runtime_f5_bytes_unchanged_from_accepted_implementation':True,'control_allocation_and_empty_ledger_verified':True,'all_retry_and_scientific_authority_fences_retained':True,'external_files':external,'historical_observation_only_references':observation_only,'prohibited_references_not_followed':prohibited,'allocations':0,'dispatches':0,'fits':0,'outcome_looks':0})
 print('CONSISTENCY_PASS',len(external))
if phase=='seal':
 fresh();consistency=load(P/'internal-consistency-02.json');assert consistency['status']=='PASS';assert load(P/'independent-review.json')['status']=='PASS'
 for d in consistency['external_files'].values():check(d)
 for path in sorted(P.rglob('*')):
  if path.is_file() and not path.name.endswith(('.sha256','.complete')):verify_evidence(path)
 files={str(path.relative_to(P)):bind(path) for path in sorted(P.rglob('*')) if path.is_file()}
 write(P/'PACKAGE_SEAL_OPERATIONAL.json',{'schema':'bl-f5-timing-operative-package-seal/v1','status':'SEALED_FULL_TIMING_PARITY_CALIBRATION_ONLY_PENDING_MANUAL_REVIEWED_DISPATCH','created_utc':now(),'package_path':str(P),'job_id':JOB,'source_commit':PIN,'broker_commit':BROKER,'qualified_cli_tree':TREE,'operative_files':operative(),'manifest':bind(P/'manifest-02.json'),'control':bind(P/'control.json'),'ledger':bind(P/'verified-ledger.json'),'cumulative_accounting':bind(P/'cumulative-accounting-final.json'),'plan_operation':bind(P/'plan-operation-binding-02.json'),'independent_review':bind(P/'independent-review.json'),'internal_consistency':bind(P/'internal-consistency-02.json'),'files':files,'external_files':consistency['external_files'],'historical_observation_only_references':consistency['historical_observation_only_references'],'dispatch_command':shlex.join(final_command()),'dry_run_output':load(P/'dispatch-dry-run-02.json')['command']['stdout'].strip(),'self_exclusions':['PACKAGE_SEAL_OPERATIONAL.json and companions','PACKAGE_PUBLICATION.json and companions'],'state_absent':True,'output_absent':True,'scientific_admission':False,'scientific_fits_authorized':False,'numerical_looks_authorized':False,'release_authorized':False,'allocations':0,'dispatches':0})
 print('SEALED',bind(P/'PACKAGE_SEAL_OPERATIONAL.json'))
if phase=='publish':
 fresh();seal=load(P/'PACKAGE_SEAL_OPERATIONAL.json')
 for d in list(seal['files'].values())+list(seal['external_files'].values()):check(d)
 dest=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5w/research/bl_residual/ledger')/P.name;assert not dest.exists();dest.mkdir()
 def put(path,raw):
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(raw)
 rows=[]
 for path in sorted(P.rglob('*')):
  if not path.is_file():continue
  rel=path.relative_to(P);raw=path.read_bytes();q=dest/rel;row={'source':bind(path),'relative_path':str(rel)}
  if len(raw)>1900000:
   compressed=gzip.compress(raw,compresslevel=9,mtime=0);parts=[]
   for i,start in enumerate(range(0,len(compressed),1500000)):
    part=pathlib.Path(str(q)+f'.gz.part-{i:03d}');payload=compressed[start:start+1500000];write(part,payload);parts.append({'path':str(part.relative_to(dest)),'sha256':sha(payload),'size':len(payload)})
   index=pathlib.Path(str(q)+'.COMPRESSED_INDEX.json');write(index,{'original':bind(path),'encoding':'gzip-concatenated-parts','compressed_sha256':sha(compressed),'parts':parts});row['representation']={'index':str(index.relative_to(dest))}
   assert gzip.decompress(b''.join((dest[x['path']]).read_bytes() for x in parts))==raw
  else:put(q,raw);row['representation']={'path':str(rel)}
  rows.append(row)
 published={str(path.relative_to(dest)):bind(path) for path in sorted(dest.rglob('*')) if path.is_file()}
 write(P/'PACKAGE_PUBLICATION.json',{'schema':'bl-f5-round08-complete-package-publication/v1','status':'SEALED_TIMING_ATTEMPT02_ONLY','created_utc':now(),'source_package':str(P),'ledger_directory':str(dest),'original_files':rows,'published_files':published,'lossless_large_metadata_reconstruction_verified':True,'science_payloads_copied':False,'operative_seal':bind(P/'PACKAGE_SEAL_OPERATIONAL.json'),'dispatch_command':shlex.join(final_command()),'dispatches':0})
 for suffix in ['', '.sha256','.complete']:put(dest/('PACKAGE_PUBLICATION.json'+suffix),(P/('PACKAGE_PUBLICATION.json'+suffix)).read_bytes())
 print('PUBLISHED_REPRESENTATION',len(rows),len(published)+3,bind(P/'PACKAGE_PUBLICATION.json'))
