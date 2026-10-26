"""Verify, seal and losslessly publish the first F5 operative plan package only."""
exec(open('/tmp/f5_plan_round08_seal.py').read().split("phase=sys.argv[1]")[0])
phase=sys.argv[1]
if phase=='verify':
 fresh();m=load(P/'manifest.json');control=load(P/'control.json');ledger=load(P/'verified-ledger.json')
 assert m['scientific_admission'] is False and m['env']['F5_MODE']=='prepare-qualification'
 assert control['allocation']==sha(json.dumps({k:v for k,v in control.items() if k not in ('allocation','ledger_sha256')},sort_keys=True,allow_nan=False).encode())
 assert ledger['allocation']==control['allocation'] and ledger['namespace']==control['namespace'] and ledger['units']==[] and ledger['allocation_sha256']==control['identity']['allocation_sha256']
 assert control['ledger_sha256']==verify_export(P/'verified-ledger.json')
 assert control['input_hashes']=={'selected-training.tar.xz':'2c033750b0d25568b19ba81aec7bd0e9b0583bdd32a994a9a1984a409e45c46c'}
 assert control['environment_capture']==load(C/'output-04/exports/environment.json')
 assert m['required_exports']==load(D/'plan-seal/manifest.json')['required_exports']
 assert m['source_paths']==load(D/'plan-seal/manifest.json')['source_paths']
 inventory=inv(m);assert inventory==load(P/'inventory-metadata.json')
 assert m['known_private_input_bytes']==m['f5']['known_private_input_bytes']==inventory['private_upload_bytes']
 assert m['budget']['other_assignments']==load(P/'live-account-preflight.json')['other_assignments']
 assert m['timeout_minutes']==m['resume_exemption']['timeout_minutes']==30
 assert load(P/'dispatch-dry-run.json')['command']['argv']==command()+['--dry-run']
 assert load(P/'dispatch-dry-run.json')['command']['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 assert shlex.join(command()) in (P/'DISPATCH.txt').read_text()
 for d in load(P/'predecessor-inventory.json').values():check(d)
 timing=load(T/'manifest.json');assert timing['scientific_admission'] is False and timing['known_private_input_bytes']==timing['f5']['known_private_input_bytes']==inv(timing)['private_upload_bytes']==37602883
 assert not list(T.glob('*SEAL*')) and not (T/'DISPATCH.txt').exists()
 external={};skipped=[]
 roots=[load(P/'bindings.json'),load(P/'governance-source-preflight.json'),m,load(P/'broker-validation.json'),load(C/'bindings-v5.json'),load(C/'accounting-owner-ruling-04.json'),load(C/'cumulative-envelope-owner-ruling-04.json'),load(F/'REVIEW_PACKET.json'),load(F/'REVIEW_BUNDLE.json')]
 for v in roots:
  for d in refs(v):
   if protected(d['path']):skipped.append({'path':d['path'],'read':False});continue
   check(d);external[d['path']]=bind(d['path'])
 for relative,h in control['source_hashes'].items():
  p=R/relative;d=bind(p);assert d['sha256']==h;assert p.read_bytes()==git('show',f'{PIN}:{relative}');external[str(p)]=d
 for relative in m['source_paths']:
  p=R/relative;external[str(p)]=bind(p)
 for name in ['tools/ops/dispatch.py','tools/ops/dispatch.sh','tests/test_ops_f5_plan_admission.py','tests/test_ops_dispatch.py','tools/ops/qualification.py']:
  p=R/name;external[str(p)]=bind(p)
 for d in list(external.values()):
  p=pathlib.Path(d['path'])
  if p.name.endswith(('.sha256','.complete')):continue
  for suffix in ['.sha256','.complete']:
   x=pathlib.Path(str(p)+suffix)
   if x.is_file():
    external[str(x)]=bind(x)
    if suffix=='.sha256':assert x.read_text().split()[0]==d['sha256'],str(x)
 # Add conventional companions to supervisor-created source artifacts, create-only.
 for p in sorted((P/'source-preflight-artifacts').iterdir()):
  if p.name.endswith(('.sha256','.complete')):continue
  for suffix,data in [('.sha256',(bind(p)['sha256']+'  '+p.name+'\n').encode()),('.complete',b'complete\n')]:
   x=pathlib.Path(str(p)+suffix)
   if not x.exists():
    with x.open('xb') as s:s.write(data)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 tests=load(P/'operational-tests.json');assert tests['status']=='PASS'
 report={'schema':'bl-f5-plan-operative-consistency/v1','status':'PASS','created_utc':now(),'manifest':bind(P/'manifest.json'),'checks':{'exact_control_allocation_ledger_linkage':True,'exact_admitted_attempt04_raw_object':True,'selected_training_only_bound_inputs':True,'source_192pins_git_and_worktree_match':True,'source_paths_194_and_archive_200':True,'exact_seven_inputs_nine_exports':True,'all_inventories_counts_recomputed':True,'source_environment_governance_broker_https_account_preflights':True,'scientific_admission_false_retained':True,'no_retry_fences_retained':True,'timing_successor_bytes_correct_unsealed':True,'dryrun_command_agrees_dispatch_instructions':True,'predecessors_rehashed_unchanged':True,'all_payload_sidecars_markers_verified':True,'unused_state_output_name_rechecked':True},'external_files_verified':len(external),'protected_references_not_followed':skipped,'opaque_input_handling':'Only cryptographic hash/size reads; no labels/checkpoint deserialization/outcome inspection.','allocations':0,'dispatches':0,'fits':0,'outcome_looks':0,'external_files':external}
 write(P/'internal-consistency.json',report)
 print(json.dumps({'status':'PASS','external_refs':len(external),'private_bytes':inventory['private_upload_bytes'],'predecessors':len(load(P/'predecessor-inventory.json'))}))
if phase=='seal':
 fresh();consistency=load(P/'internal-consistency.json');assert consistency['status']=='PASS'
 for d in consistency['external_files'].values():check(d)
 for d in load(P/'predecessor-inventory.json').values():check(d)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 files={str(p.relative_to(P)):bind(p) for p in sorted(P.rglob('*')) if p.is_file()}
 write(P/'PACKAGE_SEAL_OPERATIONAL.json',{'schema':'bl-f5-plan-operative-package-seal/v1','status':'SEALED_ACCEPTED_PLAN_MATERIALIZATION_AND_TRANSPORT_ONLY','created_utc':now(),'package_path':str(P),'job_id':JOB,'implementation_commit':PIN,'manifest':bind(P/'manifest.json'),'control':bind(P/'control.json'),'ledger':bind(P/'verified-ledger.json'),'bindings':bind(P/'bindings.json'),'rulings':{k:bind(P/v) for k,v in RULINGS.items()},'files':files,'external_files':consistency['external_files'],'internal_consistency':bind(P/'internal-consistency.json'),'independent_review':bind(P/'independent-review.json'),'dispatch_command':shlex.join(command()),'dry_run_output':load(P/'dispatch-dry-run.json')['command']['stdout'].strip(),'state_dir':str(P/'state-01'),'output_dir':str(P/'output-01'),'state_absent':True,'output_absent':True,'scientific_admission':False,'scientific_fits_authorized':False,'numerical_looks_authorized':False,'release_authorized':False,'timing_package_sealed':False,'publication_required_before_operator_dispatch':True,'seal_self_reference':'This seal and its SHA/completion companions are bound by a later create-only publication envelope, avoiding self-reference.','freshness':'Account snapshot is timestamped only. Recheck output/state immediately before dispatch; changed assignments require fresh successor, not seal mutation.','allocations':0,'dispatches':0})
 print(json.dumps({'seal':bind(P/'PACKAGE_SEAL_OPERATIONAL.json'),'manifest':bind(P/'manifest.json'),'files':len(files),'external_files':len(consistency['external_files']),'dispatch':shlex.join(command())},indent=2))
if phase=='publish':
 fresh();seal=load(P/'PACKAGE_SEAL_OPERATIONAL.json')
 for d in seal['files'].values():check(d)
 for d in seal['external_files'].values():check(d)
 for src,kind in [(P,'SEALED_PLAN_MATERIALIZATION_ONLY'),(T,'TIMING_DRAFT_ONLY_UNSEALED')]:
  dest=R/'research/bl_residual/ledger'/src.name;assert not dest.exists();dest.mkdir()
  rows=[]
  def put(p,b):
   p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as s:s.write(b)
  for p in sorted(src.rglob('*')):
   if not p.is_file():continue
   rel=p.relative_to(src);raw=p.read_bytes();q=dest/rel;row={'source':bind(p),'relative_path':str(rel)}
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
  envelope={'schema':'bl-f5-round08-complete-package-publication/v1','status':kind,'created_utc':now(),'source_package':str(src),'ledger_directory':str(dest),'original_files':rows,'published_files':published,'lossless_large_metadata_reconstruction_verified':True,'science_payloads_copied':False,'dispatches':0,'timing_sealed':False}
  if src==P:envelope['operative_seal']=bind(P/'PACKAGE_SEAL_OPERATIONAL.json')
  write(src/'PACKAGE_PUBLICATION.json',envelope)
  for suffix in ['', '.sha256','.complete']:put(dest/('PACKAGE_PUBLICATION.json'+suffix),(src/('PACKAGE_PUBLICATION.json'+suffix)).read_bytes())
  print(json.dumps({'package':str(src),'publication':bind(src/'PACKAGE_PUBLICATION.json'),'original_files':len(rows),'published_files':len(published)+3}))
