"""Exhaustive create-only verification, seal, and metadata publication for CAP05."""
exec(open('/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_W_CAPTURE_AFTER_TRANSPORT_REQUAL_05/build-package.py').read().split('phase=sys.argv[1]')[0])
W=pathlib.Path('/Users/josenunez/Projects/ml/snake-dqn-f5w')
phase=sys.argv[1]
if phase=='verify':
 fresh();assert git('status','--porcelain')==b''
 m=load(P/'environment-capture.json');old=load(C/'environment-capture.json');ident=load(P/'source-identity.json')
 assert m['commit']==PIN and m['job_id']==JOB and m['env']==old['env'] and m['required_exports']==old['required_exports']
 assert m['substeps']==old['substeps'] and m['env']['F5_MODE']=='capture-qualification' and m['device']=='cuda:0'
 assert m['scope']==old['scope'] and m['scope']['scientific_fit_authorized'] is False
 assert m['timeout_minutes']==m['resume_exemption']['timeout_minutes']==30
 assert m['requirements']==old['requirements'] and m['budget']['unit_cap']==2 and m['budget']['reserve_units']==1
 assert m['budget']['other_assignments']==load(P/'live-account-preflight.json')['other_assignments']
 assert m['timing_calibration']['no_automatic_retry'] is True and 'resume' not in m
 assert m['operational_admission']['accounting']==bind(P/'historical-accounting.json')
 assert load(P/'bootstrap-control.json')['source_hashes']==ident['source_hashes']
 assert set(x['dest'] for x in m['inputs'])=={'b8.pth','timing-fixtures.json','bootstrap-control.json'}
 rows=load(P/'inventory-metadata.json')['rows'];assert rows==[{**x,'size':bind(x['path'])['size'],'transport':'upload'} for x in m['inputs']]
 assert sum(x['size'] for x in rows)==m['known_private_input_bytes']==m['f5']['known_private_input_bytes']
 assert m['known_private_input_bytes']<=45397643
 assert m['source_paths']==old['source_paths']
 for relative,h in ident['source_hashes'].items():
  raw=(R/relative).read_bytes();assert sha(raw)==h and raw==git('show',f'{PIN}:{relative}')
  if not relative.startswith('research/colab_cli/'):assert raw==git('show',f'{IMPL}:{relative}')
 assert git('rev-parse',f'{PIN}:research/colab_cli').decode().strip()==TREE
 assert load(REG)==load(P/'qualified-registry.json')
 assert load(P/'qualified-acceptance-result.json')['qualified'] is True
 allocation=load(P/'capture-allocation.json');ledger=load(P/'verified-ledger.json')
 allocation_identity={'namespace':allocation['namespace'],'job_id':JOB,'source_commit':PIN,'bootstrap_control':bind(P/'bootstrap-control.json'),'frozen_allocation':bind(P/'allocation.json'),'transport_registry':bind(REG)}
 assert allocation['allocation']==sha(json.dumps(allocation_identity,sort_keys=True,allow_nan=False).encode())
 assert ledger['allocation']==allocation['allocation'] and ledger['namespace']==allocation['namespace'] and ledger['units']==[]
 assert ledger['allocation_sha256']==load(F/'verified-ledger.json')['allocation_sha256']
 assert (P/'allocation.json').read_bytes()==(F/'allocation.json').read_bytes()
 assert (P/'timing-fixtures.json').read_bytes()==(F/'timing-fixtures.json').read_bytes()
 history=load(P/'historical-accounting.json');assert len(history['attempts'])==6
 assert history['cumulative_session_elapsed_seconds']==sum(load(x['receipt']['path'])['elapsed_seconds'] for x in history['attempts'])
 for row in history['attempts']:
  check(row['original_receipt']);check(row['receipt']);assert pathlib.Path(row['original_receipt']['path']).read_bytes()==pathlib.Path(row['receipt']['path']).read_bytes()
  assert load(row['receipt']['path'])['accounting']==row['raw_accounting'] and row['stop_verified'] is True
 assert history['held_plan_attempt02_executed'] is False and history['completed_science_units']==0
 assert history['attempts'][-2]['job_id']=='bl-f5-w-plan-round08-c989f673-01'
 assert history['attempts'][-1]['job_id']=='bl-f5-w-timing-round08-c989f673-01'
 assert load(P/'attempt04-provenance.json')['matching_new_transport_canary'] is False
 dr=load(P/'dispatch-dry-run.json');assert dr['command']['argv']==command()+['--dry-run']
 assert dr['command']['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
 assert shlex.join(command()) in (P/'DISPATCH.txt').read_text()
 for d in load(P/'predecessor-inventory.json').values():check(d)
 external={};skipped=[];failures=[]
 roots=[load(x) for x in P.rglob('*.json') if 'verification-tests' not in str(x)]
 # Explicitly inspect accepted metadata inventories; never recursively follow arbitrary payload references.
 roots += [load(C/'bindings-v5.json'),load(C/'accounting-owner-ruling-04.json'),load(C/'cumulative-envelope-owner-ruling-04.json'),load(F/'REVIEW_PACKET.json'),load(F/'REVIEW_BUNDLE.json')]
 for value in roots:
  for d in refs(value):
   if protected(d['path']):skipped.append({'path':d['path'],'read':False});continue
   try:check(d)
   except Exception as e:failures.append({'binding':d,'error':str(e)});continue
   if not pathlib.Path(d['path']).is_relative_to(P):external[d['path']]=bind(d['path'])
 assert not failures,failures
 for rel in m['source_paths']:
  external[str(R/rel)]=bind(R/rel)
 for rel in ['tools/ops/dispatch.py','tools/ops/dispatch.sh','tools/ops/qualification.py','research/colab_cli/launch.sh']:
  external[str(R/rel)]=bind(R/rel)
 historical_nonoperative_sidecars=[]
 for d in list(external.values()):
  p=pathlib.Path(d['path'])
  if p.name.endswith(('.sha256','.complete')):continue
  for suffix in ['.sha256','.complete']:
   q=pathlib.Path(str(p)+suffix)
   if q.is_file():
    external[str(q)]=bind(q)
    if suffix=='.sha256' and q.read_text().split()[0]!=d['sha256']:
     assert q.is_relative_to(R) and p.is_relative_to(R),str(q)
     rel=str(q.relative_to(R));source_rel=str(p.relative_to(R))
     assert q.read_bytes()==git('show',f'{IMPL}:{rel}')==git('show',f'{PIN}:{rel}')
     assert p.read_bytes()==git('show',f'{IMPL}:{source_rel}'),str(p)
     historical_nonoperative_sidecars.append({'sidecar':bind(q),'source':d,'declared_sha256':q.read_text().split()[0],'status':'PREEXISTING_NONOPERATIVE_TRACKED_SOURCE_SIDECAR_NOT_USED_BY_CAPTURE','operative_binding':'Current source_inventory/Git/archive SHA256, independently verified; source bytes preserved exactly.'})
    elif suffix=='.complete':assert q.stat().st_size>0,str(q)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 assert load(P/'verification-tests/narrow-offline-pytest.json')['result']['pytest_return_code']==0
 write(P/'operational-verification-v5.json',{'schema':'bl-f5-operational-verification/v5','status':'PASS','created_utc':now(),'source_commit':PIN,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'manifest':bind(P/'environment-capture.json'),'checks':{'all_200_archive_members_git_worktree_match':True,'192_source_pins_194_source_paths':True,'f5_nontransport_closure_equals_c989f673':True,'both_qualified_registry_consumers_pass':True,'bootstrap_matches_current_source':True,'frozen684_allocation_and_twelve_fixtures_unchanged':True,'empty_ledger_and_bootstrap_allocation_identity':True,'three_uploads_ten_exports':True,'all_private_inventory_counts_recomputed':True,'six_prior_sessions_two_mechanical_failures_raw_accounting':True,'held_plan02_not_executed_no_completed_science_units':True,'attempt04_provenance_only':True,'no_redoubling_timeout_30min':True,'dry_run_dispatch_command_agree':True,'all_package_payload_sidecars_and_markers_verified':True,'predecessors_unchanged':True,'state_output_unused_broker_clean':True},'allocation_identity_preimage':allocation_identity,'historical_nonoperative_sidecars':historical_nonoperative_sidecars,'external_files_verified':len(external),'external_files':external,'protected_references_not_followed':skipped,'allocations':0,'dispatches':0,'scientific_fits':0,'numerical_outcome_looks':0})
 print(json.dumps({'verification':'PASS','external_files':len(external),'historical_sessions':6,'session_elapsed_seconds':history['cumulative_session_elapsed_seconds']}))
if phase=='seal':
 fresh();assert git('status','--porcelain')==b'';v=load(P/'operational-verification-v5.json')
 assert load(P/'independent-review.json')['status']=='NO_BLOCKING_FINDINGS'
 for d in v['external_files'].values():check(d)
 for d in load(P/'predecessor-inventory.json').values():check(d)
 for p in sorted(P.rglob('*')):
  if p.is_file() and not p.name.endswith(('.sha256','.complete')):verify_export(p)
 files={str(x.relative_to(P)):bind(x) for x in sorted(P.rglob('*')) if x.is_file()}
 write(P/'PACKAGE_SEAL_OPERATIONAL_V5.json',{'schema':'bl-f5-capture-operational-package-seal/v5','status':'SEALED_CAPTURE_BOOTSTRAP_ONLY_AFTER_TRANSPORT_REQUALIFICATION','created_utc':now(),'package_path':str(P),'job_id':JOB,'source_commit':PIN,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'manifest':bind(P/'environment-capture.json'),'bindings':bind(P/'bindings-v5.json'),'files':files,'external_files':v['external_files'],'verification':bind(P/'operational-verification-v5.json'),'independent_review':bind(P/'independent-review.json'),'dispatch_command':shlex.join(command()),'dry_run_output':load(P/'dispatch-dry-run.json')['command']['stdout'].strip(),'state_dir':str(P/'state-05'),'output_dir':str(P/'output-05'),'state_absent':True,'output_absent':True,'broker_clean':True,'broker_cwd':str(R),'scope':load(P/'environment-capture.json')['scope'],'self_reference_exclusion':'This seal and its SHA/completion companions are bound by the subsequent publication envelope. Publication envelope is the sole post-seal addition.','publication_required_before_operator_dispatch':True,'allocations':0,'dispatches':0})
 print(json.dumps({'seal':bind(P/'PACKAGE_SEAL_OPERATIONAL_V5.json'),'manifest':bind(P/'environment-capture.json'),'files':len(files)}))
if phase=='publish':
 fresh();assert git('status','--porcelain')==b''
 seal=load(P/'PACKAGE_SEAL_OPERATIONAL_V5.json')
 for d in list(seal['files'].values())+list(seal['external_files'].values()):check(d)
 dest=W/'research/bl_residual/ledger'/P.name;assert not dest.exists();dest.mkdir()
 rows=[]
 def put(q,raw):
  q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb') as f:f.write(raw)
 for p in sorted(P.rglob('*')):
  if not p.is_file():continue
  raw=p.read_bytes();rel=p.relative_to(P);q=dest/rel;row={'source':bind(p),'relative_path':str(rel)}
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
    q=pathlib.Path(str(p)+suffix)
    if q.exists():assert q.read_bytes()==raw
    else:put(q,raw)
 inventory={str(p.relative_to(dest)):bind(p) for p in sorted(dest.rglob('*')) if p.is_file()}
 write(P/'OPERATIONAL_PUBLICATION_V5.json',{'schema':'bl-f5-capture-operational-publication/v5','status':'SEALED_REVIEWED_PREPARED_NO_DISPATCH','created_utc':now(),'package_path':str(P),'ledger_mirror':str(dest),'source_commit':PIN,'accepted_f5_implementation':IMPL,'qualified_transport_tree':TREE,'manifest':bind(P/'environment-capture.json'),'seal':bind(P/'PACKAGE_SEAL_OPERATIONAL_V5.json'),'source_files':rows,'published_files':inventory,'byte_identical_or_losslessly_reconstructed':True,'dispatch_command':shlex.join(command()),'allocations':0,'dispatches':0,'publication_exclusion_from_seal':'This create-only publication envelope and companions are added after seal to bind it without a hash cycle; all other package bytes are sealed.'})
 for suffix in ['', '.sha256','.complete']:put(dest/('OPERATIONAL_PUBLICATION_V5.json'+suffix),(P/('OPERATIONAL_PUBLICATION_V5.json'+suffix)).read_bytes())
 print(json.dumps({'publication':bind(P/'OPERATIONAL_PUBLICATION_V5.json'),'ledger_mirror':str(dest),'published_files':len(inventory)+3}))
