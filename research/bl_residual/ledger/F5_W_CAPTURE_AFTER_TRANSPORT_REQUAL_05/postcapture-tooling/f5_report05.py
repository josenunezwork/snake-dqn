exec(open('/tmp/f5_05_work.py').read().split("if __name__")[0])
import sys,collections
sys.path.insert(0,str(R))
from research.bl_residual.f5_environment import validate_environment,compare_environment,classify_receipt,collect_environment_failures,policy_binding
from research.bl_residual.f5_qualification import load_fixtures
from research.bl_ppo.exports import verify_export
from research.colab_cli.canary import evidence
out=P/'output-05';r=load(out/'receipt.json');m=load(out/'manifest.json');env=load(out/'exports/environment.json');q=load(out/'exports/qualification/qualification.json')
assert r['status']=='PASS' and r['stop_verified'] is True and r['error'] is None and r['cleanup_error'] is None
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R)==subprocess.check_output(['git','rev-parse','origin/main'],cwd=R)
publication_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
for rel,row in load(P/'OUTPUT_PUBLICATION_05.json')['published_files'].items():
 raw=(D/rel).read_bytes();assert sha(raw)==row['sha256'];assert subprocess.check_output(['git','show','origin/main:'+str((D/rel).relative_to(R))],cwd=R)==raw
validate_environment(env);comparison=compare_environment(env,env,mode='same-assignment');classification=classify_receipt(env)
assert q['environment_evidence']==comparison
fixtures,fixture_sha=load_fixtures(P/'timing-fixtures.json');samples=q['samples'];assert len(samples)==len(fixtures)==12
for x,y in zip(samples,fixtures):
 assert all(x[k]==y[k] for k in ['allocation_index','parent_id','seed','group','group_index','timing_only'])
 assert x['parity']=='PASS' and x['q_shape']==[1,6] and x['cuda_selection_seconds']>0
assert q['fixture_sha256']==fixture_sha and q['device']=='cuda:0' and q['fit_or_optimizer_executed'] is False
proof=evidence(m,r,out,R);assert proof==r['canary_evidence']
for x in r['exports']:
 p=out/'exports'/x['path'];assert verify_export(p)==x['sha256'] and p.stat().st_size==x['size']
assert len(r['exports'])==10
obs=proof['cuda_observations'];assert any(x['caller'].startswith('research/bl_residual/f5_qualification.py:') and x['operation']=='Tensor.argmax' and x['device']=='cuda:0' for x in obs)
saved=P/'state-05'/r['job_id']/'sessions.json';assert load(saved)=={}
assert r['usage_after']['assignments']==r['usage_after']['rate']==0
assert [x['command'] for x in r['cli_calls'][-3:]]==['stop','sessions','usage'] and all(x['status']=='completed' for x in r['cli_calls'][-3:])
def maxima(rows,keys):
 groups=collections.defaultdict(list)
 for x in rows:groups[tuple(str(x.get(k,'')) for k in keys)].append(x)
 return [{'group':dict(zip(keys,k)),'count':len(v),'max_elapsed_seconds':max(x['elapsed_seconds'] for x in v),'max_call':max(v,key=lambda x:x['elapsed_seconds'])} for k,v in sorted(groups.items())]
failed=[x for x in r['cli_calls'] if x['status']!='completed']
late=[x for x in r['cli_calls'] if x['elapsed_seconds']>=30]
report={'schema':'bl-f5-attempt05-postpublication-verification/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verdict':'PASS_CAPTURE_AND_DISJOINT_CUDA_TRANSPORT_ONLY','evidence_publication_commit':publication_commit,'publication_verified_before_report':True,'local_equals_origin_main_at_verification':True,'publication_inventory_sha256':bind(P/'OUTPUT_PUBLICATION_05.json')['sha256'],'job_id':r['job_id'],'source_commit':m['commit'],'implementation_commit':'c989f6738f6aad143eb97bc47f6bd9e6687057ce','qualified_transport_tree':'6e8c1982b5c9ed115bfebd3145ad0720b15522fa','receipt_sha256':bind(out/'receipt.json')['sha256'],'elapsed_seconds':r['elapsed_seconds'],'environment':{'admitted':True,**policy_binding(),'file_sha256':bind(out/'exports/environment.json')['sha256'],'canonical_sha256':comparison['actual_canonical_sha256'],'same_assignment_comparison':comparison,'diagnostic_failures':collect_environment_failures(env),'field_classification_count':len(classification)},'fixtures':{'count':12,'allocation_indices':[x['allocation_index'] for x in samples],'fixture_sha256':fixture_sha,'atol':q['parity_atol'],'rtol':q['parity_rtol'],'all_cuda_cpu_q_allclose':True,'all_exact_selected_actions_equal':True,'evidence_basis':'All 12 saved PASS rows from pinned f5_qualification.run: PASS follows Q allclose and exact action equality. Raw Q/actions not exported; no new CUDA execution.'},'observer':obs,'transfer':{'seconds':proof['transfer_seconds'],'inclusive_bar_seconds':150,'pass':proof['transfer_seconds']<=150},'staging':{'seconds':proof['upload_elapsed_seconds'],'source_delivery_seconds':proof['source_delivery_seconds'],'inclusive_bar_seconds':300,'pass':proof['upload_elapsed_seconds']<=300},'export_recovery':{'count':10,'verified':True,'exports':r['exports'],'verifier':'research.colab_cli.canary.evidence and research.bl_ppo.exports.verify_export'},'runtime_release':{'stop_verified':True,'verified_from_saved_evidence':True,'saved_sessions':{},'saved_sessions_sha256':bind(saved)['sha256'],'usage_after':r['usage_after'],'basis':'Saved completed stop then sessions then usage; empty durable sessions snapshot. No current provider query.'},'per_call_latency':{'cli_by_command_operation_thread_status':maxima(r['cli_calls'],['command','operation','thread','status']),'contents_by_operation_thread_status':maxima(r['contents_transfers'],['operation','thread','status']),'control_by_operation_thread_status':maxima(r['control_operations'],['operation','thread','status']),'verified_transfer_by_direction_status':maxima(r['transfers'],['direction','status']),'cli_calls_at_least_30_seconds':late,'nonterminal_failed_cli_calls':failed,'max_cli_seconds':max(x['elapsed_seconds'] for x in r['cli_calls']),'max_successful_cli_seconds':max(x['elapsed_seconds'] for x in r['cli_calls'] if x['status']=='completed'),'interpretation':'Spikes were present and operational completion survived them: bootstrap PUT completed in 30.6766s; background heartbeat failed after 37.7272s; four foreground renewal execs hit approximately120s. Verified transfer57.3522s and staging76.8991s remained within unchanged150/300s bars. No staging recovery-idle call was observed, so this run does not isolate repair causality or prove the enlarged recovery budget was exercised. No latency elimination or future reliability claim.'},'refusals':[],'side_effects':{'colab_calls':0,'dispatches':0,'science_payloads_opened':0},'limits':['Capture and disjoint CUDA transport only; full source timing, optimizer/resume calibration, production forecast and Mac timing remain separate.','Paired bootstrap exports are calibration-only; no scientific fit/optimizer executed.','No fit acceptance, numerical outcome look, serving or release is authorized.']}
write(P/'POST_CAPTURE_VERIFICATION_05.json',report)
publish([(P/'POST_CAPTURE_VERIFICATION_05.json',pathlib.Path('POST_CAPTURE_VERIFICATION_05.json'))],D)
print(json.dumps({'report':bind(P/'POST_CAPTURE_VERIFICATION_05.json'),'verdict':report['verdict'],'transfer':report['transfer'],'staging':report['staging'],'latency_max':report['per_call_latency']['max_cli_seconds'],'successful_max':report['per_call_latency']['max_successful_cli_seconds']}))
