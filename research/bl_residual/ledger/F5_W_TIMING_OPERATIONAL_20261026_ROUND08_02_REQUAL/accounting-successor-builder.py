"""Create-only accounting successor after independent plan release. No allocation."""
import pathlib, sys, json, re, shlex
P=pathlib.Path(__file__).parent
exec((P/'operative-builder-v6.py').read_text().split("phase=sys.argv[1]")[0])
fresh()
receipt_source=PLAN/'output-01/receipt.json'
raw=receipt_source.read_bytes();receipt=json.loads(raw)
assert receipt['job_id']=='bl-f5-w-plan-r08-sr75b14427-01' and receipt['stop_verified'] is True
write(P/'plan-terminal-receipt-snapshot.json',raw)
calls=[run(['/Users/josenunez/.local/bin/colab','--auth','oauth2',x]) for x in ['sessions','usage','sessions']]
assert all('No active sessions found on server.' in calls[i]['stdout'] for i in [0,2])
assert 'Active assignments: 0' in calls[1]['stdout'] and 'Usage rate: 0.00/hr' in calls[1]['stdout']
assert not list(pathlib.Path('/Users/josenunez/.config/colab-cli/snake').glob('active-*.json'))
assert not list(pathlib.Path('/Users/josenunez/.config/colab-cli/snake').glob('pending-*.json'))
binding=load(P/'plan-operation-binding.json');binding.update({'state':'ENDED_TERMINAL_RECEIPT_BOUND','terminal_receipt':bind(P/'plan-terminal-receipt-snapshot.json'),'terminal_source':bind(receipt_source),'terminal_status':receipt['status'],'stop_verified':True,'endpoint_absent_in_both_fresh_sessions':True,'former_active_binding':bind(P/'plan-operation-binding.json')})
write(P/'plan-operation-binding-02.json',binding)
write(P/'live-account-preflight-02.json',{'schema':'bl-f5-live-account-preflight/v5','status':'PASS','observed_utc':now(),'calls':calls,'sessions':{},'other_assignments':{},'plan_operation':bind(P/'plan-operation-binding-02.json'),'account_assignment_cap':3,'active_broker_records':[],'pending_broker_records':[],'qualified_registry':bind(REG),'auth':'oauth2','allocation_calls':0,'fresh_post_plan_release':True})
m=load(P/'manifest.json');m['budget']['other_assignments']={};m['operational_admission']['live_account_preflight']=bind(P/'live-account-preflight-02.json');m['operational_admission']['plan_operation']=bind(P/'plan-operation-binding-02.json')
m['accounting_successor']={'supersedes':bind(P/'manifest.json'),'reason':'Independent plan terminal release occurred during package preparation; refresh actual account map before seal without modifying prior files.','science_change':False}
write(P/'manifest-02.json',m)
cmd=command();cmd[2]=str(P/'manifest-02.json');cmd[-1]=str(P/'DISPATCH-02.txt')
text=(P/'DISPATCH.txt').read_text().replace(str(P/'manifest.json'),str(P/'manifest-02.json')).replace(bind(P/'manifest.json')['sha256'],bind(P/'manifest-02.json')['sha256']).replace(str(P/'live-account-preflight.json'),str(P/'live-account-preflight-02.json')).replace(str(P/'plan-operation-binding.json'),str(P/'plan-operation-binding-02.json')).replace(str(P/'DISPATCH.txt'),str(P/'DISPATCH-02.txt'))
text+='\nOperative accounting successor: original manifest.json, DISPATCH.txt and active-map preflight remain historical, superseded before any timing execution. Plan terminal receipt is bound; fresh account has zero assignments/rate. This file is the operative DISPATCH successor.\n'
write(P/'DISPATCH-02.txt',text)
cmds=[run(['python3',R/'tools/ops/qualification.py','check','--repo',R,'--record',REG,'--ref',v]) for v in ['HEAD',PIN]]
cmds.append(run([R/'research/colab_cli/launch.sh','--state-dir',P/'state-01','--colab','/Users/josenunez/.local/bin/colab','validate',P/'manifest-02.json']))
write(P/'broker-validation-02.json',{'status':'PASS','created_utc':now(),'commands':cmds,'registry':bind(REG),'manifest':bind(P/'manifest-02.json'),'allocations':0})
dr=run(cmd+['--dry-run']);fresh();assert dr['stdout'].strip()==f'PASS_DRY_RUN {JOB} allocations=0 live_accounting=NOT_CHECKED'
write(P/'dispatch-dry-run-02.json',{'status':'PASS','created_utc':now(),'command':dr,'manifest':bind(P/'manifest-02.json'),'dispatch_instructions':bind(P/'DISPATCH-02.txt'),'separate_live_account_preflight':bind(P/'live-account-preflight-02.json'),'allocations':0,'state_absent':True,'output_absent':True})
write(P/'OPERATIVE_FILES.json',{'status':'ACCOUNTING_SUCCESSOR_OPERATIVE_NO_TIMING_EXECUTION','manifest':bind(P/'manifest-02.json'),'dispatch':bind(P/'DISPATCH-02.txt'),'dry_run':bind(P/'dispatch-dry-run-02.json'),'broker_validation':bind(P/'broker-validation-02.json'),'plan_binding':bind(P/'plan-operation-binding-02.json'),'account_preflight':bind(P/'live-account-preflight-02.json'),'superseded_historical_files':['manifest.json','DISPATCH.txt','dispatch-dry-run.json','broker-validation.json','live-account-preflight.json','plan-operation-binding.json'],'reason':'Plan independently released while preparation was in progress. Original active-map files immutable; no timing runtime consumed. Only accounting metadata changed; source, control, ledger, input/export bytes and scope unchanged.','dispatch_command':shlex.join(cmd),'allocations':0,'dispatches':0})
print(dr['stdout']);print(json.dumps({'plan_terminal_status':receipt['status'],'stop_verified':receipt['stop_verified'],'manifest':bind(P/'manifest-02.json')}))
