exec(open('/tmp/f5_plan_round08_seal.py').read().split("phase=sys.argv[1]")[0])
fresh()
assert bind(R/'tools/ops/dispatch.py')['sha256']=='3247b1811b1e5cb033dde649704ed37fc84b2ae76d75994d79163171bfef3104'
write(P/'seal-publication-builder-final.py',pathlib.Path('/tmp/f5_plan_round08_finalize.py').read_bytes())
write(P/'final-validation-builder.py',pathlib.Path(__file__).read_bytes())
cmds=[run(['python3',R/'tools/ops/qualification.py','check','--repo',R,'--record',REG,'--ref',x]) for x in ['HEAD',PIN]]
cmds.append(run([R/'research/colab_cli/launch.sh','--state-dir',P/'state-01','--colab','/Users/josenunez/.local/bin/colab','validate',P/'manifest.json']))
write(P/'broker-validation-final.json',{'schema':'bl-f5-broker-validation/v4','status':'PASS','created_utc':now(),'commands':cmds,'registry':bind(REG),'qualified_acceptance_result':bind(load(REG)['acceptance_result']),'broker_wrapper':bind(R/'tools/ops/dispatch.py'),'manifest':bind(P/'manifest.json'),'allocations':0,'predecessor_validation':bind(P/'broker-validation.json'),'predecessor_wrapper_snapshot':bind(P/'broker-wrapper-preflight-observed.py'),'successor_reason':'Final wrapper check refuses changed account declarations before creating a live sibling; scientific_admission remains false; qualified transport unchanged.'})
dr=run(command()+['--dry-run']);fresh()
write(P/'dispatch-dry-run-final.json',{'status':'PASS','created_utc':now(),'command':dr,'wrapper':bind(R/'tools/ops/dispatch.py'),'manifest':bind(P/'manifest.json'),'allocations':0,'separate_live_account_preflight':bind(P/'live-account-preflight.json'),'state_absent':True,'output_absent':True})
files=['tests/test_ops_f5_plan_admission.py','tests/test_ops_dispatch.py','tests/test_bl_residual_f5_broker.py','tests/test_colab_cli_source.py','tests/test_colab_cli_schema_failures.py','tests/test_f5_capture_diagnostics.py']
argv=[str(PYTHON),'-m','pytest','-q','-n','0','-p','no:cacheprovider','--basetemp','/tmp/f5-round08-plan-operational-pytest',*files]
env=dict(os.environ);env['DYLD_LIBRARY_PATH']=str(R.parent/'snake-dqn/venv/lib/python3.12/site-packages/torch/lib')
def priority():os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
start=time.monotonic();stamp=now();v=subprocess.run(argv,cwd=R,capture_output=True,text=True,timeout=240,env=env,preexec_fn=priority)
write(P/'operational-tests.log',(v.stdout+v.stderr).encode());write(P/'operational-tests.json',{'schema':'bl-f5-plan-operational-tests/v1','status':'PASS' if v.returncode==0 else 'FAIL','command':argv,'started_utc':stamp,'finished_utc':now(),'elapsed_seconds':time.monotonic()-start,'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'test_sources':{x:bind(R/x) for x in files},'pytest_workers':0,'numeric_threads':1,'absolute_niceness':15,'device':'cpu','allocations':0})
print(v.stdout);print(v.stderr);assert v.returncode==0
commands=[run([PYTHON,'-m','black','--check','--line-length','100','tools/ops/dispatch.py','tests/test_ops_f5_plan_admission.py']),run([PYTHON,'-m','isort','--check-only','--profile','black','tools/ops/dispatch.py','tests/test_ops_f5_plan_admission.py']),run([PYTHON,'-m','flake8','tools/ops/dispatch.py','tests/test_ops_f5_plan_admission.py']),run(['git','diff','--check'])]
write(P/'lint-verification.json',{'status':'PASS','commands':commands,'allocations':0});print('FINAL_VALIDATION_PASS',dr['stdout'].strip())
