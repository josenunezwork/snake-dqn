# O2 V2 preparation verification — 2026-10-21

Outcome: **PREPARED_NO_DISPATCH**. No Colab call, account query or selection
trajectory. V1 package bytes and all protected packages remain unchanged.

Implementation/source: `f19d3f39050adfedc6eaef001a3364318fbb28bc`. Registration publication:
`040a6fcd853d04a57bc36da47dcbebca65c62023`. Machine registration SHA-256:
`82c1986ff42c7a98161b1f6ab657db0a61f82ab527cdd8bba4620410b698ce88`. It binds 260 exact source/rule inputs; actual sparse source
contains 264 files including the supervisor's standard additions.

Prospective measured-T23 power: smallest balanced dependence-robust allocation
**5097 worlds (1699/mix)**, bound .800179; prior 5094 bound .799974.
Holm family alpha .05. One-million-replicate Welch proxy: global-null any false
rejection .049071 (95% MC interval .048648–.049494); single-null configurations
.037524 and .049723. These are planning-model results, not O2 data.
The requested <=150-unit fallback freezes **1260 worlds (420/mix)** and bound
**.339888**; it remains underpowered. The retained observed mean>=40 point bar
precludes .80 full-GO power at true +40 under symmetric continuous error.

Round 5 audit is CLEAR: 1164 added selection worlds + 9 excluded fixtures,
all 96 original selection worlds retained, seven FRP exclusions, 34 published
inventory checks for added identities and another 34 for inherited worlds.
Whole rounds 0–4 rejected for collisions and preserved. The sealed Round 0
artifact directory was excluded before traversal and never opened.

## Actual Mac preflight

Real `launch_v2.sh` entrypoint, `nice -n 10`, CPU fp32, native thread caps 1,
serial 1 worker and pool 2 workers; AC/open lid/no thermal warnings checked.
Three excluded paired worlds, 40 base frames/arm, roots 10/20, H500 production
branches. Serial **50.888594167s**; pool
**30.604175208s**. All 9 full arm/world records match
exactly after removing only `elapsed_seconds`/`search_elapsed_seconds` recursively.
Warm measured base/search minima: 360.569881/331.595181 rows/s, exceeding frozen
planning floors 200/200. These tiny early-body tests do not measure late-body,
L4, paced serving or efficacy performance.

## Forecast, manifests and upload readiness

Conservative forecast: 37.366667 aggregate runtime-hours / 74.733333 units.
Hard caps: 149.733333 units, including 29.946667 reserve; each job <=462 minutes,
10 CPU workers bounded by nproc−2, one L4/runtime, 2 u/h planning rate.
Full-horizon excluded runtime timing/parity admission remains mandatory.
Next worker wave requires 4524 timeout-minutes versus 4500 available.

All 10 manifests passed local supervisor validation, exact pinned source
preparation and full `upload_readiness`/Z-9 qualification. No allocation.
Normal upload: 1,214,804 bytes, forecast at 96.308302 s within the 150 s normal budget;
compressed source fallback+fetch bound
241.971111 s <=300 s shared deadline.
Six public checkpoints are downloaded and SHA/size-verified on runtime.
Only the registration, catalog and preflight receipt are uploaded.

| Manifest | Bank slice | Timeout min | Hard units | SHA-256 |
|---|---:|---:|---:|---|
| chunk-00.json | 0:130 | 462 | 15.400000 | `f8a6520ba1b478998a1766f554edd862b84436f63d603fa4b77bd1d569c84042` |
| chunk-01.json | 130:260 | 462 | 15.400000 | `ece13cdeefc04d60fa6e7af7a0740fbf9c8b2d8ba8e8d704ba3bcdfaaf494a99` |
| chunk-02.json | 260:390 | 462 | 15.400000 | `11ca201c5a77bda3b36317491f979cbc771b24d687856be3caec184ff75fa2cb` |
| chunk-03.json | 390:520 | 462 | 15.400000 | `e2627c001bb39c9971860d263e93a5ae010eb364a481e0daa0991bb894c3c11b` |
| chunk-04.json | 520:650 | 462 | 15.400000 | `fc3797b367eed7a61dbc1739f70ebcd8cf88d983abfc3cbc86a9cc3e051e6095` |
| chunk-05.json | 650:780 | 462 | 15.400000 | `ee572c06f65c5822eef9607add09b32125b8395f2da0861269f0033e36a678f3` |
| chunk-06.json | 780:910 | 462 | 15.400000 | `9802a80145e5f4926f00b7853e4aff568b620a36ef705551ebc4eff68220e35d` |
| chunk-07.json | 910:1040 | 462 | 15.400000 | `45f40795dead0182797ab01891c0cdb2c448905c0d7b214d62ee7f0a4b6edc83` |
| chunk-08.json | 1040:1150 | 398 | 13.266667 | `ff728b87fd11e219783d2ec4c1eb7264e59f5f811c7f0bb8b7d771066c872abf` |
| chunk-09.json | 1150:1260 | 398 | 13.266667 | `b0c73b980e49a350ad24dbb35b62c695d5df7f03529d88805b706590caccb2ec` |

## Verification and artifacts

`nice -n 10`, all numerical thread caps 1, project venv:

- `python -m pytest -q research/t34_o2_harvest`: **77 passed in 7.95s**.
- `python -m pytest -q tests/test_colab_cli_source.py tests/test_colab_cli_supervisor.py -k 'source or module_entrypoint_validate'`: **16 passed, 62 deselected in 73.03s**; fake transport only.
- Black 100, isort Black profile, flake8, new Bash syntax and diff checks pass.
- Independent read-only code review and frozen 260-file/source/bank audit pass.

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o2-20261021`. `v2-manifest-validation.json` binds every manifest and
upload receipt. `v2-serial-parallel-proof.json` binds tested payloads;
`v2-preflight-report.json` binds measured forecast input. Every publication has
SHA/completion sidecars. Root `DISPATCH.txt` now points to V2 only; original
V1 dispatch and its original sidecars are preserved under
`v1-dispatch-preserved/`. All output/state paths are fresh.

No Colab execution or live account query; no O2 selection data, training,
paced test, strict gate, serving qualification or release. No active tmux
session modified (`git-sync`, `phase0-selection` observed only). Runtime
full-horizon timing, final export/receipt validity and independently confirmed
runtime disappearance remain future evidence. Forecasts are not completed jobs.
