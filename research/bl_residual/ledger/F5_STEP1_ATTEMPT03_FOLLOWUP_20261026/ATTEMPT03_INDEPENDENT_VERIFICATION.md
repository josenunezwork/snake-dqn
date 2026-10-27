# F5 step-1 preliminary qualification attempt 03: independent verification

Status: **PASS for the requested preliminary-evidence verification**.

This is a read-only local verification of the complete-evidence publication and
the saved attempt package. It does not admit measurement, perform a scientific
look, prepare or seal a target, call a provider, dispatch a job, or establish
current account state.

Verified source checkout:

- repository: `/Users/josenunez/Projects/ml/snake-dqn-f5rawenv`
- HEAD and `origin/main`: `bc22e072a70941fab45b37fc624276bcd7453e08`
- pinned runtime source in the attempt package: `a2574377194b1688722eab4003ec9b697bde812d`
- pinned source tree: `a950c12f6b24f65f40594921996b68fc06912529`
- checkout was clean at the start and immediately after read-only verification; an unrelated untracked `research/bl_residual/ledger/F5_STEP1_ATTEMPT03_FOLLOWUP_20261026/` directory appeared later from the concurrent workflow and was not touched by this verifier

## Publication and reconstruction

`research/bl_residual/ledger/F5_STEP1_ATTEMPT03_20261026/PUBLICATION.json`
reconstructs all 71 declared originals, totaling 14,868,322 bytes. All 73
payload parts were present, each had a matching SHA sidecar and `complete`
marker, and there were no orphan or missing payload parts. Every reconstructed
byte stream matched its original path, size, and SHA256. The publication and
README sidecars and completion markers also matched.

Publication SHA256: `05cc31778183ce951f70e81ebb0f90a26a1ef12e47c0cce8490b010418e47122`.

The two required exports were independently checked against their original
sidecars and completion markers:

| Export | Size | SHA256 | Result |
|---|---:|---|---|
| `exports/environment.json` | 13,420,032 | `eed2ed89c41d99ccbadc769752376125036d06d3e79e1435502c0a7e4893b55e` | PASS |
| `exports/qualification.json` | 421 | `a8c18a3047844648eb0d13ffcfaade56de9b8bdcbd671b92afa52bd1e250fd0d` | PASS |

## Environment and CUDA evidence

The current export passed the repository's v6 validator:

- schema: `bl-residual-f5-environment-receipt/v6`
- capture status: `SUCCESS`
- schema pin matched `818ddf0adaa0e892679cc5bba54b0fe1579932e7fad222717eb661a6d85cee3b`
- environment descriptor matched the admitted v6 descriptor `bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5`
- lock matched `2577ed9dd9fe34468f5562262def7bccb352c9c0bdce06b293bb4b12a0ea5301`
- `collect_environment_failures()` returned `[]`

The pinned comparison-only capture-05 adapter independently validated the saved
v5 receipt and compared it through the v6 canonical projection. The comparison
passed with identical canonical identity `ed7087af5a8e9257f36952f853b97ac224b56d57a9f2116894f893da93beeba8`.
The raw assignment identities remained distinct: capture-05 projected raw SHA
`7032951df9142ea60889f40bcb7bb880e6d4a2dcfaac9344c72248b7861500d9`; attempt
03 raw SHA `f61d9c47cf7f260c303a9032f155e366736758d0dd022506c3ddeeb02785701b`.
This is consistent with a fresh allocation: capture-05 used GPU UUID
`f6a73155-cd8e-56d4-2b05-6423d04387e2` and kernel `6.6.122+`; attempt 03 used
GPU UUID `e2073ab6-5ef3-4248-60e9-d26ab8556260` and kernel `6.6.159+`.
Common admitted identity remained L4, CUDA 12.6, Torch 2.9.1+cu126, CPython
3.12.3, x86_64, 12 CPUs, the same descriptor/lock, package versions, numeric
controls, and thread controls.

The saved terminal observer contains exactly one attributable observation:
`research/bl_residual/f5_qualification.py:136:_select_cuda`, operation
`Tensor.argmax`, device `cuda:0`, shape `[1, 6]`, `cuda_decisions: 1`.
The pinned source at commit `a2574377` confirms that `_select_cuda` performs a
CUDA fp32 forward, masked action selection, and `torch.cuda.synchronize`; the
driver calls that source-native path and then compares CUDA and CPU Q values and
actions before emitting `native_cuda_cpu_parity: PASS`. Critical pinned source
hashes for `f5_qualification.py`, `f5_step3_driver.py`, `f5_environment.py`,
the v6 schema and descriptor, `canary_observer.py`, `source.py`,
`release.py`, and `runtime.py` all matched the staged source plan and pinned
tree.

## Receipt, qualification, and release evidence

The saved attempt package reports:

- `receipt.json`: `status=PASS`, `stop_verified=true`, elapsed `753.1065790839493` seconds, `cleanup_error=null`, `error=null`
- `terminal.json`: `status=succeeded`, `returncode=0`, `reason=null`, terminal line `{"qualification": "qualification.json", "status": "QUALIFIED"}`
- `qualification.json`: `status=PASS`, `native_cuda_cpu_parity=PASS`, `physical_assignment=NVIDIA_L4`, redacted export
- saved broker release evidence: `sessions.json={}`, receipt accounting `assignments=[]`, and `usage_after.assignments=0.0`

The release conclusion is limited to saved evidence. No fresh broker/provider
query was made, so this report does not claim the assignment's disappearance
from the live account at verification time.

## Commands and outcomes

All Python verification commands used `PYTHONDONTWRITEBYTECODE=1`,
`nice -n 15`, and numeric thread caps of one. No timing workload ran.

1. `git status --short --branch; git rev-parse HEAD; git log -1 --oneline --decorate` — clean checkout at the publication commit.
2. A standard-library verifier iterated `PUBLICATION.json`, checked every payload SHA sidecar and completion marker, decompressed gzip entries, and compared every reconstructed stream to its original — 14 checks passed, including totals, uniqueness, payload coverage, exports, receipt, terminal, observer, v6 pins, manifest admission, and saved release evidence.
3. `python3` calling `research.bl_residual.f5_environment.validate_environment()` and `collect_environment_failures()` — v6 validation passed; failures `[]`.
4. `python3` calling `f5_environment_reference.capture05_reference()` and `compare_capture05_reference()` — capture-05 comparison passed; canonical hashes matched.
5. `python3 -m pytest -q -c /dev/null -p no:cacheprovider tests/test_f5_environment_exhaustive.py tests/test_f5_step3_package.py tests/test_f5_step3_driver.py` — **39 passed in 36.69s**.

The repository's ordinary pytest configuration could not start under the
available interpreter because `pytest.ini` requires pytest 9.0 while the
available pytest is 7.4.4. The `-c /dev/null` run bypassed only that
pre-existing tool-version gate; it did not modify source or product files.
The checkout has no `./venv/bin/python`, so the constrained checks used the
available `/opt/anaconda3/bin/python3` interpreter.

## Remaining scope and risks

The verification does not independently repeat the CUDA operation, collect
timing, measure a forecast, call the broker, inspect protected data, perform a
science look, or accept a target package. It verifies the saved source-bound
observer and receipt evidence. The saved receipt also records one failed
renewal-heartbeat control entry, but the terminal receipt has `cleanup_error`
and `error` null, later control operations completed, and the required saved
release evidence is internally consistent. This remains an operational detail
for Astra's acceptance review.

The next workflow must preserve the current HOLD because no lawful complete
forecast is available: ENGINEERING_V4 forbids inferring long-body behavior from
short-body timing, and no accepted long-body primitive metadata is present in
this preliminary package. The owner workflow may run `prepare-target` with the
HOLD forecast to record the expected refusal; that produces no `TARGET.json`.
Only a later complete measured forecast can make a target package eligible for
Astra's independent acceptance, followed by final seal, exact-package review,
family canary, and separately authorized measurement.
