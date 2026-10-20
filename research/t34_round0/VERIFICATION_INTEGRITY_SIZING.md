# Round 0 integrity and sizing verification — 2026-10-20

Status: **DRAFT_PENDING_OPERATOR_RATIFICATION**. Implements
`docs/research/specs/2026-10-20_t34_round0_integrity_sizing.md`.
Executable source commit: `8900d2b546bf6a2b0c8356ae9e99f877d865cb61`.
Final documentation commits do not change the source closure in these receipts.
No ratification, selection-bank execution, Colab operation, or tmux operation was
performed. The exact future ratify/manifest/dispatch/publication/replay/analysis
commands are in [README.md](README.md). The original preregistration, previous
amendment, all banks, seed audits and existing evidence remain byte-identical.

## Implemented controls

- A5 layout: L4, K12 CPU inference processes, one numerical thread/process, W1.
  Both arms and their metric replay execute in the same worker/runtime. Results
  publish in bank order while running work continuously fills the pool.
- Fixed study root, chunk/attempt and replay/analysis paths; create-only remote
  claims; immutable terminal inventories; append-only repo receipt-hash ledger.
  Admission fetches main and requires ledger publication and retained claims.
- Detached source fence at the ratified commit, including during consumer phases;
  unrecorded imported analysis/replay calls refuse admission. One decision file.
- Operator abort, source drift and unexplained interruption are KILL. Explained
  unconditional controller faults retain STOP, including the supervisor's actual
  failed-terminal receipt shape. They close further dispatch. One explicit,
  outcome-independent mechanical rerun is permitted only for the registered
  measured artifact-download failure after successful work and verified release.
- Mac replay selects exactly ceil(10% of allocation) per chunk, stratified by
  mix with largest-remainder quotas and canonical ties. The concrete 141/99
  partition samples 15+10=25 pairs: 10 scripted, 8 frozen, 7 mixed, both arms.
  >2% drops stop/report; any missing pair still blocks the fixed-N scientific look.
- Dispatch validates the complete script/hardware/environment/input/source and
  sizing closure before a remote claim or allocation. Ratification recomputes
  measurements from retained A5/A6 bytes, not editable derived numbers.

## Final A6, measurements and draft forecast

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020/`.
Final A6 is `a6-03/`: two excluded H500 pairs, CPU, one worker/thread,
**13.340194040996721 seconds**, status complete. The current draft validator
returns PASS with `ratified=false`, `dispatch_authorized=false`, and
`scientific_arms_run=false`. Earlier `a6-01/` and `a6-02/` remain preserved as
superseded source snapshots, not selection data or replacements for this receipt.

The two saved A5 benchmark reports under
`t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s{0,1}-output/exports/benchmark.json`
and their verified supervisor receipts supply teacher rates
57.17339765610936 and **54.37214064869172 worlds/hour/lane**. The latter prices
this plan. A6 records raw per-arm rollout and metric times. For planning,
max-over-the-two H500 fixture costs scale by 10 to H5000: bare
11.21844459004933 seconds/world and metric replay 3.808377089735586 seconds/pair.
Only those CPU costs receive a conservative 3x platform factor and divide by
K12; the A5 teacher rate is already aggregate. These are explicit projections,
not measured paired L4 throughput. A 15-minute setup/export/cleanup allowance
exceeds the observed A5 non-workload elapsed debit. No new Colab measurement ran.

| Chunk | Pairs | Workload incl. setup, min | Estimate, min | Session timeout, min | Export projection, MiB | Operational cap, u | Additional 20% reserve, u |
|---|---:|---:|---:|---:|---:|---:|---:|
| chunk-000 | 141 | 179.4226466843605 | 180 | 360 | 1586.25 | 12.0 | 2.4 |
| chunk-001 | 99 | 130.4456880975297 | 131 | 262 | 1113.75 | 8.733333333333333 | 1.7466666666666666 |

Each timeout is >=workload+25 and >=2x estimate. Exports use
9 MiB/world *1.25 and remain strictly below 2 GiB. At 2.0 u/h, initial operational
caps total **20.733333333333334 u**, reserve **4.1466666666666665 u**, total
**24.88 u**. Initial attempts plus the maximum single mechanical rerun of every
chunk reserve **49.76 u**. These are prospective liabilities, not spend.
At most three account assignments; the operator runbook uses serial chunk
admissions with ledger publication between them. It makes no concurrent-lane
speedup claim.

## Physical worker parity and bounded source transport

The separate `check_worker_parity.py` at the artifact root runs the two excluded
fixtures as direct serial, spawned worker-serial, and two spawned workers, with
one numerical thread/process and deterministic ordered comparison. All three
scientific payload SHA-256 values are:
`09d00b7a304d861f880d52e14c8ad9d22567770667ae6991574586110fd12917`.
Comparison includes both arms' complete six-snake action traces, physical-state
hash streams, identities and physical replay metrics. Profile timing is excluded.
This is two-world, same-Mac scheduling parity, not x86 identity or L4 qualification.
Evidence: `worker-parity-01/receipt.json`; all three normalized payloads and raw
traces are preserved.

Local source preparation passed: **839 files**, **3,395,318 compressed bytes**,
**13,508,201 unpacked bytes**, both below 32 MiB. Evidence:
`source-01/source-plan.json`, `source-01/source.tar.gz`, `source-01/qualification.json`.
This validates bounded delivery preparation without allocation; it does not prove
a runtime executed or released. The job requires the normal detached Git source
delivery for its commit/worktree fence; sparse non-Git fallback refuses execution.

## Exact verification commands

All numerical commands below use these limits; no xdist run exceeds two workers:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-round0-integrity
export PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export PREP=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020
export A6="$PREP/a6-03"
export A5ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991
"$PY" -m research.t34_round0.bank --preflight-inputs "$A6/inputs"
SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes \
  "$PY" -m research.t34_round0.runner --preflight \
  --bank "$A6/inputs/bank.json" --audit "$A6/inputs/audit.json" \
  --checkpoint /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth \
  --output-dir "$A6/output" --runtime-id t34-round0-integrity-a6-03 \
  --start 0 --count 2 --device cpu --threads 1 --seconds 600
"$PY" -m research.t34_round0.sizing \
  --a5-benchmarks "$A5ROOT/a5-l4-s0-output/exports/benchmark.json" "$A5ROOT/a5-l4-s1-output/exports/benchmark.json" \
  --preflight-dir "$A6/output" --setup-minutes 15 --out "$A6/A5_RECEIPT.json"
"$PY" -m research.t34_round0.validate --preflight-dir "$A6/output" \
  --a5 "$A6/A5_RECEIPT.json" --out "$A6/VERIFICATION.json"
SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes PYTHONPATH="$PWD" \
  "$PY" "$PREP/check_worker_parity.py"
"$PY" -m pytest -q tests/test_t34_round0_registration.py tests/test_t34_round0_manifest.py \
  tests/test_t34_round0_runner.py tests/test_t34_round0_integrity.py tests/test_t34_round0_analysis.py \
  tests/test_colab_cli_supervisor.py tests/test_colab_cli_source.py tests/test_colab_cli_admission.py \
  tests/test_t34_teacher_v2.py tests/test_t34_engine_fanout.py
BLACK_NUM_WORKERS=1 make PY="$PY" lint
BLACK_NUM_WORKERS=1 "$PY" -m black --check --line-length 100 research/t34_round0 tests/test_t34_round0_*.py
"$PY" -m isort --profile black --check-only research/t34_round0 tests/test_t34_round0_*.py
"$PY" -m flake8 research/t34_round0 tests/test_t34_round0_*.py
bash -n research/t34_round0/run.sh
git diff --check
```

The source-archive qualification used `research.colab_cli.source.prepare` with
`commit=head()`, `script=research/t34_round0/run.sh`, and
`source_paths=sorted(source_hashes())`, writing only to fresh `source-01/`.
The commands above reproduce verification, but existing create-only outputs
must be preserved; choose fresh paths for a reproduction.

Final focused/integration result: **176 passed, 1 failed in 560.53 seconds**.
All **68 Round 0 tests passed**. The single combined-batch failure was
`tests/test_colab_cli_supervisor.py::test_provider_calibration_timeout_is_a_real_elapsed_bound`:
it expected exactly 135 mocked seconds but observed 191.31300000001343 seconds.
Its clean-release and INCOMPLETE-calibration assertions passed before the timing
assertion. This branch does not change the supervisor or that test/its fake
provider fixtures (verified against starting commit `4cced4f94db72a0dccb8a87e024359a8ba98dd08`).

The exact node was then tested in isolation:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020/baseline-4cced4f9
"$PY" -m pytest -q tests/test_colab_cli_supervisor.py::test_provider_calibration_timeout_is_a_real_elapsed_bound
cd /Users/josenunez/Projects/ml/snake-dqn-round0-integrity
"$PY" -m pytest -q tests/test_colab_cli_supervisor.py::test_provider_calibration_timeout_is_a_real_elapsed_bound
```

Starting commit: **1 passed in 13.37 seconds**. Updated worktree: **1 passed in
14.91 seconds**. This establishes an isolated pass on both trees; the cause of
the combined-batch timing failure is unresolved. No shared transport code or
qualified-source fingerprint was modified to suppress it. The baseline checkout
was created solely for this verification and is retained under the artifact root.
The combined batch is not reported as fully green.

Project lint, all 18 scoped Python files' Black/isort/flake8 checks, Bash syntax,
whitespace checks, real A6 validation, saved A5/A6 sizing validation, local source
archive preparation and physical worker parity passed.

The broader `"$PY" -m pytest -q -n 2 -m 'not slow'` collected 8,882 tests and was
intentionally interrupted before completion (about 27% progress) to keep closeout
bounded to this research package. No complete/full-suite green result is claimed.
Only that owned pytest process was interrupted; process inspection confirmed no
pytest/execnet process remained. No scientific phase was dispatched or aborted.

## Review and limits

Independent read-only review findings on pickle-safe fault propagation, K12
worker replenishment, supervisor transfer/fault receipt shapes, KILL precedence,
consumer look admission, complete manifest closure and sizing measurement
bindings were addressed with focused regressions. No live L4 validation, Mac
strict gate, serving qualification or release is represented by this work.
