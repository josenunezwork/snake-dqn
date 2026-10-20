# T34 retention preregistration package verification — 2026-10-20

Status: **draft package complete; no pre-check or retention efficacy run**.
Worktree `/Users/josenunez/Projects/ml/snake-dqn-retention-prereg`, branch
`t34-retention-prereg-20261020`; clean starting commit
`ff3ab91a` (target spec). Final executable source commit:
`70230bff90c494797a405e267a4f9bc719ad4d8c`. Publication of generated JSON and
this note does not change the executable closure.

## Evidence delivered

- `PREREGISTRATION_RETENTION_V2.md`: exact design, metric denominators, inference,
  bars, decision quotations, conservative uncertainty precedence, caps and
  operator-only admission. Header remains DRAFT_PENDING_OPERATOR_RATIFICATION.
- `RETENTION_PROFILE_V2.json`: deployment, controller, timing, survival amendment
  hash/ratification tuple and prospective quiet-host policy.
- `RETENTION_BANK_V2.json` and `SEED_AUDIT_RETENTION_V2.json`: 32 terminal worlds
  per mix, 16 product sessions and one timing seed; 113 unique seeds, separate
  namespaces, zero concrete collision hits. Frozen allocator was committed
  before allocation. The audit scanned docs/, research/, src/, scripts/ and
  visible artifact text, plus seven historical FRP exclusion inventories.
  Recorded limits exclude opaque binaries and unrecorded external runs.
- `RETENTION_CHECKPOINTS_V2.json`: six exact checkpoint paths and hashes, also
  embedded in the registration; no checkpoint byte was changed.
- `RETENTION_OC_V2.json`: complete 20,000-replicate development-only simulation
  for each of five declared proxy/stress models, source-bound pool and simulator,
  scenario rates, Monte Carlo intervals, mass power and fixed/sequential costs.
- `REGISTRATION_RETENTION_V2.json`: all required path/hash bindings, final source
  commit/inventory, `ratified:false`, `trace_receipt:null`,
  `load_profile.quiet_ratified:false`. Production validator returns
  `{"valid":true,"authorized":false,"mode":"draft"}`.
- `registration_retention_v2.py` and `retention_oc_v2.py`: create-only package
  commands, source/evidence validation, one-thread bounded/resumable simulation.
  New tests cover collision rejection, malformed/mutated evidence, unauthorized
  ratification, exact uncertainty arithmetic and interrupted checkpoint writes.

## Final-source verification

Exact environment and commands:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-prereg
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
"$PY" -m pytest -q \
  tests/test_t34_pathb_registration_retention_v2.py \
  tests/test_t34_pathb_retention_oc_v2.py \
  tests/test_t34_pathb_retention_v2.py \
  tests/test_t34_pathb_analysis_v2.py \
  tests/test_t34_pathb_timing_v2.py
"$PY" -m black --workers 1 --check --line-length 100 src web/backend \
  research/t34_pathb/registration_retention_v2.py research/t34_pathb/retention_oc_v2.py \
  tests/test_t34_pathb_registration_retention_v2.py tests/test_t34_pathb_retention_oc_v2.py
"$PY" -m isort --profile black --check-only src web/backend \
  research/t34_pathb/registration_retention_v2.py research/t34_pathb/retention_oc_v2.py \
  tests/test_t34_pathb_registration_retention_v2.py tests/test_t34_pathb_retention_oc_v2.py
"$PY" -m flake8 --jobs 1 src web/backend \
  research/t34_pathb/registration_retention_v2.py research/t34_pathb/retention_oc_v2.py \
  tests/test_t34_pathb_registration_retention_v2.py tests/test_t34_pathb_retention_oc_v2.py
git diff --check
```

**71 passed in 3.88 seconds**; Black checked 153 files, isort/flake8 and diff
checks passed. Serial pytest and numerical pools limited to one. No shared
simulator/trainer/serving implementation was modified; the full non-slow suite
was not rerun for these package utilities.

Exact preparation commands (already executed; create-only outputs now exist):

```bash
"$PY" -m research.t34_pathb.registration_retention_v2 allocate
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out research/t34_pathb/RETENTION_OC_V2.json \
  --state /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-prereg-20261020/oc-state-70230bff.json \
  --reps 20000 --seed 3420261020 --max-seconds 60
"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
"$PY" -m research.t34_pathb.registration_retention_v2 validate
```

Allocation returned `bank_rows:96, seed_count:113`. OC returned PASS,
`completed:20000, failed_cells:0`. Maximum protected joint mass-and-band rate:
**0.01445**, below **0.06**. This is a simulated design check, never retention
performance. At true +40, released-s3 scripted-shape proxy mass-sign power is
0.60425 (conservative Holm first-rank alpha .05), 0.75265 (alpha .10); probability
of point >=40 plus positive bound is 0.49245, and with survival bands 0.33055.
Illustrative sequential expected N/mix is 31.9128 vs fixed 32. Frozen/mixed
terminal variance and product success remain unproven. See the receipt's full
proxy, bounded-survival-shift and cost limitations.

An earlier development OC receipt bound source `aa364509` and was preserved
without editing at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-prereg-20261020/oc-aa364509.json`.
Read-only review found a checkpoint-temp recovery issue; it was fixed and tested
before final source freeze and the final receipt above. Only that final receipt
is bound by this registration. No scientific rule or threshold was retuned.

## Deferred by the spec

Delegated operator review/ratification, actual final-source PASS trace receipt,
authorized >=50/64 T3 pre-check, observed Phase 0 completion and quiet-host load,
then the one registered retention run. Exact commands are in
[README_V2.md](README_V2.md#retention-registration-package). Ratification binds
the exact HEAD recorded by the real pre-check; JSON-only ratification commits
may follow, with the execution closure unchanged. No fake evidence or automatic
retry is provided. The simulator and source hashes must still match at use.

No live services or tmux sessions were started, killed or modified. Released
checkpoints, serving pins/defaults, leaderboard, release-monitor records and
historical hash-bound documents were untouched. No cloud allocation, teacher
labels, strict gate or release occurred. Historical M2/M3 NO-GO findings stand.
