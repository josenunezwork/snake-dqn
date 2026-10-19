# Q0b A3 preserved A100 and L4 packets; Q1 admission refused

Outcome: **both preserved Q0b r6 packets QUALIFIED under A3; Q1 REFUSED**.
The complete Q1 forecast exceeds both unchanged runtime envelopes. No new GPU
qualification work, training, gameplay scoring, queue cancellation or enqueue
occurred. Both workers remain alive and idle under the owner's instruction.
This is retrospective operational hardware admission, not efficacy evidence,
a strict pass, serving qualification or release.

Implementation: `8a8a17f5ceaa7026182e565ff996078ad1974c78`, published on
origin/main before producing the new L4 archive. Registration A3 SHA-256 remains
`0b056540089473549493fc1d706796c912739e97d095bcc67843fa6c29309461`.
Original registrations, admissions, approved manifests and queue receipts remain
unchanged. This receipt supersedes the earlier instruction here to cancel L4 r6
and enqueue r7. The owner explicitly instructed: **"The L4 r7 manifest was never
enqueued and should not be."** The current task explicitly requests the same A3
requalification of both preserved pre-A3 attempts. Earlier A3 prose anticipated a
prospective L4 replacement; the actual L4 r6 admission here is retrospective,
using the already registered tolerance. No threshold, cap, seed or recipe changed.

## Retained rule and provenance

Policy lag counts integer published-weight versions. The pinned recipe publishes
once per `round(25/2)=12` finalized optimizer updates. A3 tolerates a mean increase
strictly below 1/12 publication (one finalized update), with no p95 increase and
unchanged ingestion-age p95 ceiling of 110% of matched control. Missing, negative
or nonfinite values and all source, parity, counter, clean-exit and duration
failures remain binding. This tolerance was written after the original A100
failure; it does not establish identical acting-weight histories or efficacy.

| Preserved cell vs matched restart control | Mean publications | Increase in updates | p95 | Age p95 seconds |
|---|---|---:|---|---|
| A100 mature/n1 | 0.14356725146198832 vs 0.14108689160943566 | 0.029764318230632 | 1 = 1 | 0.07978772130209107 vs 0.08269823640184767 |
| L4 mature/n3 | 2.250213492741247 vs 2.232656514382403 | 0.210683740306129 | 3 = 3 | 0.8343297842498032 vs 0.8175611844502327 |

Each original packet failed solely its disclosed mean-lag check. All four cells
on each GPU pass the full A3 validator, with clean exits, zero dropped rows and
zero cadence error; runtime parity passes. `requalify_q0b.py` accepts only the
A100 r6/mature-n1 and L4 r6/mature-n3 identities at pre-A3 commit
`98cb709b6febba82aa424b51796e1730e893ca77`. It verifies archive sidecars, exact
claim/terminal job, worker, token and generation, hardware, source identity,
commit, ordered/reconciled timestamps, bound pricing, Mac/runtime parity and
T22/T24 fixture hashes. Its receipt validator rechecks preserved measurement and
inclusive accounting. The exact historical tool SHA
`44f7d8376e3ee4edc7827415a43bb665e173c5406322d865c32ff2c8f99e98fe`
is accepted only for the immutable existing A100 A3 archive.

## Immutable archives and inclusive results

Mac Drive root:
`/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab`.
Paths below are relative to it. Every archive has a SHA-256 sidecar.

| Archive | SHA-256 |
|---|---|
| Original `q0q1-20261017/q0b/capacity-a100-20261018r6/result.tar.gz` | `d052e91f467f6b42bd4be6b01baaffd8effba306a89dd318acc68d2c948c77ea` |
| Existing A3 `q0q1-20261017/q0b/capacity-a100-20261018r6-a3/result.tar.gz` | `f46f85ab593fca4c770c1c36d5ee58bfa9273497251e58d5ad3f058d5789792a` |
| Original `q0q1-20261017/q0b/capacity-l4-20261018r6/result.tar.gz` | `a0d69dd39513e43116f4acd07dbac13ad5637ddb6f5e10cf48644db7fe24fb2b` |
| New A3 `q0q1-20261017/q0b/capacity-l4-20261018r6-a3/result.tar.gz` | `5aab1c9a37c30ed0dd415341b7721be07096144fa2287396161159e62d88ba9d` |

L4 claim:
`queue/claimed/q0b-capacity-l4-20261018r6/g0.notebook-bb4112838e9b4a58b46bd44681df2f5d.72046e5e16d84daa8bcdff5a97bf5bff.json`.
L4 terminal:
`queue/failed/q0b-capacity-l4-20261018r6/g0.72046e5e16d84daa8bcdff5a97bf5bff.json`.
A100 retains its original claim/token `1555b50b6783474894f6a1e5ee667fdf`
and worker `291b7018260c-c3915164906b4032af56cf8f2c1d17b4`.
The original failed queue states remain failed; no done receipt was fabricated.
The new archive preserves original measurement, wrapper failure, runtime parity,
fixtures and hash-bound claim/terminal evidence.

| Inclusive result | A100 | L4 |
|---|---:|---:|
| Claim-to-terminal seconds | 2435.551744699478 | 2612.2299268245697 |
| GPU hours | 0.6765421513054106 | 0.725619424117936 |
| Units | 3.585673401918676 | 1.451238848235872 |
| Separately billed CPU hours | 0 | 0 |
| Conservative Q1 attempt overhead seconds | 510.731003128476 | 689.3676633185694 |
| restart/n1 decisions/s | 2380.6587993910343 | 2396.930604564424 |
| mature/n1 decisions/s | 2430.777822164222 | 2387.8491010510165 |
| restart/n3 decisions/s | 13508.133308649843 | 7555.085775789426 |
| mature/n3 decisions/s | 13585.468090254257 | 7487.189438656447 |

Inclusive elapsed time charges claim arbitration, setup, parity, measurement,
export and cleanup once. Q1 overhead conservatively uses all outside-measurement
time, four parity worlds, maximum observed cell overhead and a 60-second reserve.

## Final Q0 debit and exact A3 numeric caps

`q0_accounting` completed all **12 attempts** since 2026-10-18 UTC, including L4's
post-midnight claim. Status: `COMPLETE_WITH_CONSERVATIVE_BOUNDS`. The two prior
abandoned claims retain their explicit conservative closure bounds; these are
included in T, not omitted or zero-filled. r7 has no attempt and no charge.

Accounting: `q0q1-20261017/admission/observed-q0-accounting-a3-final-r6.json`,
SHA-256 `1f6d8bfe1ba8171151a9a3540e73ce2cdbdaf286cefcc0a37679c698acbd661d`.

- **T = 68.63035524611341 units**; Q0 GPU = 13.394975767532985 hours;
  Q0 CPU = 0.06305555555555556 hours.
- A3 Q0 cap = T; observed overrun = 28.63035524611341 units.
- Combined nominal cap = T + 135 = **203.63035524611342 units**.
- Original reserve = **35 units**; usable combined = T + 100 =
  **168.63035524611342 units**. Q1 nominal allocation stays 135; usable stays 100.
- Runtime envelopes remain 19.2 usable / 24 total GPU-hours and 32 usable / 40
  total CPU-hours. Q0 CPU cap remains 8 hours; Q1 training cap remains 60 units
  plus admitted CPU evaluation.

## Actual Q1 generator refusal

The README admission command ran from clean committed source with both A3
archives into NEW directory
`q0q1-20261017/q1/jobs_approved_a3_r6_final`.
It exited 1 with exactly the GPU and CPU usable-envelope errors. It created
staged evidence but **zero queue manifests**; there is no admitted Q1 packet.
Manifest validation of 120 jobs and enqueue commands therefore do not apply.
The generator supports either or both repaired receipts, removing a failed queue
dependency only after matching the receipt's claim/terminal hashes to accounting.

Forecast uses all 20 runs (five fixed seeds per arm, 8M decisions each), all 3,200
evaluation episodes, and 100 four-world parity probes. L4 rates above imply
1,058.889–3,350.295 training seconds per run, plus 689.368 seconds per attempt;
registered actors remain 3/6 on 12 vCPUs. CPU timing is the measured L4 fixture
mean **56.45208326124998 seconds/episode**, efficiency 1, one worker/runtime.
Measured Mac staging/publish/cleanup is projected at 3x plus observed queue setup
and prelude and 60 seconds, yielding **381.4419980733728 seconds/evaluation job**.
The bound pricing receipt retains A100 5.3 units/hour observed, L4 2.0 owner
planning rate and CPU 0.5 conservative rate; these are not newly measured billing
rates. No new runtime estimates/timeouts or job manifests were emitted.

| Registered cap | Forecast | Usable limit / nominal cap | Decision and margin |
|---|---:|---:|---|
| Q0 units | 68.63035524611341 | T = 68.63035524611341 | PASS, equal |
| Combined units | 134.30072755019802 | 168.63035524611342 / 203.63035524611342 | PASS, 34.3296276959154 usable headroom |
| Q0 CPU-hours | 0.06305555555555556 | 8 | PASS |
| Combined GPU-hours | 29.468238339864364 | 19.2 / 24 | REFUSED, +10.268238339864364 usable; +5.468238339864364 nominal |
| Combined CPU-hours | 67.11074987439922 | 32 / 40 | REFUSED, +35.11074987439922 usable; +27.11074987439922 nominal |
| Q1 training units including overhead | 32.14652514466276 | 60 | PASS, 27.85347485533724 headroom |
| Q1 training + CPU evaluation units | 65.6703723040846 | 100 usable; training <=60 + admitted CPU eval | Units fit; CPU evaluation is not admitted |

Q1 training alone is 12.243442220561551 GPU-hours plus
3.8298203517698304 overhead hours. Q1 evaluation is 50.17962956555554 CPU-hours
plus 16.86806475328813 overhead hours, costing 33.523847159421834 units.
Even the nominal runtime envelopes fail; consuming the reserve cannot admit it.
No cap, seed, recipe or placement was changed to manufacture admission.

Refusal artifacts in that directory:

- `generator-invocation.json` and `generator.log`: exact real command and exit.
- `refused-study-plan.json`: requested forecast inputs, not dispatch authority.
- `admission-forecast.json`: `REFUSED`, `dispatch_authorized=false`; SHA-256
  `5feca49bc30df66025e761c190ab3a67ff84dd5ef9e635955b675c0449934f24`.
- `artifact-manifest.json`: hashes of all staged evidence, forecast and invocation;
  SHA-256 `4183b55d925fcc1033e822e111b56b3684d21047192599d06085937182176825`.

## Verification and handoff

Targeted requalification, generator, accounting, qualification, measurement,
adapter, integration and queue tests: **192 passed**. The dual-hardware tests
cover inclusive caps and ordering, archive/source tampering, r7/foreign hardware
and commit rejection, A3 p95 regression, historical tool digest restricted to
A100, and dependency removal only with matching accounting hashes. All four
repaired/unrepaired hardware combinations are tested; successful fixture packets
produce exactly 120 manifests. Real preserved archives were independently
qualified and validated read-only. Black, isort, flake8, `git diff --check` and
repository `make ... lint` passed. No whole-repository suite was rerun for this
bounded operational change.

Exact targeted verification (project venv, from the worktree):

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$PY" -m pytest -q \
  tests/test_requalify_q0b.py tests/test_make_q1_jobs.py \
  tests/test_q0q1_accounting.py tests/test_q1_qualification.py \
  tests/test_q0b_measure.py tests/test_q0b_queue.py tests/test_q1_queue_jobs.py \
  tests/test_q1_admission_integration.py tests/test_colab_queue.py \
  tests/test_colab_queue_safety.py tests/test_colab_queue_worker.py
make PY="$PY" lint
"$PY" -m black --check --line-length 100 \
  research/q0q1_20261017/requalify_q0b.py research/q0q1_20261017/make_q1_jobs.py \
  tests/test_requalify_q0b.py tests/test_make_q1_jobs.py
"$PY" -m isort --profile black --check-only \
  research/q0q1_20261017/requalify_q0b.py research/q0q1_20261017/make_q1_jobs.py \
  tests/test_requalify_q0b.py tests/test_make_q1_jobs.py
"$PY" -m flake8 \
  research/q0q1_20261017/requalify_q0b.py research/q0q1_20261017/make_q1_jobs.py \
  tests/test_requalify_q0b.py tests/test_make_q1_jobs.py
git diff --check
```

The real Mac requalification CLI published the new archive, then the real
`make_q1_jobs` CLI reached the registered budget refusal. No launcher or simulator
source changed and no GPU entrypoint was run. README has the reproducible
create-only review commands. **No enqueue commands are authorized or supplied.**
No worker, notebook pin, tmux session, checkpoint, serving pin, original archive,
original queue receipt or existing A100 A3 archive was changed.
