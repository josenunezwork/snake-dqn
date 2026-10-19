# S1 implementation receipt — 2026-10-18

**Engineering package reviewed; production admission REFUSED. Registration DRAFT.**
No Colab allocation, study training, study decision-world evaluation, or serving
change was performed. Local fixtures are operational evidence only. Supervisor
acceptance alone cannot make this package launchable.

## Registration and identities

The owner authorized the two-arm 60-unit design in chat, recorded verbatim in
`OWNER_AUTHORIZATION.json`. Claude must insert the approval status specified in
`PREREGISTRATION.md` before dispatch; nothing here supplies that approval.

Five fresh paired blocks are `223213833, 3055723932, 1783343308, 1125234883,
1809856951`. The complete shared bank has 640 worlds, 64/32/32 per seed,
under `s1-mature-h5000-terminal-hero/20261018/v5`. It supports 1,920 policy
episodes: N1 final, N3 final, and shared B8. The ten timing worlds are disjoint.
The collision audit rejected whole provisional allocations before data, then
passed the final allocation against visible prior inventories including Q1,
bound strict banks, FRP exclusions, and visible artifact JSON/JSONL. Opaque
binaries remain an explicit coverage limit.

B8 is pinned to
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
`INPUTS.json` binds the actual initializer, qualified demo/league archives,
seven-file upload inventory, Q0b L4 packet, and Mac parity reference. The
qualified packet was extracted from the original archive after verifying
`5aab1c9a37c30ed0dd415341b7721be07096144fa2287396161159e62d88ba9d`;
packet SHA is `e66dc4fe06227b3f0046c517c4fb819c52558a54bc0c0f2ed8ce17ea2d2ef53a`.
It independently validates as QUALIFIED. The existing burned Q1 four-world
parity fixture is reused only for operational identity, never S1 efficacy.

## Measured preflight and refusal

`TRAIN_PREFLIGHT.json` preserves both actual CPU adapter runs and their external
artifacts under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/preflight/`.

- The 124-second counter fixture targeted 12,000 decisions and reached 12,048.
  With two actors, eight worlds and three seats, the inherited bound is 47;
  observed overshoot was 48. The adapter refused its final export.
- The separate 125-second, 12,001-target fixture exercised chunk export and
  recovery. The real loader recognized the recovered snapshot and its online,
  target, optimizer, replay, rings, labels, RNG, schedules and counters. This is
  export-only evidence; choosing a nonmultiple target does not qualify the
  production counter boundary.

The trainer continues ingesting ordinary actor frames while a final full-state
barrier is pending (`train_m3c.py`, loop around lines 4463–4528). The inherited
`actors * worlds * seats - 1` bound remains unchanged. No shared trainer or Q1
source was patched to conceal the incompatibility.

`EVAL_PREFLIGHT_FINAL.json` and its status sibling bind the final evaluator
source. Ten disjoint H5000 worlds were run both with two one-thread workers and
serially; ordered result bytes were identical. Parallel maximum episode time
was 8.311568 seconds, total 67.207463; serial maximum was 7.901620, total
63.676467. Observed complete command wall time was 154 seconds. The concurrent
repository test run may have added contention. Older timing evidence remains
preserved. Partial world artifacts are external, not committed.

Raw full replay at production capacity is approximately 9.4 GB before demo,
optimizer and actor state. Immutable chunks satisfy the 2 GiB per-file limit,
but do not establish total RPC transfer time. A tiny Mac snapshot cannot price
that path. Its complete L4 liability remains unknown and blocks admission.

## Forecast against unchanged caps

`ADMISSION_STATUS_FINAL.json` is produced by `price.py` and exits 1 with
`status: REFUSED`. Its complete upper bound is deliberately null.

| Dimension | Conditional reservation | Usable cap | Nominal cap |
| --- | ---: | ---: | ---: |
| Units | 38.3333 | 48 | 60 |
| GPU allocation hours | 19.1667 | 20 | 25 |
| Mac worker hours | 15.8657 | 19.2 | 24 |
| Elapsed hours | 27.0995 | 38.4 | 48 |

These numbers are conditional, **not admission**. Historical Q0b training
estimates are 30 minutes for N3 and 68 for N1; ten runs total 8.1667 hours and
16.3333 units at 2/hour before the new full-state transport cost. Reservations
use 60/136-minute timeouts plus five cleanup minutes per job, and include four
acceptance units/two conservative GPU hours. Mac pricing uses twice the slowest
measured episode, ten minutes of setup per identity at two workers, and two
worker-hours for audit, local preflight and verification. The watchdog bounds
the actual Mac phase. A completed full-size transport qualification must fit
inside those same envelopes, without borrowing reserves or raising caps.

The durable ledger charges acceptance once, reserves every remaining job before
allocation, keeps failed/unknown jobs outstanding, and charges request through
verified disappearance. It verifies fresh account evidence and freezes the
reviewed earliest `study_started_at`, so ledger initialization cannot restart
the 38.4-hour elapsed window. Owner-reported 1174.37 units / zero usage / zero
assignments is historical context, not a live measurement by this task.

## Verification

Commands run from `/Users/josenunez/Projects/ml/snake-dqn-s1-mature`:

```bash
./venv/bin/python -m black --check --line-length 100 research/s1_mature_20261018 tests/test_s1_*.py
./venv/bin/python -m isort --profile black --check-only research/s1_mature_20261018 tests/test_s1_*.py
./venv/bin/python -m flake8 research/s1_mature_20261018 tests/test_s1_*.py
bash -n research/s1_mature_20261018/train.sh research/s1_mature_20261018/launch.sh
SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -q tests/test_s1_*.py
make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint
SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -q tests/test_colab_cli*.py tests/test_s1_*.py
```

Scoped study checks pass: 48 tests, including complete coordinator flow with
allocation/evaluation boundaries faked, serial job ordering, scientific KILL
skip behavior, immutable snapshot recovery, independent audit, and ledger
concurrency/cap/evidence tests. The updated supervisor and study combined check
passed 134 tests before the final additional transport-layout regression test;
the supervisor subset was unchanged by that added study test. Core lint passes.

The broad command below completed with **7,824 passed, 26 skipped, eight failed**
and six warnings in 399.74 seconds; it is not a green full-suite claim:

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ./venv/bin/python -m pytest -q -n 4 -m 'not slow'
```

A separate detached baseline at `352144892658817717e2a6a78b3fe6d849f91b9a`
without S1 reproduced the M6 telemetry/reward oracle failures, T4 frozen actor
trace failure, missing built frontend T15 failure, serving fallback assertion,
and T21 hero NumPy/pytest failure-rendering segmentation fault. The other T21
seat variant showed the same worker-crash pattern. These existing failures were
preserved and not changed by this package.

## Operator sequence after the blockers are resolved

First qualify the original counter invariant and full-size transport inside the
unchanged caps, complete independent live CLI acceptance/review, and have Claude
approve the complete registration. Publish the approved source; `prepare`
requires a clean worktree exactly at the fetched `origin/main`. This is an
explicit readiness dependency, not permission to launch qualification now.

The independent acceptance review must bind the normal CPU receipt, deliberate
L4 fault receipt and observer, every current transport source, the original
study start, and acceptance charges. Its schema is enforced in `ledger.py`.
A raw `qualified: true` acceptance result alone is insufficient.

The following are the exact preparation and one-command dispatch sequence.
`pricing-qualified.json` and `acceptance-review.json` must be real reviewed
artifacts; neither exists as a fabricated passing placeholder in this package.
The commands currently refuse the DRAFT/refused state.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-s1-mature
git fetch origin main
git merge --ff-only origin/main
./venv/bin/python -m research.s1_mature_20261018.admission prepare \
  --seed-bank /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/SEED_BANK.json \
  --world-bank /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/WORLD_BANK.json \
  --seed-audit /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/SEED_AUDIT.json \
  --pricing /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/pricing-qualified.json \
  --artifact-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/production-01 \
  --manifest-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/jobs-01 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/plan-01.json

/Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/launch.sh \
  --plan /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/plan-01.json \
  --ledger /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/ledger-01.json \
  --acceptance-review /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/acceptance-review.json \
  --snapshots /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/snapshots-01 \
  --history q1=/Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/q0q1_20261017/PREREGISTRATION.md \
  --history a2=/Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/q0q1_20261017/OWNER_ADMISSION_20261018_A2.json \
  --history a3=/Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/q0q1_20261017/OWNER_ADMISSION_20261018_A3.json \
  --state-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/supervisor-01 \
  --artifact-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/production-01 \
  --supervisor /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/colab_cli/launch.sh \
  --colab /Users/josenunez/.local/bin/colab \
  --account-config /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission/account-01.json
```

No retry, purchase, retention, seed replacement or cap raise is built into this
sequence. Unknown state stops it. A valid scientific tripwire closes that arm,
retains its authenticated evidence, and continues only the healthy arm. A GO
buys fresh Mac strict preparation and cannot change serving.

The primary checkout, `git-sync` tmux session, released checkpoints, serving pins,
Q1 bytes and historical ledgers were not modified. Protected supervisor changes
were only fast-forwarded from the other Sol's published commits.
