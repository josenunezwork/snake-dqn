# S1 parallel lanes: operator handoff (2026-10-19)

Implementation only. No live Colab request, dispatch, artifact publication, tmux action,
or change to dispatch-04's artifact files is part of this work.

## Dispatch-04 must exit before handoff

There is **no safe way to start two extra lanes now beside the already-loaded
serial dispatcher**. It has a static job list, assumes zero account assignments,
and rejects multiple outstanding reservations. Excluding its current job cannot
stop it from selecting a later job, and a new lock cannot govern code that is
already running. `--exclude-running` is refused.

Leave dispatch-04 alone until it exits naturally. One job settling is not proof
of dispatcher exit: its loop immediately advances. If it finishes the study,
do not start another dispatcher or repeat evaluation. If it exits before the
study finishes, the operator may use the conditional handoff below only after
independently confirming dispatcher exit, reconciling every outstanding
reservation through the existing reviewed settlement flow, and confirming
assignment disappearance. Unknown intervals or ambiguous failures block it.
These observations were not performed by the implementer.

The continuation keeps the original ledger, start time, charges, history and
settled artifacts. It binds a fresh published source plan for remaining jobs,
checks unchanged scientific and runtime identities, and recovers settled finals
from their original manifests. It cannot reset a failed attempt or grant a
rerun. Already-running jobs finish under their original plan. This handoff
supports settled successful finals, verified arm halts, and historical
pre-training staging/qualification settlements. A reconciled operational loss
or a settled manual retry is deliberately refused by this continuation and
requires a separately reviewed operator continuation. Do not apply the old
serial `--continue` workflow to a parallel-configured ledger. The original
once-per-arm eligibility, identical seed/recipe requirements, and caps remain
unchanged; no additional recovery workflow is authorized here.

## Bounds and scheduling

The amendment is `PREREGISTRATION_AMENDMENT_01_PARALLEL.md`. Both admission and
the ledger bind its SHA-256 for parallel mode. The lane count is 1 through 3;
the account maximum remains 3 including all assignments. Unknown assignments
are refused, with known study lanes identified by broker records and immutable
outstanding manifests. Each lane uses its own supervisor state directory.

N3 and N1 are interleaved in registered seed order. Claims are atomic under the
ledger lock; a second coordinator is excluded by a dispatcher lock. A lane
failure retains its reservation and evidence while healthy lanes can complete.
No automatic rerun follows a failure. Verified tripwires stop future jobs for
that arm; already submitted jobs retain their evidence.

With total training liability `W`, largest indivisible job `M`, and `L` lanes,
the prospective training elapsed bound is `W/L + (1 - 1/L)*M`. Acceptance and
Mac evaluation are added serially. This bounds dynamic list scheduling rather
than assuming each long job can be divided across lanes. Consumed wall time is
never divided or reset; the ledger checks the remaining forecast before claims.

The existing timing fixture gives these **offline primary planning bounds**,
including acceptance. These are forecasts, not measured parallel throughput or
authorization to allocate:

| Lanes | Units | GPU allocation hours | Mac worker-hours | Elapsed hours |
| --- | ---: | ---: | ---: | ---: |
| 1 | 38.3333333333 | 19.1666666667 | 15.8656728896 | 27.0995031115 |
| 2 | 38.3333333333 | 19.1666666667 | 15.8656728896 | 19.6911697781 |
| 3 | 38.3333333333 | 19.1666666667 | 15.8656728896 | 17.2217253337 |

Usable caps remain 48 units / 20 GPU-hours / 19.2 Mac worker-hours / 38.4 elapsed
hours. Existing failed attempts and delays remain charged and can prevent a
handoff even when this fresh-study table fits. No `research/colab_cli` changes or
new live acceptance are part of this implementation; existing qualified
transport evidence must still match all pinned sources at operator admission.

## Offline preparation after branch push

Run commands from a clean checkout matching the published branch. All new
outputs are create-only; choose another fresh suffix if this namespace exists.
The existing plan and ledger paths below come from the specification and prior
operator documentation, and were not inspected as live evidence.

```bash
set -euo pipefail
cd /Users/josenunez/Projects/ml/snake-dqn-colab-safety
S1_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
S1_REPO=/Users/josenunez/Projects/ml/snake-dqn-colab-safety
S1_BASE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018
S1_OLD="$S1_BASE/settlement-basis-fix-20261019"
S1_NEXT="$S1_BASE/parallel-lanes-20261019-01"
S1_LEDGER="$S1_BASE/admission-run9-20261019/dispatch/ledger-01.json"
S1_ACCEPTANCE="$S1_BASE/admission-public-release-20261019/acceptance/acceptance-1792425613/acceptance-result.json"

mkdir "$S1_NEXT"

"$S1_PY" -m research.s1_mature_20261018.price --lanes 3 \
  --eval-preflight "$S1_REPO/research/s1_mature_20261018/EVAL_PREFLIGHT_FINAL.json" \
  --training-preflight "$S1_REPO/research/s1_mature_20261018/TRAIN_PREFLIGHT_DISPATCH03.json" \
  --acceptance-result "$S1_ACCEPTANCE" --out "$S1_NEXT/pricing.json"

"$S1_PY" -m research.s1_mature_20261018.admission prepare \
  --repo "$S1_REPO" --published-ref origin/t31-colab-safety-20261018 \
  --lanes 3 \
  --parallel-amendment "$S1_REPO/research/s1_mature_20261018/PREREGISTRATION_AMENDMENT_01_PARALLEL.md" \
  --input-delivery public-release \
  --seed-bank /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/SEED_BANK.json \
  --world-bank "$S1_REPO/research/s1_mature_20261018/WORLD_BANK.json" \
  --seed-audit "$S1_REPO/research/s1_mature_20261018/SEED_AUDIT.json" \
  --pricing "$S1_NEXT/pricing.json" --artifact-root "$S1_NEXT/production" \
  --manifest-dir "$S1_NEXT/jobs" --out "$S1_NEXT/plan.json"

"$S1_PY" -m research.s1_mature_20261018.admission preflight-inputs \
  --repo "$S1_REPO" --plan "$S1_NEXT/plan.json" \
  --out "$S1_NEXT/release-input-preflight.json"
```

`preflight-inputs` reads the public input assets. It neither allocates nor
publishes. Offline preparation does not establish the quiescence needed for
continuation, and must never be used to overwrite the running dispatch's files.

## Conditional operator handoff and execution

Execute the following only if dispatch-04 **has exited naturally before study
completion**, every reservation has been reconciled, and the operator has
independently verified the exit. Do not run these while dispatch-04 continues.
The exit flag records the operator's observation; software cannot infer it from
an empty account or settled job. The first command makes only read-only account
requests through the supervisor and writes new handoff evidence. It does not
allocate, dispatch, or stop a runtime. The second command launches remaining
work and belongs to the authorized operator.

Continue in the same shell as the preparation commands:

```bash
"$S1_PY" -m research.s1_mature_20261018.prepare_parallel_handoff \
  --repo "$S1_REPO" --plan "$S1_NEXT/plan.json" \
  --previous-plan "$S1_OLD/plan.json" --ledger "$S1_LEDGER" \
  --colab /Users/josenunez/.local/bin/colab \
  --account-config "$S1_BASE/admission-run9-20261019/dispatch/account-01/sessions.json" \
  --zero-snapshot "$S1_NEXT/handoff-zero-account.json" \
  --legacy-dispatcher-exited --out "$S1_NEXT/handoff.json"

"$S1_REPO/research/s1_mature_20261018/launch.sh" \
  --lanes 3 --previous-plan "$S1_OLD/plan.json" \
  --parallel-handoff "$S1_NEXT/handoff.json" \
  --handoff-snapshot "$S1_NEXT/handoff-zero-account.json" \
  --plan "$S1_NEXT/plan.json" --ledger "$S1_LEDGER" \
  --acceptance-review "$S1_BASE/admission-public-release-20261019/acceptance-review.json" \
  --snapshots "$S1_NEXT/dispatch/snapshots" \
  --state-root "$S1_NEXT/dispatch/supervisor" \
  --artifact-root "$S1_NEXT/production" \
  --supervisor "$S1_REPO/research/colab_cli/launch.sh" \
  --colab /Users/josenunez/.local/bin/colab \
  --account-config "$S1_BASE/admission-run9-20261019/dispatch/account-01/sessions.json"
```

Launch promptly after handoff creation: stale account evidence is refused. A
changed ledger or plan also invalidates the handoff. Never modify evidence to
make admission succeed. If the remaining worst case no longer fits the original
window, leave the study blocked for the operator rather than resetting its clock.

## Local verification

The final S1 suite passed **216 tests in 31.59 seconds** with CPU and single-thread
OpenMP/BLAS settings:

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_s1_*.py
```

The unchanged CLI suite passed **255 tests in 292.70 seconds** using six
bounded offline workers and the same environment:

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 6 tests/test_colab_cli*.py
```

The final CLI log is `/tmp/s1-parallel-final-cli-pytest.log`, with exit status
`0` in `/tmp/s1-parallel-final-cli-pytest.exit`. The initial serial combined
run was interrupted after 292 passing tests because intentional fake-client
acknowledgement stalls used real wall time; it is not counted as a completed
gate. The final CLI run retains the original tests and their timeout semantics.

Black (100 columns), isort (Black profile), flake8 and `git diff --check` passed
for the changed Python files. The repository's `make lint` also passed for
all 147 existing source/backend Python files. Real CLI `--help` entrypoints for
admission, pricing, dispatch and handoff preparation passed without account
access. Handoff tests execute the real preparation CLI with a fake account
snapshot. Scheduler tests include a three-thread barrier, unique interleaved
claims, arm halts, failure fencing, and original-final recovery after handoff.

No live throughput, assignment disappearance, dispatch-04 exit or production
handoff was tested. Those remain operator observations and cannot be inferred
from these offline checks.
