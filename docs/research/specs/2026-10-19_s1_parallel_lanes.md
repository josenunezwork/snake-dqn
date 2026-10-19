# Spec: S1 parallel L4 lanes (pre-data operational amendment)

Owner: Claude (plan/review). Implementer: Sol. Date: 2026-10-19 PT.

## Owner instruction

Owner, 2026-10-19 ~10:25 PT: "how many can u run thru colab cli at once do as much as u need to work
faster". The account-wide maximum is three Colab assignments (supervisor-enforced). The approved
pre-registration (`research/s1_mature_20261018/PREREGISTRATION.md` lines ~14, 60, 165-166) says one L4
run at a time / runs are serialized / no parallel L4 study run. This spec implements an owner-approved
**operational amendment, pre-data**: no S1 training metric has been read by anyone. Science is unchanged:
arms, seed blocks, 8,000,000 learner decisions, B8 initializer, recipe, metrics, analysis, decision rule,
caps, operational-rerun rule, Mac evaluation.

## Required outcome

1. `PREREGISTRATION_AMENDMENT_01_PARALLEL.md` (new file; never edit the approved prereg bytes): states the
   change (up to 3 concurrent L4 lanes, account-wide max 3 including any other assignment), why it cannot
   affect results (independent seeds, fixed decisions, no selection), the owner quote and time, that no
   efficacy data had been observed, and that units/GPU-h/Mac caps are unchanged while the elapsed bound
   is recomputed for lanes. Admission must require this amendment file (hash-bound) for parallel mode.
2. `run.py`: a `--lanes N` (1..3) mode. Each lane runs the existing per-job path (reserve -> fresh zero
   account snapshot logic adapted to "assignments <= allowed" -> supervisor run -> settle) independently.
   Keep the ledger the single source of truth with its file lock; the ledger must admit concurrent
   outstanding reservations only when the sum of outstanding liabilities + spent + remaining planned fits
   every usable cap. Fresh account snapshots must account for the study's own live lanes (known endpoints)
   and refuse unknown assignments. Interleave arms (e.g. n3/n1 alternating) so a tripwire or cap stop does
   not leave one arm with zero runs. A lane failure must not kill healthy lanes; tripwire/KILL semantics per
   prereg still apply per arm.
3. The currently running job `s1-mature-n3-223213833` (dispatch-04, tmux `s1-dispatch-04`, plan
   `/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/settlement-basis-fix-20261019/plan.json`,
   ledger `.../admission-run9-20261019/dispatch/ledger-01.json`) is live and outstanding. Design a safe way
   for Claude to start 2 additional lanes NOW for the next jobs while dispatch-04 continues serially
   (e.g. `--lanes 2 --exclude-running <job>` on the same ledger), without the serial dispatcher and the
   new lanes ever both picking the same job. If that cannot be made safe, say so and give the safe
   alternative (e.g. wait for job 1 to settle, then `--lanes 3`).
4. Pricing/admission: recompute the elapsed upper bound for lanes; all other bounds unchanged.
5. Tests: concurrent reserve under the lock, cap refusal with 3 outstanding, unknown-assignment refusal,
   lane-failure isolation, no double pick, amendment required. All `tests/test_s1_*.py` and
   `tests/test_colab_cli*.py` green; Black/isort/flake8. Avoid `research/colab_cli` changes (they force a
   new acceptance run); if one is unavoidable, say why.
6. Final message: exact operator commands in order, and anything unverified.

## Rules

Work in `/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (branch `t31-colab-safety-20261018`);
commit and push when green. No live Colab, no dispatch, no tmux, no publishing. Do not disturb the
running dispatch-04 or its files.
