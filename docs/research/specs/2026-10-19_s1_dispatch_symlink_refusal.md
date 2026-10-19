# Spec: S1 dispatch refused pre-allocation on a committed symlink

Owner: Claude (plan/review). Implementer: Sol. Date: 2026-10-19 PT. Owner approved S1 dispatch.

## Evidence

Dispatch 01 (07:24 PT), command in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-run9-20261019/dispatch/dispatch-01.cmd`,
log `dispatch-01.log`:

- `research/colab_cli/source.py:144` raised `symlink prohibited in fallback source:
  research/s1_mature_20261018/S1_SEED_AUDIT.json` (a committed symlink -> `SEED_AUDIT.json`,
  added in 9a18e661). The supervisor exited 1 before any `colab` call: no allocation, no
  session (`colab sessions` empty, usage rate 0), `supervisor-01/.../` contains no files.
- `ledger-01.json` was initialized (acceptance settled 0.958 units) and now holds an
  **outstanding reservation** for `s1-mature-n3-223213833` (2.1667 units / 1.0833 GPU-h)
  for a job that never allocated.

## Constraints

- **Do not modify anything under `research/colab_cli`.** The final-only acceptance PASS
  (run 9) is bound to those bytes; the review at
  `.../dispatch/acceptance-review.json` passes `ledger._review_acceptance` today.
- Never delete, reset or rewrite `ledger-01.json`; settle through ledger APIs with evidence.
- Do not change the approved pre-registration's science.

## Required outcome

1. Remove the symlink from the S1 package (replace with a regular file only if anything
   reads it; grep found no `.py` reference). Audit the whole tree that the fallback source
   packs for any other symlink/non-blob and fix those too.
2. Add an offline packaging preflight to S1 admission/dispatch: build the exact fallback
   source the supervisor will build (call the same `research/colab_cli/source.py` function,
   no Colab) for every prepared manifest and refuse before ledger reservation if it fails.
   Reserving a job must happen only after this check passes.
3. Settle the stranded `s1-mature-n3-223213833` reservation in `ledger-01.json` with a
   zero-allocation, evidence-bound record (e.g. via `reconcile_operational_failure` or a
   new narrowly scoped method), proving no `new` call: empty supervisor state dir, the
   before-snapshot, and a fresh zero-assignment snapshot. It must not consume the arm's one
   operational replacement unless the pre-registration says it must; state which in the
   commit message with the prereg line you relied on.
4. Re-prepare the plan for the new published commit (offline), and make continuing the SAME
   ledger possible for the new plan (the job ids/seeds are unchanged), or explain exactly why
   it cannot and what the minimal safe alternative is. Write the refreshed admission status.
5. Tests for: symlink refusal pre-reservation, stranded-reservation settlement, continuation.
   All `tests/test_colab_cli*.py` + `tests/test_s1_*.py` green; Black/isort/flake8.
6. Final message: the exact dispatch/continue command to run next (absolute paths), and
   anything unverified.

## Rules

Work in `/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (branch
`t31-colab-safety-20261018`); commit and push when green. No live Colab, no dispatch, no tmux.
You may write ledger-01 only through the ledger API and only if tests prove the path;
otherwise leave the exact command for Claude to run.
