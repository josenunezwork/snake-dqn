# T14 CI closure validation — 2026-10-17

This sibling note records CI maintenance under the
[T14 closure spec](specs/t14_close_campaign_spec_2026-10-17.md). No research run,
serving timing qualification, or promotion was performed.

## Source provenance

Commit `fe69d20864c344612030cfde4068f6e63edb9418` repairs pytest/tool configuration
and lint failures. Independent read-only review against parent
`3cbbbdc0af318fde4ba8ab7e000634c9bd99bbe1` found:

- `src/core/runtime_contract.py` and `web/backend/serving_speed.py` have identical
  Python ASTs before and after formatting.
- All five changed Python files have equivalent normalized ASTs after removing
  docstrings and merging/sorting same-module imports. The other three files are
  `src/scripts/tournament_eval.py`, `src/training/rollout_policies.py`, and
  `web/backend/safety_veto_serving.py`.
- The seven pinned veto sources (`safety_veto.py`, and `_v3` through `_v8`) are
  byte-identical; the V5/V7/V8 serving source pins still match. The v8 SHA remains
  `faf3695fa9d60550f1f681c04b8c8aba0447fecaa34cd75f16f75efe91e4ac05`.

These edits nevertheless create new source bytes. The historical T12B auditor
compares current files with its frozen intent and will report those changes.
Original intents, receipts, manifests, and results remain untouched. Static AST
comparisons do not constitute a new timing qualification or make the current tree
byte-identical to the historical T12/T12B qualification snapshot.

## CI tool configuration

Pytest is constrained to version 9 and the ini minimum is 9, matching its strict
configuration keys. Black and isort share the existing 100-character line width.
Isort retains Black-style trailing-comma imports; version 8 avoids the version-5
invalid import rewrite observed for an aliased import with that setting. No lint
rule is disabled and the workflow still selects `pytest -q -m "not slow"`.

## Test portability and isolation

The test repairs preserve the workflow selection and add no skip markers:

- The plan argument test now checks every committed plan, allowing only an explicit
  per-run initial-checkpoint substitution relative to the old argument list.
- A function-scoped fixture restores the global game configuration after each test.
  This repairs the reproduced 61-input research configuration leak into collected
  58-input checkpoint fixtures; the previously failing sequence passed 252 tests.
- The pre-M6 actor test still checks legacy messages and ring rows bitwise. It
  separately validates the new length-bucket telemetry, including exact accounting
  and equality between default and explicit-zero configurations; both cases passed.
- T2/T4c tests run the real H5000 evaluator with deterministic local checkpoints,
  comparing observer/no-observer outputs and requiring a terminal death. T6 uses
  small real audit fixtures plus separate historical checks: the committed snapshot
  roster and the reconstructed 1,200 collection seeds retain count/digest bindings
  and exclusion from rounds 72/73. Production snapshot and collection constants are
  also checked against the committed receipt. All 23 portable tests passed, followed
  by 8 T6 tests after the final constant assertions. These are test fixtures, not
  new campaign evidence.
- SciPy and scikit-learn are declared development dependencies so diagnostic tests
  can run in a clean GitHub checkout.

## Verification procedure

The workflow remains [CI](../../.github/workflows/ci.yml). Its exact lint and test
commands are:

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
black --check --line-length 100 src web/backend
isort --profile black --check-only src web/backend
flake8 src web/backend
pytest -q -m "not slow"
```

Frontend commands, from `web/frontend`, are `npm ci`, `npm test`, and
`npm run build`; local verification passed all 175 frontend tests and the build.
All three lint commands passed locally. Full Python suite and per-commit hosted
results are checked through `gh run list` and `gh run view`; the final run URL and
published SHA are reported in the task completion message.

Local dependencies were installed in a task-owned temporary Python 3.12 environment,
leaving the shared project venv untouched. For the full local rerun, inherited
`CONDA_PREFIX`/`CONDA_DEFAULT_ENV` were unset and `CXX=/usr/bin/clang++` plus a
task-owned `TORCHINDUCTOR_CACHE_DIR` selected Apple's compiler without Conda's
unresolvable `@rpath/libc++.1.dylib`. This is local environment repair, not a skipped
compile test or a workflow relaxation. An exploratory compiler-only invocation
accidentally included slow tests: 27 passed before its last slow test was stopped;
that interrupted invocation is not claimed as a full suite pass. Only that owned
process tree was stopped. No tmux session or pre-existing process was modified.
