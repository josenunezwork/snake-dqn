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
- Function- and module-scoped fixtures restore the global game configuration.
  Both scopes are necessary: module fixtures initialize before per-test fixtures.
  This repairs the reproduced 61-input research configuration leak into collected
  58-input checkpoint fixtures; the expanded regression sequence passed 328 tests.
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
- Portable checkpoint and roster fixtures preserve real evaluator construction,
  mask parity, ordered roster rotation, and a live moving T1 replay/clone trace.
  The affected evaluator shard passed 90 tests with 8 pre-existing opt-in skips.
- Seed fixtures supply local pilot and observed-seed inputs to the real exclusion
  parsers. T12 exercises 5,805 synthetic records and 45 prefix controls through
  the real frozen rule, inventory hashing, and recomputation, using separate
  evidence and immutable artifact copies. A separate test checks production pins
  against committed envelopes. Synthetic data are not release evidence.
- Stock-wait tests isolate their fake clock to the two modules under test, so
  unrelated background-thread sleeps cannot advance it. All 37 tests passed,
  including a regression checking that process-wide sleep remains unchanged.
- SciPy and scikit-learn are declared development dependencies so diagnostic tests
  can run in a clean GitHub checkout.

## Hosted-only follow-up repairs

The first integrated local suite passed 6,945 tests (15 existing skips and 101
slow deselections). An earlier hosted Mac run then exposed additional environment
differences: Torch 2.14.1 was installed where the local and qualified parity tests
use 2.9.1, the runner's available RAM was below production watchdog floors, and
subprocess smoke tests could not inherit in-process artifact fixtures.

Development requirements now pin Torch 2.9.1. CI prints its build configuration
and runs the unchanged opponent, MPS compile, learner, and stacked-learner parity
assertions before the full suite. Test supervision helpers select a memory floor
explicitly and test low-memory refusal separately; tiny observation fixtures
provide deterministic memory readings while retaining resource-error tests.
The bank-only CLI smoke passes its scripted identity fixture through a temporary
child-interpreter startup hook. Production identity and resource guards remain
unchanged. Local-agent tests now require authenticated health readiness, capture
startup logs, and fail explicitly on child exit or deadline instead of falling
through a socket-poll loop. The two affected test files passed all 108 tests
locally; the unchanged Torch group passed 86 selected tests and its exact CI
preflight passed all 8 tests.

## Hosted environment and metadata portability

The Python quality job uses `macos-15` ARM64, matching the platform authority for
historical numerical plans. Linux and Darwin produced a nominal probability
of `1.941938163774193e-06` and `1.941938163829704e-06`, respectively, from the same
boundary `4.617501629587915`. The path uses stdlib `NormalDist` and platform
`erf`/`erfc`, not SciPy; pinning SciPy cannot repair this last-bit SHA difference.
Every original plan hash and production refusal guard remains intact. A 76-test
numerical preflight runs before the full suite. The frontend stays on Ubuntu.
[GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
lists this standard ARM64 runner. Full Git history is checked out for source-parity
tests and `rg` is explicitly ensured for artifact scanners.

The macOS 26 attempt passed the frozen-plan preflight but failed six unchanged
Torch numerical tests even with the identical 2.9.1 wheel used locally.
[GitHub runner issue 14380](https://github.com/actions/runner-images/issues/14380)
records a separately reproduced OS Metal numerical regression and validates
macOS 15 as a workaround that still executes MPS. The workflow therefore uses
macOS 15 and retains all original CPU bitwise and MPS assertions, including the
compiled actor availability assertion. This is a runner qualification choice;
no parity tolerances or MPS assertions were changed to accommodate macOS 26.
The macOS 15 preflight passed all four MPS cases without skips and three of four
CPU cases; the remaining uneven CPU group exposed the padded-row assumption.
The optional merged CPU forward now receives explicit original row counts and
uses each uneven group's original linear-layer shape, preserving its bitwise
contract rather than inferring padding from zero values. Balanced groups retain
the stacked path. This is a correctness repair to optional inference code, not a
new performance or serving qualification.

T10's prefix-boundary writer now records `directory_birthtime_utc_approx: null`
when the host filesystem exposes no birth time, rather than inventing one from
modification time or failing on Linux. The timestamp is auxiliary metadata;
receipt hashes, horizon and branch checks remain enforced. A regression test
simulates missing birth time. This creates new research-driver source bytes;
all historical T10 receipts and source-bound evidence remain unchanged.

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
