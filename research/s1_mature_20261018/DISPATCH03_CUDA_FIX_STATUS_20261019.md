# S1 dispatch-03 qualification repair and operator handoff

Implemented locally under the dispatch-03 spec. No live Colab, S1 dispatch,
publication, tmux operation, or live-ledger settlement was performed.

## Diagnosis and repair

`q0b_cuda_fixtures.run_cuda_pool()` imports `tests.test_t22_frozen_pool`, which
imports `pytest` before any CUDA tensor operation. The ordinary runtime
`requirements.txt` does not install pytest. Blocking that import on the Mac
reproduced `ModuleNotFoundError: pytest` before CUDA. Dispatch-03's exact original
exception remains unavailable because its VM receipt was not exported; this is
a reproduced packaging defect consistent with that failure, not recovered VM
error evidence.

S1 now selects `research/s1_mature_20261018/requirements-runtime.txt`, which
includes the ordinary requirements and pytest. Its explicit sparse source list
includes retained T22/T24 tests and oracles, Q0/Q1 and T23 parity adapters,
the trainer's M2 guard imports, the deployment config, and the serving modules
that supply the frozen pool's v8 source pins. Those source files are copied;
serving behavior and pins are unchanged.

Final offline admission also caught the broad research directories pulling
historical result payloads into the sparse archive, exceeding the conservative
staging estimate. The runtime inventory now names the needed S1, Q0/Q1, T23 and
identity files explicitly, excluding those result/bank/reference payloads.
The registered bank and Mac reference still arrive as hash-bound inputs. This
follow-up changes only the fallback inventory; `train.py`, `train_m3c.py`, the
training command and the counter-smoke evidence remain unchanged.
The narrowed local snapshot contains 283 files / 4,901,436 unpacked bytes and a
1,224,101-byte fallback archive. Its Contents forecast is 236.70 seconds and
public-input forecast is 154.31 seconds, each below its separate 300-second cap
(391.01 seconds combined). `local-smoke-02/staging-preflight.json` records the
priced staging proof; the published candidate records its own exact inventory.

The Mac packaging preflight now runs the real `--cpu-preflight` fixture
entrypoint and the existing Q1 `--verify-reference` comparison in both a full
commit archive (the GitHub-equivalent layout) and the exact supervisor sparse
fallback. It stages the hash-verified frozen inputs, extracts the league, creates
the `fixture-inputs/checkpoints` symlink and `init/B8_s3_final.pth`, and uses the
same parity environment with one CPU thread. It runs once per distinct pinned
commit/input identity before any ledger/account operation. A missing fixture,
dependency, receipt, or failed comparison refuses admission. The explicit label
is `PASS_CPU_ONLY`, with `cuda_qualified: false`; L4 still runs real CUDA checks.

Every pre-training qualification subprocess now logs its failure receipt/error
and completes a small immutable diagnostic export before propagating failure.
Large parity receipts are parsed under a separate 64 MiB input limit and reduced
to a diagnostic of at most 64 KiB, retaining the source hash/size and bounded
error/comparison. The existing supervisor transfers terminal exports before
rejecting a nonzero return code. `research/colab_cli` was unchanged, so the current
final-only acceptance remains reusable if its existing source/hash gates pass.

Pricing still requires a current adapter-source-bound training preflight.
`TRAIN_PREFLIGHT_DISPATCH03.json` is a new sibling receipt from the tiny local
CPU fixture; the historical `TRAIN_PREFLIGHT_FINAL_ONLY.json` bytes are preserved.
The fixture uses its smoke namespace, reduced replay and 12,000-decision target,
not the production S1 training plan. Its counter and final-artifact checks retain
the existing rules, and it supplies no L4 throughput or CUDA qualification.

## Settlement evidence

The local review and ready command are in:

`/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/dispatch03-cuda-fix-20261019/`

- `qualification-failure-review.json`: binds the dispatch-03 manifest, receipt,
  failed terminal and final poll; no training metrics, progress or exports.
- `settle-04.cmd`: local-only settlement command for Claude.

Reconciled spend is **0.04000000000019077 units** and **339.0773167080042 seconds**
(0.09418814353000117 allocation hours). The real settlement CLI and subsequent
persisted `status` passed against an isolated copy of `ledger-01.json`: the same
N3 job returned to planned, no job remained outstanding, and `retry-n3` remained
unused. The live ledger remained unchanged, SHA-256
`04b536f6be136681cfcbf648087d4bbd83e2ca2a1ca173e9e29a310a010849a3`.

Settlement requires verified stop, reconciled spend, matching terminal/poll
identity, a qualification-failure traceback, and no training progress or local
training artifacts. It checks all retained polls and their embedded log records.
It preserves the prior staging settlement and selects the latest failure by its
settlement time when continuing the same job, independent of JSON key sorting.

## Offline preparation and next commands (Claude only)

The final published-commit candidate is prepared create-only under
`$S1_NEXT/candidate/plan.json`. It retains the original seeds, science and input
hashes. It is deliberately pending fresh pricing and failure settlement, so it
cannot be dispatched. Use a fresh directory for any later preparation.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-colab-safety
S1_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
S1_REPO=/Users/josenunez/Projects/ml/snake-dqn-colab-safety
S1_BASE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018
S1_OLD="$S1_BASE/admission-public-release-20261019"
S1_NEXT="$S1_BASE/dispatch03-cuda-fix-20261019"
S1_LEDGER="$S1_BASE/admission-run9-20261019/dispatch/ledger-01.json"
S1_ACCEPTANCE="$S1_OLD/acceptance/acceptance-1792425613/acceptance-result.json"

# Apply the prepared, reviewed local settlement. No allocation occurs here.
"$S1_PY" -m research.s1_mature_20261018.settle_pretraining_qualification_failure \
  --ledger "$S1_LEDGER" --job-id s1-mature-n3-223213833 \
  --receipt "$S1_OLD/production/training/s1-mature-n3-223213833/receipt.json" \
  --review "$S1_NEXT/qualification-failure-review.json"

# No acceptance rerun is required by this change: research/colab_cli is unchanged.
# If those transport bytes change later, stop and obtain fresh final-only
# acceptance/review through the existing operator workflow before pricing.

"$S1_PY" -m research.s1_mature_20261018.price \
  --eval-preflight "$S1_REPO/research/s1_mature_20261018/EVAL_PREFLIGHT_FINAL.json" \
  --training-preflight "$S1_REPO/research/s1_mature_20261018/TRAIN_PREFLIGHT_DISPATCH03.json" \
  --acceptance-result "$S1_ACCEPTANCE" --out "$S1_NEXT/pricing.json"

"$S1_PY" -m research.s1_mature_20261018.admission prepare \
  --repo "$S1_REPO" --published-ref origin/t31-colab-safety-20261018 \
  --input-delivery public-release \
  --seed-bank /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/SEED_BANK.json \
  --world-bank "$S1_REPO/research/s1_mature_20261018/WORLD_BANK.json" \
  --seed-audit "$S1_REPO/research/s1_mature_20261018/SEED_AUDIT.json" \
  --pricing "$S1_NEXT/pricing.json" --artifact-root "$S1_NEXT/production" \
  --manifest-dir "$S1_NEXT/jobs" --out "$S1_NEXT/plan.json"

# Hash-check public release bytes on the Mac, without account access.
"$S1_PY" -m research.s1_mature_20261018.admission preflight-inputs \
  --repo "$S1_REPO" --plan "$S1_NEXT/plan.json" \
  --out "$S1_NEXT/release-input-preflight.json"

# Operator-only continuation, after reviewing settlement, pricing and plan.
"$S1_REPO/research/s1_mature_20261018/launch.sh" \
  --continue-staging --previous-plan "$S1_OLD/plan.json" \
  --plan "$S1_NEXT/plan.json" --ledger "$S1_LEDGER" \
  --acceptance-review "$S1_OLD/acceptance-review.json" \
  --snapshots "$S1_NEXT/dispatch/snapshots" \
  --state-root "$S1_NEXT/dispatch/supervisor" \
  --artifact-root "$S1_NEXT/production" \
  --supervisor "$S1_REPO/research/colab_cli/launch.sh" \
  --colab /Users/josenunez/.local/bin/colab \
  --account-config "$S1_BASE/admission-run9-20261019/dispatch/account-01/sessions.json"
```

The continuation still rechecks all historical evidence, current acceptance,
source/input hashes and remaining lifetime/budget caps. Any refusal must be
retained and resolved before allocation; this handoff does not authorize a run.

To regenerate only an offline candidate at a later clean published commit:

```bash
"$S1_PY" -m research.s1_mature_20261018.prepare_release \
  --repo "$S1_REPO" --published-ref origin/t31-colab-safety-20261018 \
  --previous-plan "$S1_OLD/plan.json" \
  --artifact-root "$S1_NEXT/candidate-next/production" \
  --manifest-dir "$S1_NEXT/candidate-next/jobs" --out "$S1_NEXT/candidate-next/plan.json"
```

The local smoke evidence is under `$S1_NEXT/local-smoke-01/`: real retained
fixtures and the existing four-world CPU parity comparison passed in both
source layouts in **138.00 seconds** total. This is packaging/runtime proof,
not new S1 outcome data or CUDA qualification.

## Verification

- Broad local run:
  `SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 venv/bin/python -m pytest -q -n 4 tests/test_colab_cli*.py tests/test_s1_*.py tests/test_q0b_measure.py tests/test_requalify_q0b.py tests/test_t22_random_safe.py tests/test_t22_frozen_pool.py tests/test_t24_v8_bodymap.py`.
  All 348 Colab CLI/affected Q0b/T22/T24 tests passed. The two S1 failures were
  expected stale training-receipt source bindings; the old receipt was preserved.
- Final S1 run after the fresh smoke receipt:
  `SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 venv/bin/python -m pytest -q -n 4 tests/test_s1_*.py`:
  **178 passed in 19.02 seconds**, including the two repaired pricing checks and
  all final diagnostic/settlement regressions. These completed runs cover all
  **526 distinct tests** in the final selected files.
- Tiny real adapter entrypoint: **exit 0, 120.49 seconds**, 12,000 decisions,
  actor exits `[0, 0]`, zero overshoot against bound 47, 11,760 replay rows,
  45 updates, debt 240, reconciled n-step accounting and no ring overruns. The
  final-only archive roundtrip passed; exported archive was 12,180,688 bytes.
- Fresh real `price` entrypoint with `TRAIN_PREFLIGHT_DISPATCH03.json` and the
  retained acceptance result: **PASS**, saved as
  `$S1_NEXT/pricing-local-verification.json` (separate from the next operator run).
- Black `--check --line-length 100`, isort `--check-only --profile black`, and
  flake8 passed on all 15 changed Python files. `git diff --check`, Bash syntax
  on the launchers/settlement command, and both new CLI help commands passed.
- Independent final review found no remaining substantive findings.
