# S1 settlement charge basis repair — 2026-10-19

This restores the existing README budget rule. Both pre-training failure
settlements charge `max(observed account debit, priced ceiling * allocation hours)`.
The ceiling comes from the hash-bound reserved manifest. GPU allocation hours
and elapsed hours use the same terminal receipt interval. The supervisor's
`elapsed_seconds` conservatively spans admission before the allocation request,
verified backend disappearance, and the final usage read. The shorter last
account cross-check, observed provider rate, trainer runtime, and human delay
before settlement do not replace that interval.

| Attempt | Receipt seconds | Observed debit | Required charge at 2 units/hour |
| --- | ---: | ---: | ---: |
| dispatch-02 input staging | 496.2047262920096 | 0.19000000000005457 | 0.27566929238444976 |
| dispatch-03 qualification | 339.0773167080042 | 0.04000000000019077 | 0.18837628706000234 |

The dispatch-02 shortfall is **0.08566929238439519 units**. The new
`ledger correct-staging-charge` command appends an evidence-bound correction event
and includes it in spent/projected totals. It preserves the original settlement,
earlier events, study start, planned/outstanding jobs, seed and rerun eligibility.
It accepts only an existing staging settlement with a positive unit-only
shortfall, recomputes the amount, checks the original reservation bound and study
caps, and rejects duplicates. Every later status/admission revalidates the review,
original settlement identity, receipt, manifest, and correction amount.

## Create-only operator artifacts

Directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/dispatch03-cuda-fix-20261019/`

- `staging-charge-correction-review.json`: binds the original dispatch-02 settlement
  by canonical JSON SHA-256, its manifest, receipt and staging review, the corrected
  total, and the additional charge.
- `correct-staging-charge.cmd`: ready local correction command.
- `qualification-failure-review-b.json`: sibling of the original qualification
  review, with only `actual_reconciled_spend.units` corrected.
- `settle-04b.cmd`: uses the sibling review and absolute project interpreter.

The original `qualification-failure-review.json` and `settle-04.cmd` remain
unchanged. Earlier status documents' debit-only amounts are historical and are
superseded by this accounting explanation. No acceptance rerun is needed for
this change: `research/colab_cli` is unchanged.

## Claude commands in order

These are operator handoff commands, not executed live by this repair. Pricing
and preparation use a fresh output namespace; do not overwrite an existing file.
Run the public-input preflight before continuation. Continuation performs the
existing current-source, history, acceptance, account and remaining-budget gates.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-colab-safety
S1_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
S1_REPO=/Users/josenunez/Projects/ml/snake-dqn-colab-safety
S1_BASE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018
S1_OLD="$S1_BASE/admission-public-release-20261019"
S1_FIX="$S1_BASE/dispatch03-cuda-fix-20261019"
S1_NEXT="$S1_BASE/settlement-basis-fix-20261019"
S1_LEDGER="$S1_BASE/admission-run9-20261019/dispatch/ledger-01.json"
S1_ACCEPTANCE="$S1_OLD/acceptance/acceptance-1792425613/acceptance-result.json"

# 1. Correction: append the dispatch-02 shortfall, preserving past events.
bash "$S1_FIX/correct-staging-charge.cmd"

# 2. Settle: charge dispatch-03 at the priced ceiling and restore the same job.
bash "$S1_FIX/settle-04b.cmd"

# 3. Price: fresh output, existing qualified transport acceptance.
"$S1_PY" -m research.s1_mature_20261018.price \
  --eval-preflight "$S1_REPO/research/s1_mature_20261018/EVAL_PREFLIGHT_FINAL.json" \
  --training-preflight "$S1_REPO/research/s1_mature_20261018/TRAIN_PREFLIGHT_DISPATCH03.json" \
  --acceptance-result "$S1_ACCEPTANCE" --out "$S1_NEXT/pricing.json"

# 4. Prepare: pin the published branch and retain original science/input identities.
"$S1_PY" -m research.s1_mature_20261018.admission prepare \
  --repo "$S1_REPO" --published-ref origin/t31-colab-safety-20261018 \
  --input-delivery public-release \
  --seed-bank /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/SEED_BANK.json \
  --world-bank "$S1_REPO/research/s1_mature_20261018/WORLD_BANK.json" \
  --seed-audit "$S1_REPO/research/s1_mature_20261018/SEED_AUDIT.json" \
  --pricing "$S1_NEXT/pricing.json" --artifact-root "$S1_NEXT/production" \
  --manifest-dir "$S1_NEXT/jobs" --out "$S1_NEXT/plan.json"
"$S1_PY" -m research.s1_mature_20261018.admission preflight-inputs \
  --repo "$S1_REPO" --plan "$S1_NEXT/plan.json" \
  --out "$S1_NEXT/release-input-preflight.json"

# 5. Continue: Claude/operator only, after reviewing the preceding evidence.
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

## Offline verification

The real correction CLI, qualification settlement CLI, and persisted `status`
all passed in order against an isolated temporary copy of ledger-01. The original
staging settlement and existing events compared equal, dispatch-03 charged
0.18837628706000234 units, and no job remained outstanding. Combined spent units
were 1.793265579444452. The live ledger was unchanged, SHA-256
`04b536f6be136681cfcbf648087d4bbd83e2ca2a1ca173e9e29a310a010849a3`.

No live Colab, dispatch, tmux, asset publication, or live ledger mutation was
performed. This repair changes accounting and provides an operator sequence;
it does not authorize compute or supply new training/qualification evidence.

The unchanged real `price` entrypoint passed using `EVAL_PREFLIGHT_FINAL.json`,
`TRAIN_PREFLIGHT_DISPATCH03.json`, and the retained final-only acceptance result.
Its create-only verification output is
`dispatch03-cuda-fix-20261019/pricing-settlement-verification.json` under the
artifact root; it is separate from Claude's future `pricing.json`.

Focused ledger suite: **59 passed**. Independent read-only review found no
substantive issues and separately reproduced all 59 passes. Black (100-column),
isort (Black profile), flake8, Bash syntax for both new command files, and
`git diff --check` passed.

Full requested regression command:

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_s1_*.py tests/test_colab_cli*.py -n 4
```

Result: **447 passed in 396.30 seconds**. No failures or skips.
