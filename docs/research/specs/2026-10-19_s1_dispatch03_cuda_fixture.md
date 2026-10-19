# Spec: S1 dispatch-03 failed in the pre-training CUDA fixture (t22)

Owner: Claude (plan/review). Implementer: Sol. Date: 2026-10-19 PT.

## Evidence

Dispatch-03 (09:16 PT, commit cd766c63, plan `.../admission-public-release-20261019/plan.json`).
Receipt dir: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/admission-public-release-20261019/production/training/s1-mature-n3-223213833/`
(`receipt.json`: INCOMPLETE, `Refused('remote terminal failed or foreign generation')`, stop_verified,
339 s; `poll-000005.json` has the job log tail). Public-release staging WORKED (inputs verified, source
from GitHub, torch `2.9.1+cu126 12.6 (8, 9)` on L4). Then `train.py:320` ran
`python -m research.q0q1_20261017.q0b_cuda_fixtures --fixture t22 --out .../s1-work/t22-cuda.json`,
which returned 2 (its own "fixture INCOMPLETE" code; the error text is in t22-cuda.json on the VM and was
never exported). No training step ran, no metrics, no exports.

## Required outcome

1. **Diagnose t22.** Reproduce as far as possible on the Mac: run `run_cuda_pool()` / `run_t22()` paths
   with CUDA checks stubbed, in the exact remote layout (repo from GitHub at the commit, `Q0Q1_INPUT_ROOT`
   fixture-inputs with the `checkpoints` symlink to the league dir and `init/B8_s3_final.pth`, env from
   `parity_env`). Look for missing files (e.g. `tests/test_t22_random_safe.py`, league members, anything
   the old Drive-queue Q0b run had that the CLI layout lacks), import errors, dependency gaps on the VM
   (the earlier note said the sparse fallback omits two runtime packages; check the GitHub path too).
   Fix the root cause. If it is genuinely CUDA-only and cannot be reproduced, say so with your best
   hypothesis.
2. **Never fail blind again.** When any pre-training qualification subprocess fails, train.py must print
   its receipt JSON (error field) to the job log and include the receipt in the terminal exports (small,
   within caps) so the Mac receives it.
3. **Settle this attempt** in `ledger-01.json` like the staging failure (actual reconciled spend), as a
   pre-training qualification failure that does NOT consume N3's rerun: no training step, no metrics, no
   efficacy look. Extend the settle method narrowly (a trainer process existed, but it must be shown it
   never passed qualification / never wrote training metrics); refuse if any training metric or export
   exists. Leave a ready settle command for Claude.
4. Before handing back, make the Mac preflight run every pre-training qualification fixture that can run
   without CUDA in the remote layout, so this class of failure is caught before allocation.
5. Re-prepare the plan offline for the new commit; leave exact next commands (settle, acceptance only if
   `research/colab_cli` changed, price, prepare, continue). Tests green (`tests/test_colab_cli*.py`,
   `tests/test_s1_*.py`, plus any q0q1 tests you touch); Black/isort/flake8.

## Rules

Work in `/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (branch `t31-colab-safety-20261018`);
commit and push when green. No live Colab, no dispatch, no tmux, no publishing. Avoid touching
`research/colab_cli` unless required (it forces a new acceptance run). Do not change the pre-registration's
science.
