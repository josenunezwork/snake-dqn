# OBS — Implement a corrected versioned raster producer

**Status:** PLANNED. **Earliest wave:** 1. **Role:** builder. **Kind:** implementation.

**Dependencies:** C0

Correct observation precedence while preserving legacy v2 inference and current network capacity.

Read [decisions and wave plan](../README.md), [agent runbook](../agent_runbook.md), and [dispatch.json](../dispatch.json) before starting. You are not alone: edit only this package's claimed files and inherit accepted dependency commits.

## Exclusive write ownership

- `src/simd_env/featurizer.py`
- `src/simd_env/gpu_featurizer.py`
- `src/simd_env/live_adapter.py`
- `tests/test_featurizer.py`
- `tests/test_gpu_featurizer.py`
- `tests/test_obs_parity.py`
- `src/training/raster_augmentation.py` — new
- `tests/test_raster_augmentation.py` — new
- `tests/fixtures/raster31v2_golden.npz` — new

## Implementation / execution steps

1. Add explicit contract dispatch. Freeze v2 observation bytes with checked-in small synthetic golden fixtures; v3 uses the documented deterministic priority reduction.
2. Reject out-of-world predicted cells and preserve actual walls. Test every type pair, reverse order, equal-type max-value tie and enemy permutation against an independent oracle.
3. Pass normalization fields explicitly through both adapters. V3 retains all 26 scalars, tensor shapes and RasterDuelingNetwork topology.
4. Expose legacy flip separately. V3 initial augmentation is none; do not silently zero coordinates, claim legacy flip physical symmetry, or edit pqn_trainer.py.

## Acceptance evidence

- V2 golden bytes match; v3 body/prediction and wall/prediction fixtures produce the correct cell codes independent of source order.
- NumPy and Torch CPU outputs agree with the oracle. Actual CUDA validation is separately required if CUDA is later claimed; Mac MPS uses NumPy observations.
- Non-default horizon/length/starvation normalization propagates. Unsupported circular raster geometry fails instead of painting rectangular walls.

Focused target files below include new files this package creates. Commands and flags described in the steps are implementation requirements, not claims that they already exist.

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_featurizer.py \
  tests/test_gpu_featurizer.py \
  tests/test_obs_parity.py \
  tests/test_raster_augmentation.py
```

## Compatibility and rollback

Route old checkpoints through frozen v2. Never rewrite their metadata or claim source-order repairs are byte compatible under v2.

## Handoff

Write command/result logs, source/config/checkpoint hashes, changed files, commit SHA, unresolved risks and interface changes to `<EVIDENCE_ROOT>/packages/obs/`. The root resolves EVIDENCE_ROOT to an absolute durable path outside disposable worktrees before dispatch. Obtain a root-issued test/compute slot before running commands. Return absolute artifact paths and hashes; root verifies the durable manifest before accepting the dependency or cleaning a worktree. A passing narrow test does not waive an upstream gate or promote a model.

Audit source: [expanded state/system review](../../../state_and_system_review_2026-09-06.md).
