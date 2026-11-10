# Step B1a (`M0-GPU-SITL-Preflight`): Batched `open-stop-h32` Certificate on `CTDeviceSim` / `DeviceSim`

This package implements Step B1a (`M0-GPU-SITL-Preflight`, `0 u` Mac preflight) from
`docs/research/advice/2026-11-09_live_search_reopen_registration_argon.md` (§3.2).

## Contents

- `open_stop_gpu.py`:
  - `evaluate_terminal_openness_device(sim)`: Batched device-native (`cpu` or `cuda`) terminal
    openness evaluator mirroring `research/watch_replay/open_stop.py:evaluate_terminal_openness`
    (`alive` at `H = 32`, `v8_best_safe_area >= min(max(4, L), 160)`, and `max_pocket_r >= L` for
    `L >= 160`).
  - `batched_open_stop_h32(sim, policy, ...)`: Expands `R` prepared root worlds on
    `CTDeviceSim` / `DeviceSim` to `6 * R` branches via `repeat_interleave(6)`, rekeys per-root
    action-independent CRN suffix seeds (`branch_suffix_seed`), steps all 6 root hero actions on
    step 1 and `b8-s3` hero + `5x frp3-s12 + v8` opponent continuations through `H = 32` (with
    optional dead-hero branch compaction), and evaluates terminal openness certificates.
  - `freeze_ego2s_value_head(net)` and `margin_loss_gradient_check(b8_checkpoint, ...)`:
    Freezes `value.weight` and `value.bias` (`requires_grad_(False)`) and runs the registered
    rescue-root hinge $\max(0, \delta - (Q_\theta(s, a^*) - \max_{a \in \text{legal}\setminus\text{safe}} Q_\theta(s, a)))$
    ($\delta = 0.3002$) on roots with both safe and unsafe legal actions, asserting finite
    non-zero gradients on all 24 non-value tensors.
- `seed_check.py`:
  - Deterministic seed derivation and repository disjointness audit for namespace
    `gpu-sitl-preflight-20261109-v1` (`0..3` natural, `4..7` long-hero, `8..39` 32-root mature
    extension cohort).
- `preflight.py`:
  - Step B1a and Step B1a-ext parity runner (`--mode {b1a,freeze-cohort,ext-shard,ext-finalize}`),
    comparing `batched_open_stop_h32` (`cpu_fast_flood=False`) on `CTDeviceSim` against CPU
    `research/watch_replay/open_stop.py:open_stop_batch` under both matched-RNG
    (`SplitMixEnvRngProxy`) and cross-RNG (CPython MT19937 vs SplitMix64).

## Usage

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
nice -n 15 ./venv/bin/python -m research.gpu_sitl.preflight
```
