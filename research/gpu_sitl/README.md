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
  - `margin_loss_gradient_check(b8_checkpoint, ...)`: Single backward pass of
    $\max(0, \delta - (Q_\theta(s, a^*) - Q_\theta(s, a_0)))$ with $\delta = 0.0632$ on unfrozen
    `Ego2sNet` initialized from `b8-s3`.
- `seed_check.py`:
  - Deterministic seed derivation and repository disjointness audit for namespace
    `gpu-sitl-preflight-20261109-v1`.
- `preflight.py`:
  - Parity runner comparing `batched_open_stop_h32` on `CTDeviceSim` against CPU
    `research/watch_replay/open_stop.py:open_stop_batch` across `E = 4` natural Watch6 v2 worlds ×
    `32` hero ticks (`128` roots, `768` branch certificates) plus long-hero (`L >= 800`) Watch6 v2
    probe states constructed via `src/simd_env/grid_scenarios.py:inject_serpentines`.

## Usage

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
nice -n 15 ./venv/bin/python -m research.gpu_sitl.preflight
```
