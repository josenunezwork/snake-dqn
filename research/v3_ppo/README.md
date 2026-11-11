# Game V3.1 PPO Self-Play (`research/v3_ppo`)

From-scratch PPO self-play trainer, throughput/VRAM ruler, deterministic M-of-N evaluator, and Colab CLI packet builder for continuous-heading circular-arena Game V3.1 (`research/game_v3`, default `R = 600`, `N = 24` snakes, `10` discrete actions).

## Key Design Contracts

1. **One `build_obs` per tick (`Uint8RolloutBuffer`):**
   - Each tick writes multi-scale `uint8` ego rasters (`13x64x64 + 13x32x32 + 13x32x32 = 79,872` bytes/step), `8` `float32` scalars, and `10` `bool` action-mask entries straight into `Uint8RolloutBuffer` via prebound zero-copy slice views (`79,938` total bytes/learner-step).
   - Post-rollout `s_T` bootstrap observations are cached and copied into `t = 0` of the next rollout (`1` `build_obs` call per simulation tick in steady state; zero observation rebuilds during PPO epochs).
   - `uint8 -> float / 255.0` conversion happens inside `EgoMultiScaleNet` under `bf16` autocast.
2. **Ego Multi-Scale Actor-Critic & Host-Sync-Free Learner (`EgoMultiScaleNet`, `V3PPOTrainer`):**
   - Three scale-specific residual Conv+GroupNorm+SiLU stems (`s1` `64->4`, `s4` `32->4`, `s16` `32->4`) fused with an `8`-scalar MLP embedding, `apply_recurrent_core` seam (feed-forward in round 1), `10`-action masked categorical policy head, and `255`-bin symlog two-hot value head.
   - Calibrated presets: `S` (`1,008,657` params) and `M` (`4,927,841` params).
   - `channels_last`, `bf16` autocast, `fast_learner_ops`, and `fused=True` `AdamW` are ON by default on CUDA (`--no-channels-last`, `--no-amp` / `--fp32`, `--no-fast-learner-ops` to disable; `--compile` opt-in across `ppo`, `ruler`, `prepare`, and `job.sh`).
   - `net.py` performs zero host synchronizations; `ppo_update` enforces uniform minibatch shapes (dropping partial remainders when `n_valid >= minibatch_size`, or masking padded rows when `n_valid < minibatch_size`) so `cudnn.benchmark` never re-autotunes, and synchronizes scalar diagnostics once per update.
3. **Learners, Episodes & Simultaneous Kill-Credit Rule:**
   - Default `24/24` pure self-play; supports `M`-of-`N` learners via `LearnerIndex` against `food_seeking`, `random_safe`, or `cardinal6_safe` scripted opponents.
   - Life = episode (`max_life_ticks == 0`; `sim.step(actions, truncate=...)` at rollout boundaries bootstraps `V(s_T)` while keeping large snakes alive in the world; death has zero bootstrap).
   - **Simultaneous kill + death:** `GameV3Sim` awards `kill_credit` to a killer even if the killer also dies on the same tick. By design, `compute_step_reward` pays only `-reward_death_penalty` on dying ticks (never rewarding kamikaze trades), whereas `eval.py` counts all `l_acted` `kill_credit` events (including simultaneous kill+death) so reported `kills_per_1k_agent_steps` matches the simulator's exact kill tally.
4. **Two-Tier Atomic Checkpointing, Immutable Exports & `work/`-Only Retention:**
   - **`LIGHT` checkpoints** (`light_u*.pt`: network, optimizer, scaler, config, counters, episode accumulators, timestamp, restart generation, and all RNG streams; no simulator state) are exported to `output/checkpoints/` at every real `light_checkpoint_interval_sec = 1800`s (`30` min) with **no per-session count cap and no interval stretching**, plus at cooperative stop / wall guard.
   - **`FULL` checkpoints** (`light_u*.pt` + whole-world `uint8`-zlib simulator shards `full_u*.sim_shard_*_of_*.pt` each `<= checkpoint_shard_bytes = 256 MiB` with `.sha256`/`.complete` sidecars + `full_u*.index.json` written last) are exported every `full_checkpoint_interval_sec = 7200`s (`2` h) up to `max_periodic_full_exports = 3` periodic publishes per session (sized by `prepare` against the session export cap), **plus always one at cooperative stop / wall guard**. Update-based checkpoint settings (`checkpoint_every_updates`, `full_checkpoint_every_updates`) obey the same session export cap.
   - **Resume choice:** prefers newest verified `FULL` set (bit-identical on CPU including `_clean_step_streak`), switching to a newer verified `LIGHT` checkpoint **only if** the `LIGHT` is newer than the newest `FULL` by more than `light_resume_margin_minutes = 45.0` min, and logs `V3_PPO_RESUME_CHOICE` (with `V3_PPO_LIGHT_RESUME_WARNING` when worlds are re-initialised).
   - **Safety & `work/`-only retention:** completed exports in `output/` are immutable and never rewritten or deleted; `"keep newest N"` (`keep_full_checkpoints = 2`, `keep_light_checkpoints = 4`) prunes **only `work/checkpoints/`**. Cooperative stops mid-rollout or mid-PPO-epoch discard half-stepped weights and publish only clean update-boundary checkpoints; in-session restarts (`max_in_session_restarts = 2`) tag post-restart checkpoints (`_r01`, `_r02`) and never bind a `FULL` index to a stale pre-restart `LIGHT`.
5. **Single-Pass Dual-Matchup Evaluation, Stop Polling & Unsynced Headline Ruler:**
   - Fixed-seed evaluation (`eval_every_minutes = 60.0`, `eval_envs = 192` -> `96` worlds vs `food_seeking` + `96` worlds vs `random_safe` in **one** `GameV3Sim` pass, `eval_learner_slots = 8`, `eval_burnin_ticks = 250`, `eval_ticks = 2000`) evaluates both matchups simultaneously without perturbing training worlds or RNG.
   - Polls the stop flag and wall guard every `16`-tick chunk (`aborted: True` on stop, never logged as a data point), skips starting an evaluation when remaining wall `< 2x` last measured eval wall (or `2x eval_estimated_seconds` before the first), logs per-eval and cumulative eval wall share, and excludes eval wall from training throughput (`sps_scope=train_excl_eval`). Family canary runs `1` default-config evaluation on the target GPU and records its wall seconds in `canary_checks.json`.
   - Completed lives and window-censored survivors are never mixed: reports per-completed-life `mean/median/p95_death_length`, `mean/median/p95_lifetime_ticks`, `mean_completed_episode_return`, separate `censored_lives` / `mean_censored_length`, and survivorship-free rates (`time_averaged_length`, `mass_gained_per_1k_agent_steps`, `deaths_per_1k_agent_steps` with raw `total_deaths`, and `kills_per_1k_agent_steps`).
   - Ruler cells default to `rollout_len = 32`, `warmup_updates = 1`, `timed_updates = 2`, measure headline collection and integrated throughput with per-stage timing syncs **OFF** (`stage_timing=False`), run `1` separate synced pass for `time_split_pct`, precheck cell VRAM before allocation, and hold at most `1` compressed mature world-count state in host RAM at a time.

## Memory & Checkpoint Export Arithmetic (`R = 600, N = 24, T = 32`)

- **Per-rollout learner storage (`E = 128, K = 24, T = 32`):** `98,304 * 79,938 = 7.858 GB` + `0.246 GB` bootstrap cache = `8.104 GB`.
- **Simulator (`GameV3Sim`):** `3.684 GB` persistent (`27.45 MiB/world`), `4.161 GB` peak (`31.00 MiB/world`).
- **Two-phase peak VRAM (max across rollout/learner phase and `192`-world single-pass eval phase):**
  - `L4` (`22.0 GB` budget): `E = 128, K = 24, T = 32` -> `19.832 GB` (`S`) / `20.298 GB` (`M`).
  - `A100` (`40.0 GB` budget): `E = 256, K = 24, T = 32` -> `32.097 GB` (`S`) / `32.563 GB` (`M`) (refused on `L4`).
  - `H100` (`80.0 GB` budget): `E = 512, K = 24, T = 32` -> `56.627 GB` (`S`) / `57.093 GB` (`M`).
- **Checkpoint shard & session export arithmetic (`16.0 MiB/world` budget; `S` LIGHT budgeted `16 MiB` [`~11.5 MiB` measured], `M` LIGHT budgeted `68 MiB` [`~56 MiB` measured]; `max_periodic_full_exports = 3` + `1` stop FULL = `4` FULLs max; `64 MiB` telemetry):**
  - **8 h session (`<= 17` LIGHTs + `<= 4` FULLs):**
    - `L4` (`E = 128`, `8` shards/FULL, `12.0 GiB` cap): `8.330 GiB` (`S`) / `9.193 GiB` (`M`).
    - `A100` (`E = 256`, `16` shards/FULL, `24.0 GiB` cap): `16.332 GiB` (`S`) / `17.195 GiB` (`M`).
  - **24 h session (`<= 49` LIGHTs + `<= 4` FULLs):**
    - `L4` (`E = 128`, `8` shards/FULL, `12.0 GiB` cap): `8.830 GiB` (`S`) / `11.318 GiB` (`M`).
    - `A100` (`E = 256`, `16` shards/FULL, `24.0 GiB` cap): `16.832 GiB` (`S`) / `19.320 GiB` (`M`).

## Packet Preparation Commands (Do Not Dispatch Without Operator)

```bash
# A100 family canary packet (includes 1 default-config 192-world single-pass evaluation)
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.v3_ppo.prepare \
  --mode canary \
  --gpu A100 \
  --commit "$(git rev-parse HEAD)" \
  --packet /tmp/v3_ppo_a100_canary_pkt

# A100 ruler packet ({128, 256, 512} x {S, M} x {24, 8}, T=32, warmup=1, timed=2)
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.v3_ppo.prepare \
  --mode ruler \
  --gpu A100 \
  --commit "$(git rev-parse HEAD)" \
  --canary-receipt /path/to/a100-or-l4-canary/output/receipt.json \
  --packet /tmp/v3_ppo_a100_ruler_pkt

# A100 256-world M training packet (LIGHT every 30m, bounded FULL every 2h, single-pass eval every 60m)
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.v3_ppo.prepare \
  --mode train \
  --gpu A100 \
  --num-envs 256 \
  --net-size M \
  --hours 6.0 \
  --commit "$(git rev-parse HEAD)" \
  --canary-receipt /path/to/a100-or-l4-canary/output/receipt.json \
  --packet /tmp/v3_ppo_a100_train_pkt
```
