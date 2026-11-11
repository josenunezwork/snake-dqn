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
   - `channels_last`, `bf16` autocast, and `fused=True` `AdamW` are ON by default on CUDA (`--no-channels-last`, `--no-amp` / `--fp32` to disable).
   - `net.py` performs zero host synchronizations; `ppo_update` enforces uniform minibatch shapes (dropping partial remainders when `n_valid >= minibatch_size`, or masking padded rows when `n_valid < minibatch_size`) so `cudnn.benchmark` never re-autotunes, and synchronizes scalar diagnostics once per update.
3. **Learners, Episodes & Simultaneous Kill-Credit Rule:**
   - Default `24/24` pure self-play; supports `M`-of-`N` learners via `LearnerIndex` against `food_seeking`, `random_safe`, or `cardinal6_safe` scripted opponents.
   - Life = episode (`max_life_ticks == 0`; `sim.step(actions, truncate=...)` at rollout boundaries bootstraps `V(s_T)` while keeping large snakes alive in the world; death has zero bootstrap).
   - **Simultaneous kill + death:** `GameV3Sim` awards `kill_credit` to a killer even if the killer also dies on the same tick. By design, `compute_step_reward` pays only `-reward_death_penalty` on dying ticks (never rewarding kamikaze trades), whereas `eval.py` counts all `l_acted` `kill_credit` events (including simultaneous kill+death) so reported `kills_per_1k_agent_steps` matches the simulator's exact kill tally.
4. **Two-Tier Atomic Checkpointing, Mid-Update Rollback & In-Session Restart:**
   - **`LIGHT` checkpoints** (`light_u*.pt`: network, optimizer, scaler, config, counters, episode accumulators, and all RNG streams; no simulator state) are published every `light_checkpoint_interval_sec = 1800`s (`30` min) and on stop.
   - **`FULL` checkpoints** (`light_u*.pt` + whole-world `uint8`-zlib simulator shards `full_u*.sim_shard_*_of_*.pt` each `<= checkpoint_shard_bytes = 256 MiB` with `.sha256`/`.complete` sidecars + `full_u*.index.json` written last) are published every `full_checkpoint_interval_sec = 7200`s (`2` h) and on a clean stop.
   - **Resume order:** newest complete `FULL` set (verified index + all shards + light file; bit-identical on CPU including `_clean_step_streak`); else newest verified `LIGHT` checkpoint with freshly initialised worlds (logging `V3_PPO_LIGHT_RESUME_WARNING` that episodic statistics need warm-up).
   - **Safety & retention:** cooperative stops mid-rollout or mid-PPO-epoch discard half-stepped weights and publish only clean update-boundary checkpoints; recoverable runtime errors restart inside the session from the newest verified checkpoint (`max_in_session_restarts = 2`); retention prunes to `keep_full_checkpoints = 2` and `keep_light_checkpoints = 4`.
5. **Burn-In + Survivorship-Free Deterministic Evaluation & Warmed Ruler:**
   - Fixed-seed evaluation (`eval_every_minutes = 30.0`, `eval_envs = 16`, `eval_learner_slots = 8`, `eval_burnin_ticks = 500`, `eval_ticks = 5000`) matures fresh worlds for `500` ticks before measuring `5,000` ticks vs `food_seeking` and `random_safe` without perturbing training worlds or RNG.
   - Completed lives and window-censored survivors are never mixed: reports per-completed-life `mean/median/p95_death_length`, `mean/median/p95_lifetime_ticks`, `mean_completed_episode_return`, separate `censored_lives` / `mean_censored_length`, and survivorship-free rates (`time_averaged_length`, `mass_gained_per_1k_agent_steps`, `deaths_per_1k_agent_steps` with raw `total_deaths`, and `kills_per_1k_agent_steps`).
   - Ruler cells default to `rollout_len = 32`, `warmup_updates = 1`, `timed_updates = 2`, report `collect_seconds` and `learn_seconds` separately, precheck cell VRAM before allocation, and hold at most `1` compressed mature world-count state in host RAM at a time.

## Memory & Checkpoint Export Arithmetic (`R = 600, N = 24, T = 32`)

- **Per-rollout learner storage (`E = 128, K = 24, T = 32`):** `98,304 * 79,938 = 7.858 GB` + `0.246 GB` bootstrap cache = `8.104 GB`.
- **Simulator (`GameV3Sim`):** `3.684 GB` persistent (`27.45 MiB/world`), `4.161 GB` peak (`31.00 MiB/world`).
- **Actor chunk + eval worlds + learner/CUDA reserve:** accounts for transient actor forward float copy (`actor_chunk_learners`) and transient evaluation `GameV3Sim` (`eval_envs = 16`, `~0.547 GB`).
- **Estimated peak VRAM:**
  - `L4` (`22.0 GB` budget): `E = 128, K = 24, T = 32` -> `15.030 GB` (`S`) / `15.298 GB` (`M`).
  - `A100` (`40.0 GB` budget): `E = 256, K = 24, T = 32` -> `27.295 GB` (`S`) / `27.563 GB` (`M`) (refused on `L4`).
  - `H100` (`80.0 GB` budget): `E = 512, K = 24, T = 32` -> `51.825 GB` (`S`) / `52.092 GB` (`M`).
- **Checkpoint shard & session export caps (`16.0 MiB/world` budget = `4.8x` tick-15k `3.35 MiB`, `11.2x` tick-3k `1.424 MiB`; `256 MiB` shards -> `320 MiB` per-file cap `<= 2.0 GiB` supervisor limit):**
  - `L4` (`E = 128`): `8` shards/FULL, worst-case session `4.157 GiB` (`S`) / `4.462 GiB` (`M`) `<= 6.0 GiB` cap.
  - `A100` (`E = 256`): `16` shards/FULL, worst-case session `8.158 GiB` (`S`) / `8.463 GiB` (`M`) `<= 12.0 GiB` cap.
  - `H100` (`E = 512`): `32` shards/FULL, worst-case session `16.160 GiB` (`S`) / `16.465 GiB` (`M`) `<= 20.0 GiB` cap.

## Packet Preparation Commands (Do Not Dispatch Without Operator)

```bash
# A100 family canary packet
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

# A100 256-world M training packet (sharded FULL every 2h, LIGHT every 30m, eval every 30m)
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
