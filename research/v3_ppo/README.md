# Game V3.1 PPO Self-Play (`research/v3_ppo`)

From-scratch PPO self-play trainer, throughput/VRAM ruler, deterministic M-of-N evaluator, and Colab CLI packet builder for continuous-heading circular-arena Game V3.1 (`research/game_v3`, default `R = 600`, `N = 24` snakes, `10` discrete actions).

## Key Design Contracts

1. **One `build_obs` per tick (`Uint8RolloutBuffer`):**
   - Each tick writes multi-scale `uint8` ego rasters (`13x64x64 + 13x32x32 + 13x32x32 = 79,872` bytes/step), `8` `float32` scalars, and `10` `bool` action-mask entries straight into `Uint8RolloutBuffer` via prebound zero-copy slice views (`79,938` total bytes/learner-step).
   - Post-rollout `s_T` bootstrap observations are cached and copied into `t = 0` of the next rollout (`1` `build_obs` call per simulation tick in steady state; zero observation rebuilds during PPO epochs).
   - `uint8 -> float / 255.0` conversion happens inside `EgoMultiScaleNet` under `bf16` autocast.
2. **Ego Multi-Scale Actor-Critic & Host-Sync-Free Learner (`EgoMultiScaleNet`, `V3PPOTrainer`):**
   - Three scale-specific residual Conv+GroupNorm+SiLU stems (`s1` `64->4`, `s4` `32->4`, `s16` `32->4`) fused with an `8`-scalar MLP embedding, `apply_recurrent_core` seam (feed-forward by default), `10`-action masked categorical policy head, and `255`-bin symlog two-hot value head.
   - Calibrated presets in `NET_PRESETS` (`research/v3_ppo/config.py`): `S` (`1,008,657` params) and `M` (`4,927,841` params). Unknown preset names fail immediately at `prepare` / config construction time and never fall through to another size.
   - `channels_last`, `bf16` autocast, `fast_learner_ops` (float32 rounding-level equivalent to dense reference ops, `~1e-7..1e-6` weight diff over multi-update trajectories, not bit-for-bit identical), and `fused=True` `AdamW` are ON by default on CUDA (`--no-channels-last`, `--no-amp` / `--fp32`, `--no-fast-learner-ops` to disable; `--compile` opt-in across `ppo`, `ruler`, `prepare`, and `job.sh`, compiling the shared multi-scale trunk executed by `forward`, rollout `act`, and learner `evaluate_actions`).
   - `net.py` performs zero host synchronizations; `ppo_update` enforces uniform minibatch shapes (dropping partial remainders when `n_valid >= minibatch_size`, or masking padded rows when `n_valid < minibatch_size`) so `cudnn.benchmark` never re-autotunes, and synchronizes scalar diagnostics once per update.
3. **Learners, Episodes & Simultaneous Kill-Credit Rule:**
   - Default `24/24` pure self-play; supports `M`-of-`N` learners via `LearnerIndex` against `food_seeking`, `random_safe`, or `cardinal6_safe` scripted opponents.
   - Life = episode (`max_life_ticks == 0`; `sim.step(actions, truncate=...)` at rollout boundaries bootstraps `V(s_T)` while keeping large snakes alive in the world; death has zero bootstrap).
   - **Simultaneous kill + death:** `GameV3Sim` awards `kill_credit` to a killer even if the killer also dies on the same tick. By design, `compute_step_reward` pays only `-reward_death_penalty` on dying ticks (never rewarding kamikaze trades), whereas `eval.py` counts all `l_acted` `kill_credit` events (including simultaneous kill+death) so reported `kills_per_1k_agent_steps` matches the simulator's exact kill tally.
4. **Two-Tier Atomic Checkpointing, `--full-exports` Policy, Immutable Exports & `work/`-Only Retention:**
   - **`--full-exports {none,final,periodic}` (default `none`):** controls whether whole-world `uint8`-zlib simulator `FULL` shards (`full_u*.sim_shard_*_of_*.pt` + `full_u*.index.json`) are exported to `output/checkpoints/`:
     - `none` *(default)*: **no** periodic `FULL` and **no** final `FULL` are built for export or published to `output/`; only unthrottled periodic `LIGHT` checkpoints (`light_u*.pt` every `light_checkpoint_interval_sec = 1800`s = `30` min) and the final boundary `LIGHT` are exported. `FULL` checkpoints are still saved inside the VM's `work/checkpoints/` at `full_checkpoint_interval_sec = 7200`s (`2` h) so in-session restarts keep mature worlds.
     - `final`: periodic `FULL` exports off (equivalent to `--max-periodic-full-exports 0`), exporting `1` final `FULL` set after the final `LIGHT` is `.complete` in `output/checkpoints/`.
     - `periodic`: periodic `FULL` exports (up to `max_periodic_full_exports = 3`) plus `1` final `FULL` export; `prepare` emits a loud `WARNING [V3_PPO_PERIODIC_FULL_EXPORTS]` because `2`-hourly `FULL` shard downloads have exhausted the Mac supervisor transfer-time cap (`UPLOAD_CAP`) and killed `8` h `L4`/`A100` runs mid-training.
     - Legacy `--max-periodic-full-exports` remains supported (`0 -> final`, `k >= 1 -> periodic`); conflicting `--full-exports` and `--max-periodic-full-exports` values are refused.
   - **`LIGHT` checkpoints** (`light_u*.pt`: network, optimizer, scaler, config, counters, episode accumulators, timestamp, restart generation, and all RNG streams; no simulator state) are exported at every real `light_checkpoint_interval_sec = 1800`s (`30` min) with **no per-session count cap and no interval stretching**, plus at cooperative stop / wall guard.
   - **Resume choice:** prefers newest verified `FULL` set when present (bit-identical on CPU including `clean_step_streak`), switching to a newer verified `LIGHT` checkpoint if newer by `> light_resume_margin_minutes = 45.0` min, and treats `LIGHT`-only prior outputs as normal (`reason=only_light_available`, logging informational `V3_PPO_LIGHT_RESUME` stating `"worlds reinitialised"` without error-like warning banners).
   - **Safety & `work/`-only retention:** completed exports in `output/` are immutable and never rewritten or deleted; `"keep newest N"` (`keep_full_checkpoints = 2`, `keep_light_checkpoints = 4`) prunes **only `work/checkpoints/`**. Cooperative stops mid-rollout or mid-PPO-epoch discard half-stepped weights and publish only clean update-boundary checkpoints; in-session restarts (`max_in_session_restarts = 2`) tag post-restart checkpoints (`_r01`, `_r02`) and never bind a `FULL` index to a stale pre-restart `LIGHT`.
   - **Per-run recipe overrides (`--recipe-json`) & recipe-changing continuations (`--allow-recipe-change`):**
     - `prepare --mode train --recipe-json <PATH_OR_JSON>` overrides any subset of the `13`-field allow-list (`RECIPE_OVERRIDE_KEYS`: `reward_alive_length_coef`, `reward_alive_const`, `reward_length_delta_coef`, `reward_kill_coef`, `reward_death_penalty`, `gamma`, `gae_lambda`, `entropy_coef`, `lr`, `ppo_epochs`, `minibatch_size`, `rollout_len`, `clip_ratio`), prints `recipe overrides: ...` (or `none`) in the `prepare` summary, logs `V3_PPO_RECIPE` once at job start, and sizes VRAM/horizon checks with the overridden `rollout_len` and `minibatch_size`.
     - By default, `prepare` and `ppo` refuse any continuation whose recipe fields differ from the prior checkpoint's config. Passing `--allow-recipe-change` permits changing **only** the `13` allow-listed recipe fields above (architecture, world count, learner seats, `game_config`, and non-allow-listed optimizer/loss settings remain immutable), logs `V3_PPO_RECIPE_CHANGE old=... new=... update=...` at resume, and appends `{update, changed_fields}` to `recipe_change_history` in `config.json`, saved checkpoints, and `train_result.json`.
     - **What is kept (not reset) on `--allow-recipe-change`:** policy and symlog value-head network weights (`model_state_dict`), `AdamW` optimizer state (`exp_avg`, `exp_avg_sq`, and per-parameter `step` counts), `GradScaler` state, global update/step/episode counters, RNG streams, and—when resuming from a `FULL` checkpoint—simulator world state and in-flight `episode_return_acc` (when resuming from a `LIGHT` checkpoint, worlds are freshly seeded and `episode_return_acc` is zeroed as usual).
     - **What is updated / reset on `--allow-recipe-change`:** the allow-listed recipe fields in `V3PPOConfig`, `optimizer.param_groups` `lr` (immediately updated to the new `cfg.lr`), the `Uint8RolloutBuffer` allocation (sized to the new `rollout_len`), and the bootstrap observation cache (`_bootstrap_obs_valid = False`, rebuilt on the first post-resume tick). Value-head weights are intentionally not zeroed and adapt to any new return scale over subsequent updates.
5. **Single-Pass Dual-Matchup Evaluation, Stop Polling & Unsynced Headline Ruler:**
   - Fixed-seed evaluation (`eval_every_minutes = 60.0`, `eval_envs = 192` -> `96` worlds vs `food_seeking` + `96` worlds vs `random_safe` in **one** `GameV3Sim` pass, `eval_learner_slots = 8`, `eval_burnin_ticks = 250`, `eval_ticks = 2000`) evaluates both matchups simultaneously without perturbing training worlds or RNG.
   - Polls the stop flag and wall guard every `16`-tick chunk (`aborted: True` on stop, never logged as a data point), skips starting an evaluation when remaining wall `< 2x` last measured eval wall (or `2x eval_estimated_seconds` before the first), logs per-eval and cumulative eval wall share, and excludes eval wall from training throughput (`sps_scope=train_excl_eval`). Family canary runs `1` default-config evaluation concurrently with a resident GPU-default training simulator and rollout buffer (`128` L4 / `256` A100 / `512` H100 worlds), recording measured wall seconds and combined `peak_memory_bytes` in `canary_checks.json`; `prepare --mode train` propagates `default_eval_wall_seconds` from the bound canary receipt into `eval_estimated_seconds`.
   - Completed lives and window-censored survivors are never mixed: reports per-completed-life `mean/median/p95_death_length`, `mean/median/p95_lifetime_ticks`, separate `censored_lives` / `mean_censored_length`, survivorship-free rates (`time_averaged_length`, `mean_alive_length`, `mass_gained_per_1k_agent_steps`, `net_length_delta_per_1k_agent_steps`, `deaths_per_1k_agent_steps` with raw `total_deaths`, and `kills_per_1k_agent_steps`), and `mean_completed_episode_return` (**recipe-dependent** — weighted by `cfg`'s `reward_*` coefficients, whereas all length, lifetime, censored-life, death, kill, and growth metrics are **recipe-invariant** and directly comparable across reward/PPO recipes).
   - Ruler cells default to `rollout_len = 32`, `warmup_updates = 1`, `timed_updates = 2`, measure headline collection and integrated throughput with per-stage timing syncs **OFF** (`stage_timing=False`), run `1` separate synced pass for `time_split_pct`, precheck cell VRAM before allocation, and hold at most `1` compressed mature world-count state in host RAM at a time.

## Memory & Checkpoint Export Arithmetic (`R = 600, N = 24, T = 32`)

- **Per-rollout learner storage (`E = 128, K = 24, T = 32`):** `98,304 * 79,938 = 7.858 GB` + `0.246 GB` bootstrap cache = `8.104 GB`.
- **Simulator (`GameV3Sim`):** `3.684 GB` persistent (`27.45 MiB/world`), `4.161 GB` peak (`31.00 MiB/world`).
- **Two-phase peak VRAM (max across rollout/learner phase and `192`-world single-pass eval phase):**
  - `L4` (`22.0 GB` budget): `E = 128, K = 24, T = 32` -> `19.832 GB` (`S`) / `20.298 GB` (`M`).
  - `A100` (`40.0 GB` budget): `E = 256, K = 24, T = 32` -> `32.097 GB` (`S`) / `32.563 GB` (`M`) (refused on `L4`).
  - `H100` (`80.0 GB` budget): `E = 512, K = 24, T = 32` -> `56.627 GB` (`S`) / `57.093 GB` (`M`).
- **Checkpoint shard & session export arithmetic (`S` LIGHT budgeted `16 MiB` [`~11.5 MiB` measured], `M` LIGHT budgeted `68 MiB` [`~56 MiB` measured]; `64 MiB` telemetry):**
  - **Default `--full-exports none` (`0` FULL exports; per-file cap `80 MiB` `S` / `85 MiB` `M`; whole-session cap `4.0 GiB`; FULL shard/size refusals do not fire):**
    - `8 h` (`<= 17` LIGHTs): `0.328 GiB` (`S`) / `1.191 GiB` (`M`) on all GPUs (`L4`, `A100`, `H100`).
    - `24 h` (`<= 49` LIGHTs): `0.828 GiB` (`S`) / `3.316 GiB` (`M`) on all GPUs (`L4`, `A100`, `H100`).
  - **`--full-exports final` (`1` FULL) & `--full-exports periodic` (`<= 4` FULLs at `16.0 MiB/world` budget; per-file cap `320 MiB`):**
    - `8 h` (`periodic`, `<= 17` LIGHTs + `<= 4` FULLs): `L4` (`E = 128`, `12.0 GiB` cap) `8.330 GiB` (`S`) / `9.193 GiB` (`M`); `A100` (`E = 256`, `24.0 GiB` cap) `16.332 GiB` (`S`) / `17.195 GiB` (`M`).
    - `24 h` (`periodic`, `<= 49` LIGHTs + `<= 4` FULLs): `L4` (`E = 128`, `12.0 GiB` cap) `8.830 GiB` (`S`) / `11.318 GiB` (`M`); `A100` (`E = 256`, `24.0 GiB` cap) `16.832 GiB` (`S`) / `19.320 GiB` (`M`).
- **Train-mode wall-guard headroom (`prepare --mode train` only; `timeout_minutes` unchanged):**
  - Sizes `headroom = setup_allowance (330 s, calibrated to measured 240–325 s session-start-to-wall_t0) + staged_input_bytes / ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS (1.0 MB/s = 1,000,000 B/s; 0 s on fresh runs, prior LIGHT/FULL bytes on continuations) + final_eval_and_save_allowance (300 s default, or canary_eval_wall_seconds + 150 s when bound receipt provides it) + final_export_bytes / ASSUMED_FINAL_EXPORT_DOWNLOAD_RATE_MBPS (1.0 MB/s = 1,000,000 B/s)`, where `final_export_bytes = final_light_bytes + 4 MiB` (`TRAIN_FINAL_TELEMETRY_EXPORT_BYTES`) in `none` mode and `final_full_bytes + final_light_bytes` (`16.0 MiB/world` budget) in `final`/`periodic` modes. Sets `V3_PPO_MAX_WALL_SECONDS = floor(timeout_seconds - headroom)` and refuses any `train` packet with `headroom > 40%` of the session.
  - Inside `V3_PPO_MAX_WALL_SECONDS`, `run_training_job` additionally holds `checkpoint_reserve_sec = min(TRAINER_CHECKPOINT_RESERVE_SECONDS, max(5 s, 0.08 * max_wall_seconds))` (`TRAINER_CHECKPOINT_RESERVE_SECONDS = 90.0 s` on sessions `>= 18.75 min`) so the active rollout/PPO update completes before final evaluation and checkpoint export begin.
  - On clean boundary stop (`completed`, `wall_clock_guard`, or boundary signal), the stop decision is evaluated once per update and reused for loop exit, and final `LIGHT` (`light_<tag>.pt`) is always exported and `.complete` in `output/checkpoints/` before any final `FULL` shard compression/publication starts in `final` or `periodic` modes.
  - Side-by-side non-continuation headroom across `--full-exports {none,final,periodic}` (default `300 s` finalize allowance, `1.0 MB/s` assumed transfer rate, `16.0 MiB/world` `FULL` budget; continuations add `staged_input_bytes / 1.0 MB/s`):

| GPU & Worlds | Net | `none` Headroom / `4 h` (`14,400 s`) / `8 h` (`28,800 s`) *(default)* | `final` Headroom / `4 h` (`14,400 s`) / `8 h` (`28,800 s`) | `periodic` Headroom / `4 h` (`14,400 s`) / `8 h` (`28,800 s`) *(warns)* |
| --- | --- | --- | --- | --- |
| `L4`, `128` worlds | `S` | `650.97 s` (`10.85 min`) · `4.52%` / `13,749 s` · `2.26%` / `28,149 s` | `2,794.79 s` (`46.58 min`) · `19.41%` / `11,605 s` · `9.70%` / `26,005 s` | `2,794.79 s` (`46.58 min`) · `19.41%` / `11,605 s` · `9.70%` / `26,005 s` |
| `L4`, `128` worlds | `M` | `705.50 s` (`11.76 min`) · `4.90%` / `13,694 s` · `2.45%` / `28,094 s` | `2,849.31 s` (`47.49 min`) · `19.79%` / `11,550 s` · `9.89%` / `25,950 s` | `2,849.31 s` (`47.49 min`) · `19.79%` / `11,550 s` · `9.89%` / `25,950 s` |
| `A100`, `256` worlds | `S` | `650.97 s` (`10.85 min`) · `4.52%` / `13,749 s` · `2.26%` / `28,149 s` | `4,942.79 s` (`82.38 min`) · `34.32%` / `9,457 s` · `17.16%` / `23,857 s` | `4,942.79 s` (`82.38 min`) · `34.32%` / `9,457 s` · `17.16%` / `23,857 s` |
| `A100`, `256` worlds | `M` | `705.50 s` (`11.76 min`) · `4.90%` / `13,694 s` · `2.45%` / `28,094 s` | `4,997.32 s` (`83.29 min`) · `34.70%` / `9,402 s` · `17.35%` / `23,802 s` | `4,997.32 s` (`83.29 min`) · `34.70%` / `9,402 s` · `17.35%` / `23,802 s` |
| `H100`, `512` worlds | `S` | `650.97 s` (`10.85 min`) · `4.52%` / `13,749 s` · `2.26%` / `28,149 s` | `9,238.81 s` (`153.98 min`) · `64.16%` **REFUSED** · `32.08%` / `19,561 s` | `9,238.81 s` (`153.98 min`) · `64.16%` **REFUSED** · `32.08%` / `19,561 s` |
| `H100`, `512` worlds | `M` | `705.50 s` (`11.76 min`) · `4.90%` / `13,694 s` · `2.45%` / `28,094 s` | `9,293.33 s` (`154.89 min`) · `64.54%` **REFUSED** · `32.27%` / `19,506 s` | `9,293.33 s` (`154.89 min`) · `64.54%` **REFUSED** · `32.27%` / `19,506 s` |

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

# A100 256-world M training packet (default --full-exports none: LIGHT every 30m + final LIGHT, eval every 60m)
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

## Configuration Precedence (`ppo` & `ruler`)

`resolve_effective_train_config` (`python -m research.v3_ppo.ppo`) and `resolve_effective_ruler_config` (`python -m research.v3_ppo.ruler`) resolve configuration in strict highest-to-lowest order:

1. **Explicit individual CLI flags** (`--num-envs`, `--lr`, `--amp`/`--no-amp`, `--compile`/`--no-compile`, `--channels-last`/`--no-channels-last`, `--fast-learner-ops`/`--no-fast-learner-ops`, etc.).
2. **Explicit `--config-json` CLI argument** (keys present in the JSON object; `ppo` only).
3. **Environment variables** (`V3_PPO_CONFIG_JSON`, overlaid by scalar `V3_PPO_*` and `SNAKE_DQN_DEVICE`).
4. **Dataclass defaults** (`V3PPOConfig()` / `GameV3Config()`).

## Adding a Network Size Preset (e.g., `L`)

All network size parameters live in the single `NET_PRESETS` registry (`NetPreset` in `research/v3_ppo/config.py`), which drives `EgoMultiScaleNetConfig.from_size`, CLI choices, memory estimation, checkpoint export caps, and ruler ordering. Adding an `"L"` preset is a single-entry addition in `NET_PRESETS` (plus widening the `NetSize = Literal["S", "M", "L"]` type alias) with:

1. **Architecture dimensions & sweep order:** `s1_c1`, `s1_c2`, `s32_c1`, `stage8_ch`, `stem_out_ch`, `scalar_embed_dim`, `trunk_dim`, `trunk_blocks`, `head_dim`, and `ruler_order` (`2`, so cheaper presets run first in `run_ruler`). Note `GroupNorm` in stems uses up to `8` groups, so conv channel widths should be multiples of `8`.
2. **Parameter counts:** `exact_param_count` (exact `EgoMultiScaleNet.parameter_count()` on default `13`-channel / `8`-scalar / `10`-action inputs) and `memory_model_param_count` (rounded parameter figure within `±10%` of `exact_param_count`, enforced by unit test; multiplied by `16` bytes/param for `fp32` weights + grads + `AdamW` moments).
3. **Measured CUDA memory figures (`bf16` + `channels_last`):**
   - `learner_act_bytes_per_row_bf16`: peak backward-saved activation + raster float-copy bytes per minibatch row during `ppo_update`, measured on CUDA via `torch.cuda.reset_peak_memory_stats()` / `torch.cuda.max_memory_allocated()` across two minibatch sizes (`(peak_B2 - peak_B1) / (B2 - B1)`, with a ~10–15% safety margin; the estimator scales this by `1.8x` when `amp=False`).
   - `actor_fwd_act_bytes_per_row_bf16`: transient `torch.no_grad()` stem/trunk activation bytes per learner row during `act` beyond the `OBS_RASTER_BYTES_PER_STEP * 2` `bf16` raster copy, measured on CUDA across two `actor_chunk_learners` sizes (scaled by `2x` when `amp=False`).
4. **Default batching:** `default_minibatch_size` (dividing typical `rollout_len * num_envs * learner_slots` without blowing learner VRAM) and `default_actor_chunk_learners`.
5. **Measured `LIGHT` checkpoint export budget:** `est_light_checkpoint_bytes` (`>=` measured `save_light_checkpoint` file size after one `AdamW` step so optimizer moments are populated, plus ~25% headroom for `estimate_checkpoint_export_caps` and session stop-reserve sizing).
