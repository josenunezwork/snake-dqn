# Game V3.1 PPO Self-Play (`research/v3_ppo`)

From-scratch PPO self-play trainer, throughput/VRAM ruler, deterministic M-of-N evaluator, and Colab CLI packet builder for continuous-heading circular-arena Game V3.1 (`research/game_v3`, default `R = 600`, `N = 24` snakes, `10` discrete actions).

## Key Design Contracts

1. **One `build_obs` per tick (`Uint8RolloutBuffer`):**
   - Each tick writes multi-scale `uint8` ego rasters (`13x64x64 + 13x32x32 + 13x32x32 = 79,872` bytes/step), `8` `float32` scalars, and `10` `bool` action-mask entries straight into `Uint8RolloutBuffer` via prebound zero-copy slice views (`79,938` total bytes/learner-step).
   - Post-rollout `s_T` bootstrap observations are cached and copied into `t = 0` of the next rollout (`1` `build_obs` call per simulation tick in steady state; zero observation rebuilds during PPO epochs).
   - `uint8 -> float / 255.0` conversion happens inside `EgoMultiScaleNet` under `bf16` autocast.
2. **Ego Multi-Scale Actor-Critic (`EgoMultiScaleNet`):**
   - Three scale-specific residual Conv+GroupNorm+SiLU stems (`s1` `64->4`, `s4` `32->4`, `s16` `32->4`) fused with an `8`-scalar MLP embedding, `apply_recurrent_core` seam (feed-forward in round 1), `10`-action masked categorical policy head, and `255`-bin symlog two-hot value head.
   - Calibrated presets: `S` (`1,008,657` params) and `M` (`4,927,841` params).
   - `channels_last` and `bf16` autocast ON by default on CUDA (`--no-channels-last`, `--no-amp` / `--fp32` to disable).
3. **Learners, Episodes & Rewards:**
   - Default `24/24` pure self-play; supports `M`-of-`N` learners via `LearnerIndex` against `food_seeking`, `random_safe`, or `cardinal6_safe` scripted opponents.
   - Life = episode (`max_life_ticks == 0`; `sim.step(actions, truncate=...)` at rollout boundaries bootstraps `V(s_T)` while keeping large snakes alive in the world; death has zero bootstrap and pays `-reward_death_penalty` on the dying snake only).
4. **Atomic Checkpointing & Deterministic Evaluation:**
   - Periodic (`checkpoint_interval_sec = 1800`s default) and cooperative-stop checkpoints save model, optimizer, config, counters, RNG states, and losslessly packed `GameV3Sim.state_dict()` with `.sha256` and `.complete` sidecars; CPU resume is bit-identical.
   - Built-in fixed-seed evaluation runs M-of-N learners vs `food_seeking` and `random_safe`, logging `mean_final_length`, `p95_final_length`, `deaths_per_1k_agent_steps`, and `kills_per_1k_agent_steps`.

## Default Memory Arithmetic (`E = 128, K = 24, T = 32`, `R = 600`)

- **Learner-steps per rollout:** `128 * 24 * 32 = 98,304`
- **Rollout `uint8` rasters:** `98,304 * 79,872 = 7,851,737,088` B (`7.852 GB` / `7.312 GiB`)
- **Rollout buffer total:** `98,304 * 79,938 = 7,858,225,152` B (`7.858 GB` / `7.319 GiB`) + `0.245 GB` bootstrap cache = `8.104 GB`
- **Simulator (`GameV3Sim`):** `3.684 GB` persistent (`27.45 MiB/world`), `4.161 GB` peak (`31.00 MiB/world`)
- **Estimated peak VRAM:** `14.395 GB` (`S`) / `14.462 GB` (`M`) — fits **22 GB L4** (`~7.6 GB` headroom) and **40 GB A100** (`~25.6 GB` headroom; `E = 256, K = 24, T = 32` is `26.660 GB`, fitting 40 GB A100 and refused by `prepare` on 22 GB L4).

## Packet Preparation Commands (Do Not Dispatch Without Operator)

```bash
# L4 family canary packet
python -m research.v3_ppo.prepare \
  --mode canary \
  --gpu L4 \
  --packet /tmp/v3_ppo_l4_canary_pkt

# L4 ruler packet ({128, 256, 512} x {S, M} x {24, 8}, bound to L4 canary receipt)
python -m research.v3_ppo.prepare \
  --mode ruler \
  --gpu L4 \
  --canary-receipt /path/to/l4-canary/output/receipt.json \
  --packet /tmp/v3_ppo_l4_ruler_pkt
```
