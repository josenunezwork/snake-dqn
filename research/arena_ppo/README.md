# Arena PPO (Stage 1 — From-Scratch GPU-Resident Full-Arena PPO)

Stage 1 implementation of GPU-resident PPO self-play on the exact mechanics-v2
`DeviceSim` (`83 x 145` rectangular arena by default, configurable `(grid_h, grid_w)`
and `num_snakes`), with zero `vector61` / `v8` / BFS flood features in the training
tick.

## Modules

- `runtime.py`: Canonical device normalization, pre-allocation CUDA lazy-init,
  single-SM native extension preloading (`move_cuda._extension()` +
  `event_cuda` `snake_dqn_a0_event_v1`), `SignalStopFlag` (`SIGUSR1` + `SIGTERM`),
  deterministic backend toggles, UTC `ARENA_PPO_PHASE` stamps, and create-only
  `.sha256` + `.complete` artifact writers.
- `sim.py`: `TrainSim` wrapping exact mechanics-v2 `DeviceSim` (`train_mode=False`,
  `allow_respawn=True`, `hero_respawn=True`, `exact_mechanics=True`), parameterized
  `(grid_h, grid_w, num_snakes)`, live-window `int16` `state_dict` serialization,
  `reset_worlds(hit_mask)`, `assert_no_capacity_overflow()`, 4-world x 100-tick
  `verify_indexed_vs_reference_parity()`, and per-world opponent routing (`self`,
  `snapshot`, `scripted` `greedy_food`). Note on host syncs: `obs.py`, `net.py`, and
  `symlog_two_hot_loss` perform zero host syncs; the pinned `DeviceSim.step` itself
  performs scalar event syncs (`bool(...any())` / `_append_many`).
- `obs.py`: Zero-sync GPU/MPS observation builder (`13 x grid_h x grid_w` allocentric
  arena with 1-cell lethal perimeter border on `CH_WALL`, `13 x ego_crop x ego_crop`
  heading-aligned ego crop with out-of-arena `CH_WALL`, `6` scalars, in-place
  dead-seat masking) and pure-NumPy CPU reference builder proven bit-identical across
  `>= 64` worlds x `200` ticks.
- `net.py`: Dual-stream `ArenaNet` presets `S` (`1,027,080` params) and `M`
  (`5,017,480` params) with adaptive `token_grid_h x token_grid_w` (`11 x 19 = 209`
  default) arena self-attention tokens, ego ResNet, masked 6-action policy head,
  zero-sync symlog two-hot value head (`255` bins), and death-within-`{16, 64, 256}`
  auxiliary heads (`channels_last`, CUDA `bf16` autocast with `fp16` fallback if
  `bf16` unsupported, optional `torch.compile`).

| Preset | Params | Arena channels (`stem / s1 / s2`, blocks/stage) | Token self-attn (`layers, heads, MLP`) | Ego channels (`stem / s1 / s2`, blocks) | Fuse dim | Default `actor_chunk_envs` / `minibatch_size` |
|---|---:|---|---|---|---|---|
| `S` | `1,027,080` | `32 / 64 / 96`, `1` block | `2` layers, `4` heads, `4x` MLP | `32 / 64 / 96`, `(1, 2)` blocks | `192` | `1024` / `1536` (`3072` larger ruler probe) |
| `M` | `5,017,480` | `64 / 128 / 192`, `2` blocks | `2` layers, `6` heads, `4x` MLP | `64 / 128 / 192`, `(2, 2)` blocks | `320` | `512` / `768` (`1536` larger ruler probe) |

- `ppo.py`: `ArenaPPOConfig` (JSON-serializable), `CompactRolloutBuffer` (lossless
  append-only `int16` body tape), clipped PPO learner with GAE (`gamma=0.999`,
  `gae_lambda=0.97`, `r_t = L_t / 1000` per alive tick), per-tick/per-minibatch
  cooperative signal stop, projected wall-budget + `max_update_seconds` (`<= 20` min)
  guards, newest-`3` periodic checkpoint retention, and config-checked resume.
  Determinism: bit-identical `resumed == uninterrupted` holds on CPU with
  `deterministic=True`; on CUDA with default `deterministic=False` (`cudnn.benchmark`,
  `bf16`, atomic `scatter_reduce_`), resume restores exact model/optimizer/sim/RNG
  state for training continuity with expected minor float reduction variance.
- `ruler.py`: Signal-safe throughput & VRAM ruler with `>= 200`-tick untimed
  `greedy_food` burn-in, world maturity telemetry (`mean_length`, `p95_length`,
  `deaths_count`, `corpse_count`), 6-stage split (`sim_step`, `rollout_record`,
  `obs_build`, `policy_forward`, `gae`, `learner`) + `untracked_seconds`, per-stage
  peak VRAM, OOM halving retries, dual minibatch probes, `10` warmup / `40` bench
  single-world forward iterations, cheapest-first ordering across
  `E in {1024, 2048, 4096, 8192} x {S, M}` plus `--compile` on the best `E` of each
  net, incremental `cells/<cell_id>.json` exports, and `run_canary_suite` (zero fake
  or stub exports anywhere).
- `prepare.py` / `job.sh` / `requirements.txt`: Three-mode Colab CLI packet builder
  supporting `L4`, `A100`, and `H100` (`T4` excluded):
  1. `canary` (`short-canary`, `<= 12` min, instrumented): runs real 4-world x
     100-tick `indexed` vs `reference` parity, tiny real `run_ruler`, and tiny real
     2-update `run_training_job` + checkpoint reload.
  2. `ruler` (`acceptance` resume exemption, `60` min runaway cap, `52` min Python
     wall guard, uninstrumented science run bound to `--canary-receipt`).
  3. `train` (`resume` block with verified checkpoint ledger, sized
     `max_export_bytes`, bound to `--canary-receipt`).
