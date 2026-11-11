# Arena PPO (Stage 1 — From-Scratch GPU-Resident Full-Arena PPO)

Stage 1 implementation of GPU-resident PPO self-play on the exact mechanics-v2
`DeviceSim` (`83 x 145` rectangular arena by default, configurable `(grid_h, grid_w)`
and `num_snakes`), with zero `vector61` / `v8` / BFS flood features in the training
tick.

## Modules

- `runtime.py`: Canonical device normalization, pre-allocation CUDA lazy-init,
  single-SM native extension preloading (`move_cuda._extension()` +
  `event_cuda` `snake_dqn_a0_event_v1`), `SignalStopFlag` (`SIGUSR1` + `SIGTERM`),
  deterministic backend toggles, UTC `ARENA_PPO_PHASE` stamps, create-only
  `.sha256` + `.complete` artifact writers, `publish_hardlinked_artifacts` (preserving
  inode `st_ctime_ns` on already-completed exports), and `RotatingCreateOnlyJSONLWriter`.
- `sim.py`: `TrainSim` wrapping exact mechanics-v2 `DeviceSim` (`train_mode=False`,
  `allow_respawn=True`, `hero_respawn=True`, `exact_mechanics=True`), parameterized
  `(grid_h, grid_w, num_snakes)`, device-side `int16` live-window `state_dict`
  serialization, exact `reset_worlds(hit_mask)` (including frame-1 food maintenance),
  GPU-resident `LateStateReservoir` (`late_reset_min_length=600`, `late_reset_prob=0.5`,
  default OFF), `assert_no_capacity_overflow()`, 4-world x 100-tick
  `verify_indexed_vs_reference_parity()`, and per-world opponent routing (`self`,
  `snapshot`, `scripted` `greedy_food`). Note on host syncs: `obs.py`, `net.py`, and
  `symlog_two_hot_loss` perform zero host syncs; the pinned `DeviceSim.step` itself
  performs scalar event syncs (`bool(...any())` / `_append_many`).
- `obs.py`: Zero-sync GPU/MPS observation builders and pure-NumPy CPU reference builders
  proven bit-identical across `>= 64` worlds x `200` ticks:
  - `build_shared_world_obs_gpu` / `build_shared_world_obs_cpu` (`shared_trunk` &
    `ego_only`): world-once `9 x grid_h x grid_w` allocentric planes (`body_ttl`,
    `head_curr`, `head_dx`, `head_dy`, `head_length`, `boost_flags`, `food_ambient`,
    `food_corpse`, `wall`) + direct zero-sync per-seat `13 x ego_crop x ego_crop`
    heading-aligned ego crop + `6` scalars (no `(E*S, 13, H, W)` intermediate).
  - `build_arena_ego_obs_gpu` / `build_arena_ego_obs_cpu` (`per_seat` reference):
    per-seat `13 x grid_h x grid_w` allocentric arena + `13 x ego_crop x ego_crop` ego
    crop + `6` scalars, with `prev_alive`-gated boost flags and dead-seat masking.
- `net.py`: Three selectable architectures (`--net-arch {shared_trunk,ego_only,per_seat}`),
  each with `S` and `M` presets, adaptive `ceil(grid_h / 8) x ceil(grid_w / 8)` (`11 x 19`
  default) token grids, masked 6-action policy head, zero-sync symlog two-hot value head
  (`255` bins), death-within-`{16, 64, 256}` auxiliary heads, cached CUDA `bf16` autocast
  (`fp16` fallback), optional `torch.compile`, and `measure_model_flops_per_row`
  (`torch.utils.flop_counter.FlopCounterMode`):
  - `shared_trunk` (default): ONE allocentric pass per world per tick (`4x4 stride-4`
    patchify stem -> `21 x 37` ResBlocks -> `stride-2` `11 x 19` + `2` self-attention
    layers) fused per seat with early-stride-2 `47 x 47` ego crop, `5 x 5`
    heading-rotated ROI gather from the `21 x 37` shared map, `1` cross-attention read
    from the seat query into the shared `11 x 19` tokens, and `6` scalars.
  - `ego_only`: cheapest reference — early-stride-2 `47 x 47` ego crop + `6` scalars +
    world-once `11 x 19` average-pooled `2`-conv coarse world summary.
  - `per_seat`: original per-seat dual-stream `ArenaNet`, unchanged bit-for-bit.

| Architecture | Size | Params | Fwd FLOPs/row | Learner FLOPs/row | Default `actor_chunk_envs` / `minibatch_size` |
|---|---|---:|---:|---:|---|
| `shared_trunk` | `S` | `1,072,888` | `70.09M` | `206.61M` | `2048` / `3072` (`6144` larger ruler probe) |
| `shared_trunk` | `M` | `4,593,584` | `282.41M` | `841.00M` | `1024` / `1536` (`3072` larger ruler probe) |
| `ego_only` | `S` | `250,344` | `10.59M` | `29.43M` | `2048` / `3072` (`6144` larger ruler probe) |
| `ego_only` | `M` | `1,382,160` | `47.06M` | `137.57M` | `1024` / `1536` (`3072` larger ruler probe) |
| `per_seat` | `S` | `1,027,080` | `308.53M` | `898.32M` | `512` / `768` (`1536` larger ruler probe) |
| `per_seat` | `M` | `5,017,480` | `1,999.46M` | `5,943.83M` | `512` / `768` (`1536` larger ruler probe) |

- `ppo.py`: `ArenaPPOConfig` (JSON-serializable), `CompactRolloutBuffer` (lossless
  append-only `int16` body tape), clipped PPO learner with GAE (`gamma=0.999`,
  `gae_lambda=0.97`, `r_t = L_t / 1000` per alive tick), `--hours` +
  `--expected-steps-per-sec` pre-start training plan (`ARENA_PPO_PLAN`), per-update
  create-only rotating JSONL logs under `output/train_logs/`, optional periodic eval
  hook (`eval_every_updates=0` OFF by default), optional `LateStateReservoir` resets
  (`late_reset_enabled=False` OFF by default), cooperative signal stop + clean
  `max_update_seconds` (`<= 20` min) / wall-budget guards, scratch `work_dir` periodic
  checkpoint retention (`keep=3` pruned only outside `output/`), create-only hard-link
  publishing into `output/checkpoints/` (`final.pt` on clean stop, newest verified
  periodic checkpoint on abort/guard stop, plus periodic snapshot hard-links every
  `2400s` (`40` min) so hard runtime kills lose `<= 45` min with zero unlinks under
  `output/`), and config-checked resume (allowing speed/operational overrides:
  `actor_chunk_envs`, `ring_chunk`, `compile_model`).
- `ruler.py`: Signal-safe throughput & VRAM ruler across
  `--net-arch in {ego_only, shared_trunk, per_seat} x {S, M} x E in {1024, 2048, 4096, 8192}`
  (cheapest cells first), `1`-PPO-epoch learner-share probe cell (`_ep1`), learner
  settings report (`ppo_epochs`, `minibatch_size`, `world_ticks_per_minibatch`),
  `>= 200`-tick untimed `greedy_food` burn-in, zero-sync `nanquantile` world maturity
  telemetry, 6-stage split + `untracked_seconds`, per-stage peak VRAM + projected `train`
  (`T=256`) rollout-buffer and peak VRAM bytes, OOM halving retries, dual minibatch
  probes, single-world (`batch=6`) latency + exact ATen FLOPs/row, `--compile` (`>= 180s`
  compile budget projection), incremental `cells/<cell_id>.json` exports, and
  `run_canary_suite` (zero-update verification of staged resume checkpoints + tiny
  2-update `shared_trunk` train).
- `prepare.py` / `job.sh` / `requirements.txt`: Three-mode Colab CLI packet builder
  supporting `L4`, `A100`, and `H100` (`T4` excluded; exports
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).

## Packet Preparation & Continuation Commands

### 1. Fresh `L4` Canary, Multi-Arch `L4` Ruler, and 6-Hour `L4` Train

```bash
# Step 1: Prepare L4 family canary (14 min base / 28 min timeout)
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.arena_ppo.prepare \
  --packet ../snake-dqn-artifacts/big-lever/exp/argon-20261026/pkt-arena-ppo-canary-l4-attempt-04 \
  --commit <merged_commit_sha> \
  --mode canary \
  --gpu L4 \
  --attempt attempt-04

# Step 2: Prepare multi-arch L4 ruler (ego_only,shared_trunk,per_seat x {S,M} x E + 1-epoch probe)
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.arena_ppo.prepare \
  --packet ../snake-dqn-artifacts/big-lever/exp/argon-20261026/pkt-arena-ppo-ruler-l4-attempt-04 \
  --commit <merged_commit_sha> \
  --mode ruler \
  --gpu L4 \
  --net-arch ego_only,shared_trunk,per_seat \
  --canary-receipt ../snake-dqn-artifacts/big-lever/exp/argon-20261026/arena-ppo-canary-l4-attempt-04/output/receipt.json \
  --attempt attempt-04

# Step 3: Prepare 6-hour L4 train packet (default --net-arch shared_trunk)
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.arena_ppo.prepare \
  --packet ../snake-dqn-artifacts/big-lever/exp/argon-20261026/pkt-arena-ppo-train-l4-attempt-01 \
  --commit <merged_commit_sha> \
  --mode train \
  --gpu L4 \
  --net-arch shared_trunk \
  --net-size S \
  --num-envs 2048 \
  --hours 6 \
  --expected-steps-per-sec <measured_ruler_steps_per_sec> \
  --canary-receipt ../snake-dqn-artifacts/big-lever/exp/argon-20261026/arena-ppo-canary-l4-attempt-04/output/receipt.json \
  --attempt attempt-01
```

### 2. Continuation / Resume Sequence (`colab_canary.check` Input Binding)

Because `research.colab_cli.canary.check` binds a science manifest's exact `inputs`
(count, destination, and SHA-256 digests) to its canary receipt, every `--prior-output-dir`
continuation first prepares and runs a continuation `canary` staging those exact prior
artifacts, then binds the continuation `train` or `ruler` packet to that continuation
canary receipt:

```bash
# Train checkpoint continuation (attempt-02):
# 2a. Continuation canary staging the prior checkpoint (0-update load check + tiny 2-update train)
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.arena_ppo.prepare \
  --packet ../snake-dqn-artifacts/big-lever/exp/argon-20261026/pkt-arena-ppo-canary-l4-attempt-02 \
  --commit <merged_commit_sha> \
  --mode canary \
  --gpu L4 \
  --prior-output-dir ../snake-dqn-artifacts/big-lever/exp/argon-20261026/arena-ppo-train-l4-attempt-01/output \
  --prior-kind checkpoint \
  --attempt attempt-02

# 2b. Continuation train bound to the attempt-02 continuation canary receipt
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.arena_ppo.prepare \
  --packet ../snake-dqn-artifacts/big-lever/exp/argon-20261026/pkt-arena-ppo-train-l4-attempt-02 \
  --commit <merged_commit_sha> \
  --mode train \
  --gpu L4 \
  --hours 6 \
  --expected-steps-per-sec <measured_ruler_steps_per_sec> \
  --prior-output-dir ../snake-dqn-artifacts/big-lever/exp/argon-20261026/arena-ppo-train-l4-attempt-01/output \
  --prior-kind checkpoint \
  --canary-receipt ../snake-dqn-artifacts/big-lever/exp/argon-20261026/arena-ppo-canary-l4-attempt-02/output/receipt.json \
  --attempt attempt-02

# Partial ruler continuation (attempt-02):
# 2c. Continuation canary staging prior verified cells/*.json
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.arena_ppo.prepare \
  --packet ../snake-dqn-artifacts/big-lever/exp/argon-20261026/pkt-arena-ppo-canary-ruler-l4-attempt-02 \
  --commit <merged_commit_sha> \
  --mode canary \
  --gpu L4 \
  --prior-output-dir ../snake-dqn-artifacts/big-lever/exp/argon-20261026/arena-ppo-ruler-l4-attempt-01/output \
  --prior-kind ruler_cells \
  --attempt ruler-attempt-02

# 2d. Continuation ruler bound to the continuation canary receipt
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.arena_ppo.prepare \
  --packet ../snake-dqn-artifacts/big-lever/exp/argon-20261026/pkt-arena-ppo-ruler-l4-attempt-02 \
  --commit <merged_commit_sha> \
  --mode ruler \
  --gpu L4 \
  --prior-output-dir ../snake-dqn-artifacts/big-lever/exp/argon-20261026/arena-ppo-ruler-l4-attempt-01/output \
  --prior-kind ruler_cells \
  --canary-receipt ../snake-dqn-artifacts/big-lever/exp/argon-20261026/arena-ppo-canary-ruler-l4-attempt-02/output/receipt.json \
  --attempt attempt-02
```
