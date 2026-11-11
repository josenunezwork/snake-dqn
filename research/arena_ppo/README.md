# Arena PPO (Stage 1 — From-Scratch GPU-Resident Full-Arena PPO)

Stage 1 implementation of GPU-resident PPO self-play on the exact mechanics-v2
`DeviceSim` (`83 x 145` rectangular arena at 1 cell = 1 sim cell), with zero
`vector61` / `v8` / BFS flood features in the training tick.

## Modules

- `runtime.py`: Canonical device normalization, pre-allocation CUDA lazy-init,
  single-SM native extension preloading (`move_cuda`, `event_cuda`), deterministic
  backend toggles, UTC `ARENA_PPO_PHASE` `START`/`END` stamps, and create-only
  `.sha256` + `.complete` artifact writers.
- `sim.py`: `TrainSim` wrapping exact mechanics-v2 `DeviceSim` (`train_mode=False`,
  `allow_respawn=True`, `hero_respawn=True`, `exact_mechanics=True`), 2-frame head
  history, legal boost action masks, and per-world opponent routing (`self`,
  `snapshot`, `scripted` `greedy_food`).
- `obs.py`: Zero-sync GPU/MPS observation builder (`13 x 83 x 145` allocentric arena,
  `13 x 47 x 47` heading-aligned ego crop, `6` scalars) and pure-NumPy CPU reference
  builder proven bit-identical across `>= 64` worlds x `200` ticks.
- `net.py`: Dual-stream `ArenaNet` presets `S` (`1,027,080` params) and `M`
  (`5,017,480` params) with `11 x 19` (`209`) arena self-attention tokens, small
  ego ResNet, masked 6-action policy head, symlog two-hot value head (`255` bins),
  and death-within-`{16, 64, 256}` auxiliary heads (`channels_last`, CUDA `bf16`
  autocast, optional `torch.compile`).
- `ppo.py`: `ArenaPPOConfig` (JSON-serializable), `CompactRolloutBuffer` (lossless
  append-only body tape avoiding multi-TB raster storage), clipped PPO learner with
  GAE (`gamma=0.999`, `gae_lambda=0.97`, `r_t = L_t / 1000` per alive tick),
  periodic (`<= 20` min) + `SIGTERM`/`SIGUSR1` atomic checkpointing, and verified
  deterministic resume.
- `ruler.py`: Throughput & VRAM ruler sweeping `E in {1024, 2048, 4096, 8192}` x
  `net in {S, M}` (`collection_only` and `integrated`), per-stage CUDA-event split,
  OOM catch-record-continue, `ARENA_PPO_MAX_WALL_SECONDS` budget guard, and family
  canary export probes.
- `prepare.py` / `job.sh` / `requirements.txt`: Colab CLI packet builder for `ruler`
  (`short-canary` `<= 28` min) and `train` (resumable session with `--hours` and
  `--prior-output-dir`).
