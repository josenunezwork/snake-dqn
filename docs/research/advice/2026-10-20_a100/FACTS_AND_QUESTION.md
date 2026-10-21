# A100 question — facts brief (2026-10-20 ~17:45 PT)

## Goal
A better served snake agent than the current Watch hero (bare B8 seed 3, released 2026-10-17), proven by the
usual strict gate on the Mac, **as soon as possible in wall-clock time**. Colab units are prepaid and the owner
wants them spent (~1,157 u left; target: spend all by ~2026-11-20, ~39 u/day) on work that produces decisions.

## Open question
Would using Colab **A100** runtimes (instead of / alongside the L4 runtimes we use now) shorten the time until we
have a better snake agent — and if so, for which part of the work and how; if not, what would?

## Facts (verify anything in the repo; sources below)
- **Agent:** B8 = ego2s-bv8 dueling CNN (`src/model/ego2s_network.py`, ~1.64M params), served bare (no veto).
  Every bare-network lever since B8 (fine-tunes, league T23, S1, M-series) was null or negative. Remaining deaths =
  self-enclosure at long length, decided 50–80 steps before death; reactive prediction AUC ~0.74–0.77.
- **Approved path (T34, "search teacher"):** exact-sim rollout search teacher (GridBatchSim snapshot restart, N 8–16
  rollouts × H 128–256, B8 rollout policy, leaf = B8 Q), gated to long/low-margin states, then DAgger-distill into
  the same B8 CNN for 3–5 rounds, serve the bare student via the strict gate. Fallback: serve the planner.
- **Running now:** Phase 0 premise test (Mac, 6 CPU workers, 100/160 worlds, ETA Wed 10-21 ~10:00–13:00 PT; GO =
  scripted survival ≥ 0.66 AND pooled mass ≥ +80, LB90 > +40). Round 0 (teacher acting vs bare B8, 240 paired
  worlds, 3 chunks serial on 1 Colab L4 lane; chunk 000 ~3.5 h; ratified; 48 h dispatch deadline ~10-22 16:45 PT).
  Path B paced retention study ratified, Mac-only, starts after Phase 0 (≤ 40 h).
- **Measured speed (where time goes):** teacher/benchmark work is ~95% single-threaded CPython (state digest,
  deepcopy, v8 opponent hooks, raster glue); the L4 GPU sits at 2–8% util; CUDA batch-1 forward was *slower* than CPU
  (4.35 ms vs 0.93 ms) because of sync. After the engine fan-out fix (12 single-thread processes per L4 runtime,
  W1, O(bytes) digest): **57.2 / 54.4 completed worlds/h per L4 lane, ~0.036 u per teacher world**; ~16–20k rows and
  ~8.7 MB per world. Colab CPU-only runtime had nproc=2 (unusable). Phase 0 teacher: ~4.5 s per searched decision on
  1 Mac core before a 3.4–4.2× speedup (B8 inference dominant there); 1M searched decisions ≈ 129 Mac-hours — the
  distillation step's label volume is the known bottleneck. CUDA throughput of the *search teacher rollouts* has
  never been measured.
- **Colab facts:** Pro+, official Colab CLI via a Mac supervisor (`research/colab_cli`, qualified), max 3 concurrent
  runtimes, ≤ 24 h each. Rates (third-party, not official): L4 ~1.7 u/h, A100-40GB ~5.4, A100-80GB ~7.5, G4 ~8.7.
  The **host vCPU count / per-core speed of an A100 runtime is unmeasured** (L4 runtime: 12 vCPU ≈ 6 physical).
  The adopted 30-day compute portfolio (Fable judge) explicitly did not fund A100/G4/H100 for host-bound work.
- **Rules:** strict gates and serving qualification stay on the Mac; Colab is pre-approved; freeze designs before
  data; never relabel a registered gate.

## Sources
`docs/research/final_run_20261019/HANDOFF_2026-10-20.md`, `CAMPAIGN_LOG.md` (same folder),
`docs/research/advice/2026-10-20_t34_a5_speed_fable.md`, `..._t34_a5_speed_astra.md`,
`docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md`, `docs/research/specs/2026-10-19_phase0_*`,
`research/t34_round0/`, `research/colab_cli/`, `src/model/ego2s_network.py`, the GridBatchSim sources.
