---
name: snake-dqn-m3c-trainer
description: |
  Use whenever you read, change, benchmark or review the M3-C multi-actor trainer of
  /Users/josenunez/Projects/ml/snake-dqn: research/redesign_m3c_20261008/train_m3c.py, learner_m3c.py,
  league_pool.py, obs_pack.py, async_checkpoint.py, or any launcher/notebook that builds a train_m3c command
  (research/t16_colab/run_smoke.py, mac_n3_baseline.sh, Colab job manifests). Covers the B8 recipe, every
  opt-in flag, the counters and telemetry, the invariants that must never silently change, and how to add a
  new flag with the required bit-identity tests. Also load snake-dqn-performance for throughput work.
---

# Snake DQN: the M3-C trainer (train_m3c.py + learner_m3c.py)

New Colab M3-C jobs are submitted only through `research/colab_cli` with a pinned manifest, uploaded inputs, and
hash-verified exports. `research/t16_colab` notebooks and Drive queue manifests are legacy Q1 support; do not use
them to launch new work.

`train_m3c.py` (~5.1k lines) is the trainer that produced B8 s3, the released no-veto Watch hero
(2026-10-17). Everything new is behind a flag; **defaults reproduce M3 bitwise** (checked by
`tests/test_m3c_trainer.py::test_defaults_equal_m3`, and `m3c_flags(default_args()) == {}`).
Read the module docstring (lines 1-140) first: it is the authoritative flag catalogue.

## The B8 recipe (what every production/smoke command uses)

Actor/recipe flags (`research/perf_actor_20261013/bench_actor.py:102` `B8_ARGS`,
`research/t16_colab/run_smoke.py:283-345`, `research/t16_colab/mac_n3_baseline.sh:49-60`):

```
--reward mass --mass-kappa 1.0 --kill-scale 0 --gamma-end 0.998 --gamma-anneal-end 8000000
--mixed-frac 0.25 --anchor-mode v8margin --onpolicy-labels --label-frac 1.0 --obs-version bv8
--v61-backend numba --learner-fast --batch-scale 2          # "R0": fp32, K=2
```
plus infrastructure: `--seed 3 --init <B8_s3 final.pth> --demo-data <m2b data> --out <fresh dir>
--device {mps|cuda} --actor-device <same> --worlds 128 --actors 2 --ring-dir <scratch>` and
optionally `--obs-pack`, `--gather-threads 3 --actor-queue-frames 2`, `--learner-seats 3
--league-b8-checkpoint <init>`. `--init` must be an `ego2s-bv8` checkpoint (bv8 = ego2s-b + 30 v8
scalars); `--onpolicy-labels --label-frac 1.0 --anchor-mode v8margin` are mandatory with bv8
(demo shards have only 30 scalars; no anchor term until `LABEL_ANCHOR_MIN_ROWS = 10_000`
labels are buffered, train_m3c.py:180).

Batch arithmetic (learner_m3c.py:88-91 and ~655-665): `AGENT_ROWS 384`, `ANCHOR_ROWS 128`,
`UPDATE_EVERY 128` (train_m3c.py:183); with `--batch-scale K=2` one update samples
**768 TD + 256 anchor rows per 256 ingested transitions** -> replay ratio 3. `TARGET_EVERY 1000`
updates and `--actor-sync-updates 25` are divided by K so they stay fixed in transitions.
`N_STEP = 5` (train_m3c.py:182), Double DQN, Huber delta 1 (`--td-huber-delta`), Adam lr 1e-4
(`LR`), grad-norm clip 10.

## Flag families (verified names; never invent flags)

- **Trainer mode:** `--actors 0` = synchronous M3 (bit-reproducible); `--actors K>=1` = multi-actor
  (`main_multi`, not bit-reproducible: policy lag). Most perf/M6 flags are **multi-actor only**.
- **Collection:** `--learner-seats {1,3,6}` (T19 static league; 1 == hero-only, 6 == old
  `--all-snakes-learner`; mutually exclusive with that flag; needs `--actors`), `--league-checkpoint-dir`
  (default env `SNAKE_CKPT_DIR`), `--league-b8-checkpoint`, `--mixed-frac`, `--frozen-frac`,
  `--segment-frames`, `--collection-profile {legacy,task-aligned-n1}` (needs seats 1), `--restart-bank`
  + `--restart-frac` (T4), `--namespace`.
- **Schedules:** `--schedule-package {restart,mature}` ("mature" fixes gamma .998, eps .01, anchor .1 from
  decision zero), `--gamma-start/--gamma-end/--gamma-hold/--gamma-anneal-end`,
  `--anchor-start/--anchor-floor/--anchor-hold/--anchor-floor-at`, `--eps-start/--eps-end/--eps-end-at`.
- **Reward/levers (M4-A, M6):** `--kill-scale`, `--death-value`, `--shield {none,v8}`, `--tight-cost`,
  `--tight-min-len` (default 600), `--death-mass-mult`, `--tight-telemetry`; label banks `--label-bank*`,
  return banks `--return-bank/--return-frac/...`; halts `--boost-halt-rate`, `--arm-halt-file`.
- **perf-learner (learner_m3c.py):** `--learner-fast` (T1, same math), `--learner-precision {fp32,fp16,bf16}`
  (needs fast; numerics change), `--batch-scale K` + `--lr-scale {none,linear,sqrt}` (CHANGES dynamics),
  `--update-every M` (CHANGES dynamics: halves replay ratio; not a speed-up), `--gather-threads N`
  (needs fast; 0 = serial oracle path), `--compile-learner {default,reduce-overhead,
  max-autotune-no-cudagraphs}` + `--compile-learner-scope {loss,forward}` (needs fast; eager fallback,
  not bitwise), `--compile-actor`.
- **perf-actor/ram/opp:** `--v61-backend {python,numba}` (env `M3C_V61_BACKEND`), `--v8-backend`,
  `--actor-impl {vectorized,reference}`, `--opp-forward {per-group,merged-cpu,merged-device}` (env
  `M3C_OPP_FORWARD`), `--obs-pack` (3,064 vs 12,203 B/row, byte-identical batches), `--demo-mmap PATH`,
  `--ring-dir` (default /dev/shm; macOS has none, pass a local dir), `--actor-queue-frames` (default 8),
  `--actor-device`.
- **Checkpoints/resume (CHECKPOINTS.md):** `--checkpoint-every` (weights-only, atomic rename, row in
  `manifest.jsonl`), `--full-state-every N` (needs `--ckpt-local-dir`), `--full-state-replay {none,full}`,
  `--ckpt-durable-dir` (Drive mirror, separate worker), `--resume-full-state DIR` (newest verified
  manifest; same recipe + fresh `--out`; `none` replay = refill, flagged `resumed-without-replay`, not
  bitwise). `--max-wall-seconds` is a weights-only stop (`stopped_for_time: true`), not a resume point.
- **Guards:** `--out` is create-only (refuses an existing dir); `--no-guard` is tests-only. A HALT writes
  `aborted.pth`, not `final.pth` (selection reads finals only).

## Counters and telemetry: how to read a run

`log.jsonl` row (train_m3c.py:4708-4740): `transitions` (= `learner_decisions`), `updates`,
`replay_rows`, `replay_rows_per_update` (must read 1024 at K=2), `rate` (tr/s), `env_s`, `learn_s`,
`eps/gamma/lambda`, `td`, `anchor`, `qmax`, `boost_rate`, `greedy_boost_rate`, `deaths_per_1k`,
`actors`, `policy_lag_mean/p95`, `frozen_forward_gpu_s`. `telemetry.jsonl` (`Telemetry.write`,
:2980-3030) adds `rss_gb_max`, the time split (`env_s`, `learn_s`, `wait_s`, `featurizer_s`, `v8_s`,
`opponents_s`, `forward_s`, `frozen_forward_s`), held-out Q calibration, `learner_wait_s`,
`policy_lag_histogram`, `ring_overrun_rows`, `queue_depth` (None on macOS), `ring_counts`.
Timers are **nested wall times**: never add a parent and its child.

- **Learner-bound:** `learn_s` ~ wall, `learner_wait_s` ~ 0, policy lag <= 1, adding actors/worlds does
  not raise `rate` (Mac: ~5.2k tr/s, ~21 upd/s at every actor count, perf_quiet RESULTS.md).
- **Actor-bound:** `learner_wait_s` grows, GPU% low, `rate` scales with actors; look at `frozen_forward_s`
  / `v8_s` / `featurizer_s` for the binder (A100 N=3 a4: frozen pool 709 of 1,063 actor-s).
- **The honest throughput metric is updates/s** (= tr/s / 256 at K=2), since the replay ratio is fixed.
  4x the Mac is ~84 upd/s. More actors raise policy lag (a4 doubled lag 4.4 -> 9.0 on Colab).

## Invariants that must never silently change

1. Defaults == M3 bitwise (`test_defaults_equal_m3`; `m3c_flags(default_args()) == {}`).
2. Replay ratio (rows per update / transitions per update) and sampling draw order; `--update-every` or
   smaller batches are recipe changes, label them so.
3. `log.jsonl` keys/values unchanged by default (new fields only when a flag is on).
4. Off == old, bitwise, for any new reward or actor lever (`tests/test_m6_reward.py` replays the pre-M6
   actor from git `BASE_REV = "836646c"` and compares frame messages + ring rows).
5. Gathered batches byte-identical across `--obs-pack`, `--gather-threads`, `--actor-impl`
   (`tests/test_obs_pack.py`, `tests/test_t17_gather.py` frozen pre-T17 oracle, `tests/test_perf_actor.py`).
6. Static-pool identity: SHA-pinned checkpoints, immutable seats within a segment, pool RNG seeded
   `SeedSequence([seed, 19_018, segment])` (train_m3c.py:3302).
7. Hash-bound docs/receipts are never edited; add a sibling erratum.

## Adding an opt-in flag safely

1. Default must be the old behaviour; add the arg name to `m3c_flags()` (train_m3c.py:1922) and, for
   learner options, to `LearnerOpts.from_args` + `learner_flags()` (learner_m3c.py:96-150, 1660).
   Validate combinations in `LearnerOpts.__post_init__` / `main()` with `SystemExit` messages.
2. Extend `test_defaults_equal_m3` so the new flag appears in `m3c_flags` only when set.
3. Write an oracle test: vendor the pre-change module (copy to `tests/fixtures/<tag>/` with a SHA-256
   pin, or `git show <rev>:<path>`), run both on the same seeded synthetic world and compare **bytes**
   (ring rows, staging blocks, actions, RNG state). Patterns: `test_t17_gather.py`, `test_t22_frozen_pool.py`,
   `test_m6_reward.py`. Off-path must be bitwise; on-path must change exactly the documented term.
4. Run a tiny real two-actor CPU smoke (`--actors 2 --worlds 8 --transitions ~12000 --device cpu`) and
   check clean actor exit codes, `final.pth`, `summary.json`.
5. Update the module docstring catalogue and, if the flag is a recipe change, say so in its help text
   ("CHANGES training dynamics").
6. `./venv/bin/python -m pytest -q -m 'not slow' -k '(obs_pack or m3c or t16 or t17) and not mps'`
   then black/isort/flake8 (line length 100).

## Mistakes that already happened

- Treating tr/s as the metric: self-play rows count every driven snake; compare upd/s and replay ratio
  (T16 smoke-02 diagnosis).
- `--update-every 2` proposed as a speed-up: it halves the replay ratio (rejected, T16 doc "Plan").
- A 100k-deque `bincount` ran every learner loop iteration (train_m3c.py ~3305, fixed in T17 P2).
- Actors did not exit cleanly on w128 self-play when queues were full at shutdown (fixed: stop-aware
  puts + continuous drain, commits 26573f65/fd3f0fc6); the smoke runner also failed to lift
  `policy_lag_mean` into rows and marked every row INVALID (compare.py bug, T16 doc).
- Colab P0 bench read a Mac-only checkpoint path (fixed 42dae602/09cd7c8d) - no `/Users/...` in jobs.
- Resume bugs: segment gamma, actor checkpoint boundary and multi-state validation on resume
  (10236da1, c53b5d37, e4f5a8fa); frozen pool models dropped across segments (99c27b0e); B8 raster
  cache not bound to frame (f15fb292). Any resume or cache change needs a restore-then-compare test.
- MPS `torch.compile` returns wrong conv-bias gradients on torch 2.9.1: the learner falls back to eager
  (`compile_execution: fallback_eager` is NOT compiled evidence).
- M3-C arms trained on 75% scripted / 25% frozen and never saw the evaluated "mixed" roster; scored
  with v8 on, every student looked like the incumbent (docs/research/m3c_m3d_learnings_2026-10-12.md).
  Score development runs **no-veto**, with `--mixed-frac 0.25`.
