# T16 Colab smoke-02 results and diagnosis (2026-10-17)

Throughput evidence only; not gate, quality or serving evidence. Runtime: Colab Pro+ A100-SXM4-40GB,
12 vCPU / 6 physical cores, Python 3.12 venv, torch 2.9.1+cu126, runner pin `6b7e280`. Parity tests
passed (2/2). Raw data: Drive `t16_colab/smoke-02/` (results JSON, per-config telemetry and logs).
Diagnosis by two local Fable reviewers (GPU/H100 and RL systems; reports reconciled and agreed).

## Results (150 s per config, steady-state window)

| config | tr/s | vs Mac | upd/s | cpu% | gpu% mean/max | bound by |
|---|---|---|---|---|---|---|
| hero_w128_a2 | 4,062 | 0.77x | 15.9 | 21 | 29/61 | actors |
| hero_w128_a4 | 8,014 | 1.5x | 31.3 | 39 | 45/73 | balanced |
| hero_w128_a8 | 9,138 | 1.7x | 35.7 | 60 | 53/75 | learner (+CPU contention) |
| selfplay_w64_a2 | 11,086 | 2.0x | 43.3 | 24 | 54/75 | learner |
| selfplay_w128_a2 | ~11,500 | 2.1x | ~45 | 17 | 42/70 | learner (row invalid: actor did not exit cleanly) |
| selfplay_w128_a4 | 11,341 | 2.1x | ~44 | 15 | 34/71 | learner (row invalid: same) |

Mac baseline: 5,283 hero / 5,459 self-play tr/s, learner-bound at ~21 upd/s.

## Diagnosis

- **Right metric is updates/s, not transitions/s.** `learner_m3c.py:655` fixes 256 transitions per
  update (replay ratio 3 gradient samples per transition). Honest 4x = >= ~84 upd/s. Colab = 2.1x.
- **Self-play is learner-bound by the learner's single host thread, not the GPU.** ~19-23 ms/update:
  rejection sampling, per-ring numba `unpack_into` from file-backed mmap rings into pinned staging
  (~22-25 MB/update), Python ingest, and a 100k-deque bincount every loop iteration
  (`train_m3c.py:3305-3308`). GPU work is ~8-11 ms and hides under it. CUDA graphs replay fine (1 graph,
  0 breaks).
- **More actors do not help** (a2 = a4; a4 doubles policy lag 4.4 -> 9.0). Hero configs are actor-bound
  (5 opponent evals + sim per hero transition) and cannot reach 4x on 6 cores.
- **H100 would not help today** (same vCPU class; GPU is a minority of each step). After the host fix the
  A100 ceiling is ~23-25k tr/s; H100 maybe 28-32k. Switch only if a post-fix profile shows kernel-busy > 80%
  with tr/s < 21.6k.

## Plan

1. Move gather off the learner thread: thread-pool the nogil per-ring unpacks / copies (cheap), GPU-resident
   packed ring with on-device gather + unpack as follow-up. Byte-identical batches required.
2. Ingest cleanup: collapse bincount to log cadence; no `tolist()` + deque per message.
3. Fix "actor did not exit cleanly" on w128 self-play.
4. Re-smoke on A100: selfplay_w128_a2/a3, `--actor-queue-frames 2-4`, >= 10 min, with `M3C_PROFILE`;
   report upd/s and policy lag beside tr/s.
5. Lane A league mode (N learner slots + frozen pool) is a prerequisite and must be re-measured.

Rejected as a "speed-up": `--update-every 2` (halves the replay ratio). `--batch-scale 4 --lr-scale sqrt` only
as a pre-registered arm after steps 1-2.
