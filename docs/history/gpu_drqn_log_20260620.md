# GPU DRQN run log — 2026-06-20

Goal: train a **recurrent (DRQN/GRU)** snake on a cloud GPU that beats the
feedforward champion (`saved_snakes/best_apex.pth`, "arena-0.8": mass ~56–60 vs
frozen S1 / ~63 vs pre). Recurrence is the credible path to the two things the
champion can't do: **crowd-aware survival** and **kills** (which need planning/memory).

## Current state (what we're doing right now)
- Training DRQN on **RunPod, 1× RTX 4090** (16-core/188 GB box), single env.
- Config `configs/gpu_drqn_best.yaml`: warm-started, GRU, `hidden 512 / gru 256`,
  `sequence_length 32`, `burn_in 8`, `batch_size 256`, `buffer 100k`, boost-fix
  reward (`boost_segment 3.0`), dense 8-snake arena (1160×664), 800 episodes.
- Launch: `bash runpod_setup.sh` (or explicit `main.py --headless --num-envs 1 --load <ckpt> --config configs/gpu_drqn_best.yaml --device cuda --episodes 800`).
- Saves `env_0_latest_snake.pth` every 30s (current policy). Pull it → eval on Mac:
  `tournament_eval.py` (architecture-aware, 20-seed paired) vs champion.
- **Pending:** no strong DRQN checkpoint has been *evaluated on the benchmark yet*
  — training reward (~458) is on the dense arena, NOT comparable to champion mass 56.
  Verdict (does DRQN beat arena-0.8?) is still OPEN.

## What WORKS
- **DRQN on GPU, warm-started → trains fast & strong.** Avg reward 8→**458**, best
  **1143**, food ~**1000/episode**, ε→0.02, still climbing (~ep167). The warm-start
  (reuse champion `feature_layer` + dueling streams, only GRU fresh) is the key to
  fast convergence. ~85–95 FPS single-env on the 4090.
- **GPU makes recurrence viable** (cuDNN GRU + the box's CPU for env-stepping).
- **Periodic latest-save** (every 30s, unconditional) — added so the live policy is
  always retrievable.
- **Headless no longer needs PyQt5** (lazy import) → clean server install.
- **Architecture-aware tournament eval** handles GRU vs feedforward, paired seeds, CI.

## What does NOT work / hard constraints
- **CPU DRQN from scratch: not viable** (mass ~4–6 vs champ ~58 after 40–90 min,
  3× confirmed incl. warm-start on CPU). Needs the GPU.
- **best-reward auto-save FREEZES** (gating bug): the trainer only saved
  `env_0_best_snake.pth` on `frame % save_interval == 0 AND >60s AND new-best`; it
  stopped firing after ~ep46 while training kept improving to best 1143 → the strong
  policy was never written. FIX: added the unconditional 30s `latest` save. **Always
  pull `env_*_latest_snake.pth`, not `best_snake.pth`.**
- **Can't extract a running process's weights on RunPod:** `ptrace_scope` is read-only
  `1`, so pyrasite/gdb injection is blocked, and the trainer doesn't save on SIGINT.
  → A run in progress can't be rescued; rely on the periodic save.
- **Multi-env (num-envs>1) on the local path = independent parallel SEEDS**, not a
  faster/better single model (no shared learner/buffer). Each ~2× slower (GPU/CPU
  contention). Only useful as a best-of-N hedge. Reverted to single env.
- **Bigger batch >256: no help** — we're DATA-limited (single env ~79 FPS), not
  compute-limited. Bigger batch re-chews the thin buffer → diminishing returns /
  overfit. 256 is the sweet spot. (Idle GPU can't be turned into a better single
  model via batch size; that needs more data generation, which DRQN can't pool.)
- **Bigger network (1024): deprioritized** — can't warm-start (arch change), and the
  model isn't capacity-limited (learns food=1000 at 512). Memory (seq length) > width
  for this problem.
- **Distributed Apex path doesn't support GRU** → DRQN is stuck on the slower
  single-env local path (can't fan out actors → one learner).
- **Kills still ~0** even in the strong DRQN (occasional 1). Kills are structural
  (need trapping/planning); longer GRU memory is the current bet, unproven.

## Infra fixes landed this effort (committable)
- `SnakeFactory` passes `use_gru` from config (DRQN was silently feedforward).
- Headless workers re-init config from `SNAKE_DQN_CONFIG` env (overrides reach spawns).
- `ApexPolicy.load_state_dict` skips TD/reward contract when `training=False`
  (inference/eval/GUI load any checkpoint).
- GUI `--eval` = true read-only watch mode (no train/auto-save); `ai_snake.cleanup`
  tolerates inference policies; GUI runtime controls (speed/ε/model/env sliders).
- New: periodic `latest` checkpoint save; `pydantic` added to requirements;
  `rebase_checkpoint.py`, `tournament_eval.py`, `ensemble_eval.py`.

## Key files / artifacts
- Champion: `saved_snakes/best_apex.pth` (+ read-only `champion_arena08.pth`).
- DRQN warm-start seed: `saved_snakes/drqn_warmstart.pth` (champion features + fresh GRU).
- GPU bundle: `~/Downloads/snake_gpu_bundle.tar.gz` (code + configs + 2 ckpts + setup).
- Configs: `gpu_drqn_best.yaml` (current), `drqn_warmstart.yaml`.
- Eval (run on Mac): `src/scripts/tournament_eval.py` vs frozen S1 + pre, 20-seed paired.

## Next steps
1. Let the current run reach strong/plateau (~30–45 min), pull `env_0_latest_snake.pth`.
2. Eval it on Mac vs champion (paired, both opponents). Promote only a CI-separated win.
3. If it wins → new champion. If head/self deaths still high → richer state (more enemy
   slots) is the next lever. If kills still 0 → that's a harder, separate problem.

## VERDICT on the v1 DRQN run (gpu_drqn_best.yaml) — LOST, and converged
Benchmark (tournament_eval, hero vs frozen S1, paired): the warm-started DRQN
**converged to a worse-than-champion optimum and lost**:
- snap3 (111k) mass 13.7/19.9 vs champ 56/63 (full 20-seed, both opponents) — SIG LOSS.
- snap3 → snap4 → snap5(newest) all **identical**: mean_mass ~22, survival ~0.40,
  deaths/ep ~1.0, kills 0 (fast eval: 8 seeds × S1 × 1500f vs champ 49.0 / surv 0.99).
- It is CONVERGED (22.0 → 22.1) → more training of this config = same 22.

**Root cause (diagnosed):** boom-bust gorge-and-die. The v1 reward made reckless death
net-POSITIVE — `food_base 3.0` rewards EATING (flow, not held mass), `death -11 FLAT`
(`death_length_scale 0`), `survival 0.01`; trained on a food-dense 8-snake board
(1160×664/230-food, denser than the benchmark) → optimal policy is to gorge then die
repeatedly → survival 0.40. (Self-play is NOT the cause; the champion also self-played
and survives 0.99.) The training reward (~458, food ~1000) was a flow/dense-arena
illusion — same eval-pitfall as the original boost-abuse, new costume.

## v2 FIX — `configs/gpu_drqn_v2.yaml` (designed via 4-lens panel + adversarial synthesis)
Flip boom-bust → hold-mass by punishing dying-while-big and matching the benchmark:
- `food_base 3.0→1.0` (stop paying for flow), `survival 0.01→0.06` (pay for stock),
  `death -11→-12`, **`death_length_scale 0→3.0`** (snake.py death path is UNCLAMPED:
  len100 death ≈ -48), geometry → **benchmark exactly** (1450×830, 6 snakes, 250/300),
  `LR 0.0003→0.0002` (gentle transition off the converged seed). Keep hidden 512 / gru
  256 / use_gru / gamma 0.99 / n_step 3 (resume-locked), seq_length 32, batch 256.
- Warm-start = **snap4** (`env_0_latest_snake.pth`): rebasing the reward contract changes
  the objective, so snap4 is no longer at an optimum; its trained GRU starts ahead of a
  fresh-GRU warmstart. (3/4 lenses preferred drqn_warmstart.pth; synthesizer + user chose snap4.)
- Resume needs the seed's reward_contract rebased to the new values
  (`rebase_checkpoint.py`). VERIFIED locally: config validates, contract matches with ZERO
  mismatches, training-mode load passes, n-step invariant = -1.81 < 0.
- **Early kill-criteria** (first ~15 min): NaN/diverging loss → abort; mean reward stuck
  < -8 for 10+ windows → death scale too strong (drop to 2.0, re-rebase); episode_length
  collapses < ~50 and stays (turtling/starve) → food_base too low (raise to 1.5);
  env_0_latest_snake.pth mtime not advancing → save broken. Healthy: reward trending up
  toward 0+, length climbing while deaths/ep fall. First real benchmark read at ~30-50k
  frames (~20-40 min); gate = survival > 0.9 AND mean_mass > 49.
