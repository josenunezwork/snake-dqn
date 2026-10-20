# Spec: make the search teacher fast enough for Phase 0 AND distillation

Owner: Claude. Implementer: Sol. Follows `2026-10-19_phase0_search_teacher.md` (owner-approved path).

## Problem (from your PREFLIGHT.json, ea87eaa4)

One searched decision (N=8, H=128, eps .02, 6 actions -> 48 rollouts x 128 steps ~ 6.1k hero steps)
costs ~4.1-5.4 s on one Mac core (~1.2-1.5k rollout-steps/s), vs GridBatchSim's ~30k world-frames/s/core.
Length gate alone searches 43.4% of decisions -> Phase 0 ~337 core-hours (89 h Mac, pod 3.51x < our 4x
RunPod rule). Distillation needs ~1-2M searched decisions per round, so at this speed the whole path is
infeasible. Serving-planner fallback is ~100-130x over the 41 ms tick. Teacher throughput is the crux.

## Required outcome

1. **Profile first** (cProfile/py-spy or manual timers) one searched decision at L~800 for each mix:
   split time into sim stepping, observation building (ego2s rasters + scalars + v8 features), B8 forward,
   frozen-opponent forwards, scripted-opponent logic, cloning/snapshot, Python overhead. Record in
   `research/phase0_search_20261019/PROFILE.md` with numbers.
2. **Optimize the dominant costs**, keeping results bit-identical to the current teacher on fixtures
   (or prove any change is a pure reordering with identical chosen actions and identical scores up to
   documented float tolerance; tie handling explicit):
   - lockstep all 48 (or N x actions) rollouts as one batched world set: one batched observation build and
     ONE batched B8 forward per step for all rollouts; same for each frozen opponent network;
   - vectorized/numba observation building over the batch;
   - optional GPU path (CUDA if available, else MPS on the Mac) for the batched forwards, with the CPU path
     kept as the reference; report both;
   - avoid per-step Python loops over rollouts; reuse buffers; clone state once per decision.
   Target: >= 20x per searched decision on Mac CPU; report what GPU batching adds.
3. Re-measure with the same PREFLIGHT protocol; update the forecast for Phase 0 (Mac, RunPod 32-vCPU CPU,
   and a Colab L4 GPU option with ~12 vCPU) and for one distillation round (1M searched decisions), and the
   serving-planner per-tick cost. If a cheaper but equivalent search (e.g., H=96, N=6, or a tighter gate
   L>=800) would be needed to hit feasibility, compute its cost but do NOT change the registered values;
   list it as a proposal for Claude.
4. Tests: batched vs reference teacher identical choices/scores on fixtures for all mixes; determinism;
   existing 31 tests stay green. Black/isort/flake8. Commit + push `t32-search-teacher-20261019`.

## Rules

Only `/Users/josenunez/Projects/ml/snake-dqn-search-teacher`. No Colab, RunPod, tmux, serving, or the
colab-safety worktree. Local Mac CPU/MPS runs <= 6 threads, a few minutes each. Final message: profile
table, speedups, new forecasts, proposals, unverified items.
