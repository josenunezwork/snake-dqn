# Spec: distillation-scale search teacher (many-worlds batched) — T33

Owner: Claude. Implementer: Sol. Advisor: Fable (2026-10-19 ~15:00 PT). Owner approved: "Have Sol build the
many-worlds batched teacher; measure on one Colab L4 ~20 min; A100/H100 only if GPU-bound; ask Fable".

Base: Phase 0 teacher in `research/phase0_search_20261019/` (read PROFILE.md, SPEED_REPORT.md, PREFLIGHT.json).
**Phase 0 is RUNNING from the other worktree; never touch `/Users/josenunez/Projects/ml/snake-dqn-search-teacher`
or its artifacts.** Put new code in `research/t33_batched_teacher/` (import Phase 0 modules read-only).

Target: >= 1M searched decisions in a few hours on 1-3 Colab L4 (~100-200 decisions/s per L4).

## Ordered work (Fable's plan)

1. **Search-budget ablation (do first; it may cut work 4-8x).** Harness that, on ~300 searched decisions
   sampled from disjoint calibration worlds (NOT the Phase 0 selection bank; reuse the calibration/timing
   fixtures or generate fresh disjoint ones with the bank tooling + collision audit), compares each cheaper
   teacher to the full N8/H128 reference: action agreement and regret (reference score gap). Variants:
   H in {32, 64} with value truncation at the leaf using B8's dueling V/masked-max Q (`leaf_scale`);
   N in {4, 8}; common random numbers across the 6 first actions (same opponent/food RNG stream per rollout
   index); optional margin gate. Run a small smoke yourself; leave the full ablation command (~1 h on <= 6
   Mac workers) for Claude.
2. **Many-worlds lockstep teacher, host-first:** W worlds x actions x N rollouts stepped in lockstep with fixed
   batch shape (dead rows masked); rasters built by a numba `prange` kernel into ONE preallocated **uint8**
   buffer (cast to fp16 on device), scalars fp32; pinned host memory, one non_blocking H2D copy per step;
   frozen opponents batched the same way — profile and fix the ~0.23 s/decision V61 carry (likely Python
   per-row work); no tie-recheck / reference-rerun paths (labels do not need bit-identity). Configurable
   search params from step 1.
3. **B8 on GPU:** fp16 autocast + `torch.compile(mode="reduce-overhead")` (CUDA graphs) with fixed shapes;
   validate Q-argmax agreement vs fp32 eager on >= 1k states (>= 99.5% required). CPU/MPS fallback for Mac tests.
4. **Throughput benchmark job for one Colab L4** (Claude runs it via the CLI supervisor): ~20 min, batch sizes
   256 / 1024 / 4096 worlds, logging per-step host ms, H2D ms, GPU ms, decisions/s, GPU util; exports a small
   JSON. Package as a supervisor manifest (inputs by public release URL + SHA-256 like S1; if a needed
   checkpoint is not yet public, list file + size + hash for Claude to publish — do not publish yourself).
   Mac-preflight the real entrypoint on CPU/MPS with a tiny batch. Decision rule recorded in the README:
   host >= 60% of wall -> scale out to 3 L4 runtimes (and consider step 5); GPU >= 60% -> one 20-min A100 probe.
5. **Only if step 4 shows host-bound:** GPU observation build (ship per-world uint8 board once per step;
   crop 31x31 ego + 37x37 coarse via gather/unfold on GPU). Do not port the sim to GPU.

## Rules

Work in `/Users/josenunez/Projects/ml/snake-dqn-batched-teacher` (branch `t33-batched-teacher-20261019`);
commit often, push the branch. No Colab/RunPod/tmux/serving/publishing; no Phase 0 or S1 interference.
Local Mac runs <= 4 threads (Phase 0 holds 6, S1 eval will take 2), a few minutes each. Tests: correctness of
batched vs Phase 0 reference teacher on fixtures (same search params -> same choices except documented
fp16/tie cases), determinism, uint8 obs equality with fp32 builder. Black/isort/flake8.
Final message: ablation smoke + exact full-ablation command, Mac throughput, fp16 agreement, the L4 job
manifest + exact launch command, anything unverified.
