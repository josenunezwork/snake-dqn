# Spec: Phase 0 — search-teacher premise test (owner APPROVED 2026-10-19)

Owner: Claude (plan/review/operate). Implementer: Sol. Advisor: Fable (2026-10-19).
Owner approval in chat 2026-10-19 ~12:30 PT: "APPROVE" (for Fable's search-as-teacher path, Phase 0 first).

## Why

Every bare-network lever since B8 was null/negative (M5, M6, T6, T23; S1 pending). Repo diagnostics
(T5/T5b/T8/T9) show no representation predicts death-within-300 above AUC ~0.74-0.77 on late states;
fatal vs surviving branches are indistinguishable at the decisive action and separate 50-80 steps later
via connected free-region size / arrival-race area at horizons 128-256, in pockets bounded mostly by the
hero's own body. Hypothesis: at L >= ~600 survival is decided by search, not by a reactive policy.
Phase 0 tests whether an exact-simulator search policy beats bare B8 by a margin large enough to be worth
distilling. If it does not, B8 is the bare ceiling and training spend stops.

B8 = ego2s-bv8 dueling CNN (`src/model/ego2s_network.py`, `obs_spec.py`), checkpoint used by S1:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth`
(SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8` for online weights per S1 prereg;
verify which file/field that hash covers).

## Deliverables (in order)

1. **Search teacher** (`research/phase0_search_20261019/`): at a hero decision, for each legal action,
   N rollouts of H steps on a bit-exact copy of the live world (use the existing vectorized GridBatchSim /
   numba sim and T4b snapshot-restart tooling — find them; the sim must be the same one the strict gate's
   evaluator uses, or prove bit-identical stepping on a fixture). Hero follows B8 greedy with small epsilon
   inside rollouts; opponents follow their real policies (scripted exact; frozen nets batched). Score =
   realized mass integral over H + B8 masked-max Q at the leaf (scaled; document the scale). Choose argmax;
   deterministic RNG per (world, decision, action, rollout). Search only when gated: hero length >= 600 OR
   B8 Q-margin (best minus second-best legal) below a threshold set on a disjoint calibration set so ~10% of
   decisions are searched; B8 acts otherwise. N, H, epsilon, threshold are pre-registered (suggest N=8-16,
   H=128-256; pick from a timing+pilot on DISJOINT worlds only).
2. **Timing preflight** (tiny, Mac, run it yourself — it is local CPU, not a live VM): measure per-decision
   search cost at hero length ~200, ~800, ~1500 with realistic opponents, worker-parallel scaling on this
   Mac, and project Phase 0 total core-hours. Also measure what a single-hero serving planner would need per
   41 ms tick (fallback serving question) — report, do not optimize yet.
3. **Pre-registration** `research/phase0_search_20261019/PREREGISTRATION.md` (DRAFT until Claude marks
   approved): fresh, collision-audited H5000 world bank — 96 scripted + 32 frozen + 32 mixed — paired
   search-policy vs bare B8 on identical worlds; endpoint H5000 full-horizon mass integral (dead frames 0),
   per-mix and equal-mix; survival per mix. Decision: **GO** if scripted survival (no-death fraction) of the
   search policy >= 0.66 AND pooled paired mass effect >= +80 with one-sided 90% lower bound > +40;
   **STOP_TRAINING** if UB90 of pooled mass effect < +40 or scripted survival < B8's + 0.05; else
   **INCONCLUSIVE**. One look; no interim efficacy reads; deterministic, resumable per-world partials, merge
   byte-identical to serial. Mirror the evaluation conventions of the existing strict gate where possible.
4. **Runner** that can execute on the Mac (bounded worker count per compute policy: up to ~6 threads, AC +
   lid open guard, thermal guard) and on a RunPod CPU pod (same code, per-world workers, result identity
   check vs Mac on a small replay subset — float ties: see repo notes on Mac<->pod BLAS tie flips; make the
   identity check tie-aware). Leave exact launch commands for Claude; do not launch Phase 0 yourself.
5. Tests: sim bit-exactness of snapshot/restore and rollout stepping; deterministic search choice;
   gating; merge identity; world-bank collision audit; a 2-world smoke end-to-end. Black/isort/flake8.

## Rules

Work only in `/Users/josenunez/Projects/ml/snake-dqn-search-teacher` (branch
`t32-search-teacher-20261019`); commit often and push that branch. Do NOT touch
`/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (S1 is running from it), no Colab, no RunPod,
no tmux, no serving changes, no publishing. Local Mac CPU work for timing/tests/smoke is allowed (keep it
<= ~6 threads, under a few minutes per run). Final message: timing table + projected core-hours and
cost on Mac vs RunPod 32-vCPU, chosen N/H/eps/threshold with justification, the prereg path, exact run
commands, and anything unverified.
