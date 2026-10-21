# Big-lever research rounds — charter (owner request, 2026-10-20 ~23:00 PT)

## Owner's words (verbatim)
"can you have fable consistently call upon and prompt 10 astra agents at a time and judge their calls. i want 10 rounds
of this with notes so we can move on to a big lever push based on ai research that can help in slitherio game like
environment that takes advantage of gpus and cuda we can use on colab and we want to use colab credits with no real
budget we can waste them all in a week if we think we can move fast this way"

## Goal
Find, design and de-risk ONE (or a small portfolio of) big-lever push(es) — grounded in AI research that works in
slither.io-like multi-agent, partially observable, continuous-steering, growth/kill/corpse environments — that uses
GPUs/CUDA on Colab at scale to produce a served snake that clearly beats bare B8 s3 (the current Watch hero) on mass at
equal-or-better survival, with a "beats B8" claim the owner can trust (Mac strict gate + serving qualification + owner
release OK stay non-negotiable). Speed matters more than thrift.

## Resources
- Colab Pro+: ~1,140 compute units, **no real budget** — the owner is fine spending all of it within a week if it buys
  speed. L4 ~1.54-2 u/h, A100 and H100 available (~5x+ L4 price), up to 3 concurrent assignments on the account (verify
  in research/colab_cli; ask whether that cap can be raised). Qualified job transport: research/colab_cli (CLI supervisor).
- Mac (Apple M5 Pro, 18 cores, no CUDA) runs the served game, strict gates, paced timing.
- Builders: Sol (Codex gpt-6.1-sol) implements in hours-to-a-day per component; Claude operates jobs; Fable judges.

## Context to read (repo /Users/josenunez/Projects/ml/snake-dqn, main)
AGENTS.md; docs/STATUS.md; docs/research/final_run_20261019/{CAMPAIGN_LOG.md,FINAL_PLAN.md,SUCCESS_CONTRACT.md};
docs/ml_redesign_blueprint_2026-07.md (PQN + ego-raster + vectorized sim, never fully run);
docs/research/advice/2026-10-21_direction/FABLE.md; docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md;
docs/research/advice/2026-10-20_a100/ (GPU experts, why A100 was "no" for the current host-bound teacher);
docs/research/specs/2026-10-21_gpu_sim_spike.md (GPU sim spike in progress); research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md;
src/simd_env/ (grid_sim.py numba sim, batch_sim.py reference), src/scripts/train_pqn.py; docs/research/t3_ceiling_test_2026-10-15.md
(T3-dense search acting +92 pooled / +122.8 on B8 s3); T23 league KILL; B8 death census.
Current campaign (defensive T34 search teacher, Round 0 verdict Wed) continues in parallel; these rounds plan what comes next.

## Process
10 rounds. Each round: one fresh Fable judge reads this charter + all prior JUDGMENT files, writes the round's QUESTION
(an open question, facts, what the previous round left unresolved — never a list of options to pick from) and 10
personas, then Claude launches 10 Astra advisors in parallel (`run_round.sh`), then the next Fable judges them.
Rounds should converge: early rounds diverge (map the space: GPU-native sims like PufferLib/EnvPool/JAX/Brax/Madrona/
Isaac-style, PPO/PQN/IMPALA at billions of steps, self-play/league/PSRO, MuZero/Gumbel-AlphaZero/sampled search,
world models/Dreamer, population-based training, distillation of search, opponent modelling, curriculum…), middle rounds
attack feasibility, cost, parity/trust and eval validity, late rounds produce a buildable, staged plan with kill rules.
Round 10's judge writes PLAN.md: the big-lever push, staged, with Sol work packages, Colab spend per stage, timelines,
decision points, and what must be true for a release claim.
