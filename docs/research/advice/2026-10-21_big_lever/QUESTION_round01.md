# Big-lever research rounds — Round 1 question (divergent: map the space)

Judge: Fable (round-1 judge, 2026-10-20 ~23:45 PT). Charter: `../CHARTER.md` (read it first).
Repo: `/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only copy of `main`; start with `AGENTS.md`).
Prior-round notes will live at `../round-NN/JUDGMENT.md` — **there are none yet; this is Round 1 of 10.**
Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/` (sealed study).

## 1. Goal (from the charter)

Find, design and de-risk ONE big-lever push (or a small portfolio), grounded in AI research that works in
slither.io-like multi-agent, partially observable, growth/kill/corpse environments, that uses **GPUs/CUDA on Colab at
scale** to produce a **served snake that clearly beats bare B8 s3** on mass at equal-or-better survival, with a
"beats B8" claim the owner can trust. The Mac strict gate, Mac serving qualification and the owner's release OK are
non-negotiable. **Speed matters more than thrift: ~1,140 Colab units may all be spent within a week.** Rounds 1–3
diverge (map the space), 4–7 attack feasibility/cost/parity/eval validity, 8–10 converge on a staged, buildable plan
with kill rules (Round 10 writes PLAN.md).

The active T34 campaign (defensive exact-sim search teacher; Round 0 verdict due Wed) continues in parallel and is
**not** yours to amend. These rounds plan what comes **after/beside** it.

## 2. Verified facts (file:line in the repo unless stated)

### 2.1 The target: what "beats B8 s3" means
- Served Watch hero = **bare B8 s3** (checkpoint SHA prefix `3e649783`), no veto, in a *mixed* Watch6 roster against
  frp3-s12 + v8 opponents (`docs/STATUS.md:29-36`). Its claim was "more mass at equal survival", not "safer".
- Skill bar (fresh paired banks, opponents fixed, terminal-hero H5000 mass integral with dead frames = 0, equal-weighted
  scripted / frozen / mixed mixes, vs bare B8 s3): **pooled mass ≥ +40 with one-sided LB90 > 0; scripted no-death
  ≥ +0.03; per-mix survival guards (survival band v2); boost share bounded**; then the Mac strict gate (Tier 1 + Tier 2)
  on fresh banks (`docs/research/final_run_20261019/SUCCESS_CONTRACT.md:27-33`).
- Product bar (continuous-respawn mixed Watch6): hero deaths/1k lower with a paired 95% CI excluding zero
  (`SUCCESS_CONTRACT.md:38`); Watch6 first (`:44`); serving at **24 fps Watch6 / 12 fps Watch8/12, update p95
  ≤ 11.688 ms, zero overruns** (`:47`). Release = STRICT_PASS + SERVING_PASS + owner OK (`FINAL_PLAN.md` §8).
- Serving runs on the owner's Mac (Apple M5 Pro, 18 cores, **no CUDA**), CPU PyTorch; existing ego2s ceilings are
  24 fps through six driven AI and 12 fps through twelve (`docs/STATUS.md:46-47`). Whatever you propose must end as
  something this host can run inside those ticks (a bare net, or a labelled assisted controller that meets the
  same bars).

### 2.2 The current network and inputs (B8 = `ego2s-bv8`)
- `Ego2sNet` (`src/model/ego2s_network.py:36-62`): local raster `(7, 31, 31)` uint8 + global raster `(4, 37, 37)`
  uint8 + **60 float scalars** → two small conv towers (32-64-64 and 16-32) → FC → LayerNorm → fuse 256 → dueling
  V + A over **6 actions** (turn left/straight/right × normal/boost). Obs shapes: `src/model/ego2s_network.py:27`,
  `src/simd_env/ego_raster_bv8.py:3-8`. **1,638,263 parameters** (counted this session). Ego-centric, heading-rotated.
- 30 of the 60 scalars are the old v8 veto's per-action safety features computed eagerly as inputs
  (`ego_raster_bv8.py:3-8`): B8 was trained *with* v8 labels and is served *without* a veto.
- How B8 was trained (M4-A recipe, `docs/research/m4_phase_r_result_2026-10-13.md:8-13`): distilled/anchored DQN on
  the numba sim, mass reward κ=1, kill reward 0, γ → 0.998, 25% mixed training worlds, v8-label margin anchor with
  on-policy v8 labels, **20M transitions, 5 seeds, on the Mac**. Phase R vs frp3-s12+v8: +128.7 mass (5/5 seeds).
- Vector61 line (feedforward MLP incumbents frp3-s12, champion_a5; the opponents) is FROZEN (`docs/STATUS.md:67`).

### 2.3 The environment
- Arena 1450×830 px, segment 10 → **145×83 = 12,035 cells**; worlds are **6 snakes**; max_food 300; H5000 frames
  (`src/core/game_config.py:36-43`; gate profile sets 6 snakes). Relative actions, 6 outputs (`AGENTS.md` "Action Space").
- **Mechanics v2 is implemented and served** (corrected 2026-10-20): a corpse leaves at most **150 pellets**
  (`src/core/mechanics_constants.py:40-64`), pickup is one pellet per snake per tick, opponents respawn immediately,
  boost drops trail pellets (`docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md:168-172`).
- B8 s3's failure mode: **a large body dying late** — scripted census median death frame 3,283, median length 972
  (`…offense_vs_defense/JUDGMENT.md:160-163`); the fatal path is a self-bounded pocket built over ~50–100 decisions,
  with no clean early separator in any tested input; horizon AUC ≈ 0.92 at 30 decisions, ≈ 0.75 at 300
  (`docs/research/redesign_campaign_retrospective_2026-10-17.md`, T8–T10 record). B8 boosts ~0.12% of frames; its
  kills are mostly passive. The offense judgment ruled **defense (late survival) is the larger mass lever; harvesting
  corpses a smaller additive one; hunting is bounded by the 150-pellet cap** (`JUDGMENT.md:160-172, 199`).

### 2.4 Simulators and measured speed (the whole story of this project is throughput)
- `GridBatchSim` (`src/simd_env/grid_sim.py`): numba/NumPy lockstep batch sim, **cell-exact and CI-parity-pinned to
  the live game**. Still scalar CPython for: collision *resolution* on event envs, food spawning via per-env
  `random.Random` rejection loops, corpse drops, respawns, contested food (`grid_sim.py:28-47`).
- Hero-only with scripted opponents: 21–28k hero steps/s per process (`docs/STATUS.md:111`, M1). With frozen
  vector61+v8 opponents the **v8 opponent hook is 56.8% of worker time, B8 inference 13.5%, digest 11.9%, sim 3.4%**
  (`research/speed_s0_baseline/RESULT_MAC_CONTENDED_01.md`, pooled); 155 active row-steps/s single-thread Mac (contended).
- Colab L4 teacher production (12 single-thread procs): **57.2 / 54.4 worlds per lane-hour, 377 / 432 active
  row-steps/s per lane**, ~0.036 u per H5000 teacher world (`CAMPAIGN_LOG.md:59`); **12 L4 vCPUs ≈ 2 Mac cores**
  (`docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md:33,56`). The GPU is ~idle (B8 forward ≈ 0.3% of
  L4 fp32 peak, `docs/research/advice/2026-10-20_a100/gpu_expert_1.md` §2).
- RL on CPU pods (M3-B, 5×20M): ~5.9k transitions/s combined (`docs/STATUS.md:114`). Every RL/fine-tune run so far
  trained on 10^7 transitions, never 10^9.
- A NumPy `BatchSim` + `raster31v2` dual-scale raster + **PQN trainer** exist (`src/scripts/train_pqn.py:1-12`,
  `src/training/pqn_trainer.py`, `src/simd_env/gpu_featurizer.py` torch raster port, parity-tested) from the July
  blueprint that planned 150k agent-steps/s and "1B steps in 2.8 h" (`docs/ml_redesign_blueprint_2026-07.md:219-223`).
  **That campaign never ran at scale**; the blueprint is superseded as a plan but its code and literature survey stand.

### 2.5 GPU work that exists today
- `research/gpu_sim_spike/` (commits `179d6044`…`c4ed9745`, 2026-10-20/21): a **torch device-tensor port of the
  whole dense-search rollout loop** — ring-body movement, ordered ambient/corpse/boost food events, death/respawn,
  advisory masks, vector61 features + v8 opponent decisions, scripted anchors, real B8 raster/scalars/forward, six-lane
  dense branching (`README.md`). Mac preflight on a tiny fixture passed (first actions agreed, mass MAE 0). **L4 and
  A100 manifests are prepared but NOT dispatched.** Frozen kill rule: CONTINUE only if GPU cost per dense-search
  decision ≤ 1/10 of the L4 CPU baseline (0.0006 u/decision → ceiling 0.00006) AND first-action agreement ≥ 99% with
  0 unexplained disagreements (`KILL_RULE.md`, `BASELINE.json`, `RESULT.md` = PENDING). Cap 15 u.
- An independent GPU expert's design (`…/2026-10-20_a100/gpu_expert_1.md:11,116,141-153`): thread-per-world
  `numba.cuda`, ~95–120 KB/world, estimated **0.25–0.4M active row-steps/s per L4 (50–100× one lane today)**; ~6,100
  hot-path lines to port; 10–14 Sol days for a full exact port; break-even ≈ 80k teacher worlds — "the enabling
  technology for the *next* campaign". A100 was ruled out for the host-bound teacher (`SYNTHESIS.md`), not for
  device-resident work.

### 2.6 What was tried and killed (argue with these numbers, do not ignore them)
| Lever | Result | Where |
|---|---|---|
| **Exact-sim dense search acting on top of B8** (every 5th decision at L≥600, all legal actions × D300, offline, no deadlines, scripted mix only) | **+92 pooled / +122.8 on s3; 54/64 vs 38/64 survival; only 75 of 301,589 decisions changed; 16,685 s wall, 55M env-frames** | `docs/research/t3_ceiling_test_2026-10-15.md:115-129` |
| Distilling those search labels into B8 (T3 screen; Claim A) | T3 screen −81…−88 mass vs control; Claim A's frozen loss flips 0.4% of decisive rows; T35 learned outcome head AUC 0.747 KILL | `…/2026-10-21_direction/FABLE.md` §1 table; `CAMPAIGN_LOG.md:68,70` |
| Serving the search live (Path B v2, async lookahead at 24 fps on the Mac) | **Smoke 2026-10-21: N=1 threads delivered 7/75 improvements (delay-only), N=6 delivered 0/75, zero horizon rescues → STOP** (M3 serial: 30–63 s per doom vs 12.5–24.9 s of lead) | `research/t34_pathb_smoke/RESULT.md:3-13`; `CAMPAIGN_LOG.md:119` |
| League / self-play fine-tune of B8 (T23: N=3 static pool, 5×8M) | **−80.8 mass vs B8, 0/5 seeds** | `research/t23_r0_screen_20261017/RESULTS_T23.md:8` |
| Bare B8 TD fine-tunes on Colab (S1) | −12.5 / −14.9 pooled mass | `CAMPAIGN_LOG.md:116` |
| Reward levers (M6), Huber (T2), label variants (T4), +24 input features (T5), restart curriculum (T6), LOCAL30 (T11) | all NULL/KILL/FAIL | `redesign_campaign_retrospective_2026-10-17.md` "Closed late-death arms" |
| Earlier vector61 fine-tunes FRP-v4 / v5-H | +6.8 PARTIAL / −41.5 KILL | `docs/STATUS.md:107-110` |
| Two-deviation search (T35) | pooled KILL 2/7, no new rescues | `CAMPAIGN_LOG.md:76` |

Standing verdicts you are allowed to challenge (that is the point of this round): the portfolio judge's "Not funded:
self-play/league/PPO, 4,096-root runs, A100/G4/H100 for host-bound work" (`CAMPAIGN_LOG.md:108`) and the direction
judge's "Do not switch to a new mechanism (self-play, PPO, league, GPU sim). Each is 2+ weeks to first signal"
(`…/2026-10-21_direction/FABLE.md:81`) were made under a **thrift + calendar** frame. The owner has now removed the
thrift constraint. The same judge's honest priors: Claim A beats B8 by mid-Nov ~20%; Claim B given in-time delivery
~25–60% (`FABLE.md:79`) — and the Path B smoke has since failed delivery at both N=1 and N=6.

### 2.7 Compute, transport, builders
- Colab Pro+: **~1,140 units**, owner will spend all within ~a week. Rates (`~/.claude/skills/colab/SKILL.md:58-64`,
  third-party, fluctuate): **L4 ~1.54–2.0 u/h (12 vCPU host, 24 GB)**; **A100-40/80 ~5.4 / 7.5 u/h**; H100 unmeasured
  (scarce; > A100-80); G4 (RTX PRO 6000 Blackwell, 96 GB, needs cu128) ~8.7; T4 ~1.2. **Three account assignments
  max, notebooks included** (`research/colab_cli/QUALIFICATION.md:56-57`); continuous execution up to 24 h. Every
  runtime we have seen has a **12-vCPU (6 physical core) host**, A100 included. Only transport: the qualified CLI
  supervisor `research/colab_cli` (hash-bound uploads, no Drive, exports ≤ 64 MiB measured, receipts, no auto-retry).
  Python 3.12 uv venv, torch cu126 (cu128 on Blackwell).
- Mac↔x86 bitwise identity fails on exact float32 Q ties ~1/50 episodes (`docs/STATUS.md:94`); strict gates therefore
  stay on the Mac. Any Colab-trained artifact is gated on the Mac sim, so label/state-distribution fidelity matters.
- Builders: Sol (Codex gpt-6.1-sol) implements components in hours-to-a-day; the last new trainer (S1) needed 10
  acceptance runs before it was qualified. Claude operates all jobs. Fable judges.

## 3. The open question for Round 1

**Given this environment, this 1.6M-parameter served baseline, the one measured large win (exact-sim search acting:
+122.8 on s3, but undistillable so far and undeliverable at 24 fps), the long list of killed levers, and ~1,140 Colab
units with up to three L4/A100/H100 runtimes for about a week — from your discipline's vantage point, what is the
mechanism that could plausibly turn GPU scale into a served snake that beats bare B8 s3 by ≥ +40 mass at equal or
better survival on the Mac gate, and what is the first ≤ 2-day, ≤ 20-unit experiment that would tell us whether that
mechanism deserves the campaign?**

Answer it as the expert you are, not as a committee. We want the map, not a vote: name the mechanism, the published
systems that make you believe it (paper/system + year), the throughput and sample numbers it needs in *this* game's
units (env-steps, searched decisions, worlds, GPU-hours, Colab units), what must be built (Sol days), what could
legitimately differ from the Mac reference sim and still produce a trustworthy gate pass, and the kill rule. Say
explicitly where you think a prior judgment above is wrong and why. Changing the game's rules or benchmark does not
count as beating B8. Separate known results from conjecture. Keep to the output format in your task prompt.
