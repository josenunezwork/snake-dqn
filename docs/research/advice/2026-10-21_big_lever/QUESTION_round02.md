# Big-lever research rounds — Round 2 question (divergent, now focused: why would scale work here?)

Judge: Fable (round-1 judge wrote this, 2026-10-21 ~00:20 PT). Charter: `../CHARTER.md` (read it first), then
`../round-01/JUDGMENT.md` (the Round 1 ruling and shortlist), then `../LIVE_FACTS.md` if present (measured results Claude
adds between rounds; authoritative). Repo: `/Users/josenunez/Projects/ml/snake-dqn-biglever-ro` (detached read-only
copy of `main`; start with `AGENTS.md`). Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`.
Round 1's `QUESTION.md` §2 (verified facts about the target, network, environment, simulators, GPU work, killed levers,
compute) still holds, with the corrections below. Do not re-derive it; cite it.

## 1. Goal (unchanged)

One big-lever push, grounded in research that works in slither.io-like environments, using GPUs/CUDA on Colab at scale
(~1,140 units, spend within a week, speed over thrift), producing a **served snake that beats bare B8 s3 by ≥ +40
pooled H5000 mass at equal-or-better survival on the Mac strict gate**, then serving qualification and owner OK.
Rounds 1–3 map the space; 4–7 attack feasibility/cost/parity/eval validity; 8–10 converge on a staged plan with kill
rules. The T34 campaign (defensive search teacher; Round 0 verdict due Wed) continues in parallel and is not yours.

## 2. What Round 1 established (see `../round-01/JUDGMENT.md` for the per-advisor record)

- **Convergence, unexpectedly early.** 5 of 10 advisors independently named the same mechanism: long-horizon on-policy
  RL from a B8 warm start against the fixed served opponents, with an undiscounted H5000 mass-integral return, at
  10^8–10^9 hero transitions, optionally recurrent, served as a small bare network. Shortlist #1 (judge's confidence
  it becomes the push 45 %, beats B8 at bar 25 %).
- **The enabler is not ready.** The GPU spike (`research/gpu_sim_spike/`) as written runs an H×W = 12,035-iteration
  flood per observation (`ego_device.py:142`) and up to 4,096 in `v8_device.py:144`; its L4/A100 manifests would time
  out. Bounded kernels come first (E1). Admission floor: ≥ 20k full six-snake world-steps/s on L4 with five real
  opponents and long bodies. Shortlist #2.
- **Representation is untested, not refuted.** B8's global raster is 8×8-cell mass blocks with two convolutions;
  local TTL clips at 255 (`src/simd_env/ego_raster.py:25-57`). No recurrent or full-resolution ego2s student has ever
  been trained (no GRU/LSTM anywhere in `src/` or `research/`; the June DRQN was on the vector61 line). T10 explicitly
  disclaims a ceiling for other inputs. Shortlist #3, with an offline information-value test (E3).
- **Search-as-labels is behind a wall.** Claim A D0 printed KILL (frozen loss cannot flip decisive rows); Path B live
  lookahead is undeliverable at 24 fps (0/75 at N=6). Any search contribution must enter *inside* a training loop
  (auxiliary/policy-improvement targets), not as another offline fit. Shortlist #4, deferred until the D0 rerun and
  Round 0.
- **Corrections to the Round 1 fact sheet.** The 150-pellet figure is a board-wide cap on corpse-class pellets (kill
  corpses + boost-trail pellets), oldest evicted (`src/core/mechanics_constants.py:30-64`), not 150 per corpse. The
  Colab 3-concurrent-assignment cap is a self-imposed assumption (test pending). A100 runtimes have the same 12-vCPU
  host as L4.
- **Unanimous rulings.** Defense (late large-body survival) is the lever, harvesting additive, hunting capped. Training
  may use different RNG streams / float order if declared and mechanics are unchanged; evaluation may not differ from
  the Mac reference. Leagues (T23 −80.8) and world models are not funded as levers this week.

## 3. The negative record the mechanism must explain (argue with these numbers)

Every training-based lever since B8 has lost mass against it: T23 league fine-tune **−80.8** (0/5 seeds; scripted
−101; survival unchanged or better — "traded mass for caution"); S1 mature bare fine-tunes **−12.5 / −14.9** with γ=.998,
ε=.01, anchor .1 fixed from decision zero; T4c tightened labels **−9.9 scripted mass, −0.0135 survival**; T6 restart
curriculum KILL; T4 RETURN arms hit **34 % boost** tripwires; M6 reward levers NULL. All were replay DQN at 8–20M
transitions with v8 anchors. B8 itself was trained on the Mac for 20M transitions × 5 seeds. The only large positive
effect is exact-sim search acting: **+122.8 on s3 by changing 75 of 301,589 decisions** (0.025 %), 54/64 vs 38/64
survival, scripted mix only, 32 worlds. The fatal path is a self-bounded pocket built over 50–100 decisions; horizon
AUC ≈ 0.92 at 30 decisions, ≈ 0.75 at 300 on every tested input.

Experiments Claude may have dispatched since Round 1 (results, if any, are in `../LIVE_FACTS.md`): E1 flood-kernel fix
+ frozen GPU spike on L4; E2 two-arm PPO pilot (recurrent vs feedforward from the B8 trunk, γ=1 mass-integral return,
8–20M each, fresh 96/32/32 screen vs B8); E3 offline information-value test (B8 inputs vs + full-res geometry vs +
history, D300 mass-return regret on T3-bank roots); E4 collector-throughput ruler on L4; E5 assignment-cap test.
Treat any reported number as authoritative over your priors.

## 4. The open question for Round 2

**Why would a B8-warm-started learner at 10^8–10^9 transitions gain mass where every 10^7-transition fine-tune lost
it — that is, what exactly must the learner be (objective and credit assignment, exploration and restart
distribution, trust region against forgetting, memory and input representation, and the role of exact-sim
counterfactuals inside the loop) so that GPU scale buys *earlier* enclosure avoidance instead of the caution-for-mass
trade that killed T23, S1 and T4c — and which single observable in the running pilots would tell us, before the week
is spent, that the answer is "wrong mechanism" rather than "needs more scale"?**

Answer from your discipline, not as a committee. Name the published systems you rely on (paper/system + year), give
numbers in this game's units (hero transitions, worlds, world-steps/s, L4/A100 hours, Colab units, Sol days), state
what may differ from the Mac reference sim in training and what may not, and give a kill rule for your claim. Say
explicitly where Round 1's judgment or shortlist is wrong and why. The red-team persona should argue the opposite
case as strongly as the evidence allows. Changing the game's rules or the benchmark is not beating B8. Separate known
results from conjecture. Keep to the output format in your task prompt.
