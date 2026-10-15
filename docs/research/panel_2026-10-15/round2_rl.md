# Round 2, deep RL memo (role: rl)

Read-only design discussion, 2026-10-14/15. Inputs: my round-1 memo, the mechanics memo
(`round1_mechanics.md`), the Mac-session report, the rendered digests, and the repository at
`/home/user/snake-dqn` (Oct 8 state; the M4-M6 trainer `train_m3c.py` is not here, so every claim about
it is inferred from `train_m3.py` plus the quoted flags). Where I checked a claim of my partner's against
code or digests I say which file and line.

## 0. What I checked and what it settled

| Claim (whose) | Check | Result |
|---|---|---|
| Hero death is terminal in the TD target (mine, implicit in diagnosis 8's "respawn") | `train_m3.py:529-531`: `if done[e]: flush(e, -1, ...)` writes `disc = 0.0` for every pending n-step item | Death IS terminal. Respawn only changes the data mix, not the value. My "respawn leaks value" point is wrong. |
| gamma 0.998 under-weights late deaths ~4x vs the H5000 integral (mine) vs "growth is compressed the same way; the ratio is preserved" (partner) | Mass reward = kappa(1-gamma)Phi(L) per step (digest block 9: "B/L add kappa(1-gamma)Phi(L), kappa=1"); death terminal | Partner is right. Death costs Phi(L) and a pellet is worth (1-gamma)/10 per step = 0.1 in value; both scale with the same effective horizon, so the growth-vs-death ratio (= L) equals the eval integral's. Lever 6 (gamma 0.999 / terminal hero) is withdrawn. |
| TD loss is `smooth_l1_loss(qa, y)` at raw Q scale, beta 1 (mine) | `train_m3.py:430`; qmax 120-153 in the M6 smokes (digest block 2); death delta = Phi(L) ~100 | Confirmed for the repo trainer; for `train_m3c.py` it is the quoted "Huber" in the M3-C summary. Still the cheapest untested lever. |
| n-step 5, target sync every 1000 updates, uniform replay sampling, eps random actions masked only by the one-step resolved mask (mine) | `train_m3.py:454, 446, 155-156, 497-500` | Confirmed (repo). `--eps-end-at 1` in M6 means eps = 0.01 from the first transition. |
| Anchor lambda = 1.0 for all 8M of an M6 fine-tune unless overridden (mine) | `Schedules.anchor_hold = 8_000_000` (`train_m3.py:75`); M6 argv (digest line 921) has `--anchor-floor 0.1` and no `--anchor-hold` | Inferred-likely. Needs one `grep` on the Mac. |
| Guard is alive-frame `surv5`, seed 0's bank only, threshold -0.05, tolerance 1e-9 (partner) | `research/sequential_phase_r/example_rules.py:14, 85-91, 122`; `hooks.py:140-147` reads `survival_fraction` from the H5000 prefix block; `metrics.py:133` = alive_frames / scored_horizon | Confirmed. The census's 0.531/0.656/0.688 are k/32 episode counts (digest line 902: deaths 11/20/12/15/17 of 32). The report's "0.64 target" mixes the two metrics. |
| greedy_food treats other snakes as static within the frame; free-space veto only when an alternative exists, need = min(L, cap) with cap = min(160, max(32, 2L)); never boosts (partner) | `src/game/ai_snake.py:60-120` (`simulate_relative_action_fatality`: "every other snake are static within a frame"); `scripted_snake.py:119-147, 221` | Confirmed. |
| Mixed = 3 frozen-pool + 2 random_safe; random_safe is deterministic given (world seed, slot, frame) (partner) | `tournament_eval.py:17, 341-354`; `fast_anchors.py:111-115` | Confirmed. Note: frozen-pool opponents are argmax networks, so counterfactual rollouts are exact in ALL three mixes, not only scripted. Only humans break determinism. |
| Respawn within ~2 frames at a uniform empty cell (partner) | `batch_sim.py:1257, 1259-1293` (`respawn_timer = frame_rate = 1`) | Confirmed. |
| Local window 23 ahead / 7 behind / 15 lateral, own TTL clamped at 255, global 8x8 counts; the FC layers are global over the window (partner) | `ego_raster.py:26-57, 121-124, 249`; `ego2s_network.py:36-58` (3 convs, RF 9, then `Linear(64*8*8, 256)`) | Confirmed. My round-1 "the RF framing is wrong" was a quibble: the limit is the window, the TTL clamp and the 8x8 counts, exactly as the partner states. |
| Boost: second cell per frame if L >= 5; one cell burned per 3 boosted frames (partner) | `batch_sim.py:98-99, 507, 736-737` | Confirmed. |
| "41 of 64 trap onsets unavoidable" (partner cites the literature track) | Digest line 1154 lists it among the literature track's codebase findings; definition not recorded | Unverified. If true it means the commitment decision is typically BEFORE the trap flag, which matters for both of us (section 2). |
| "The pocket after sealing is 30-60 cells" (partner) | Not in the digests | An inference from "dies in ~31 frames". The census captures hold the region size at onset; read it. |

## 1. Agreements (real ones)

1. **The first thing to run is the exact-replay commitment/contrast study on the 75 census deaths, with no
   training before it.** Scripted worlds are deterministic given the hero's actions (and so are frozen-pool
   worlds); at every decision in the last ~300 before death each legal action can be rolled out exactly
   with B8 greedy for 300 frames. That single study answers the partner's questions (who sealed, when was
   the last avoidable decision, did a boost or tail-side move survive, do tail-in-region/boundary-TTL
   separate fatal from surviving actions) and mine (what was B8's Q-gap between the chosen and the best
   surviving action at that decision; can a linear probe on B8's own 256-d fuse features tell the fatal
   from the safe action). It subsumes the "ceiling test" I proposed.
2. **The trap flag is the seal, not the commitment.** The census's "trap formed 31 decisions before death"
   is the frame the one-step tail-aware region (`ego_raster_b.py:34-35`, TTL <= 1 counts as free) drops
   below L. Partition of a 12,035-cell arena by a 1,000-cell body is routine; the flag fires when the
   neck closes. So "31 decisions" is a lower bound on the credit horizon.
3. **champion_a5 + v8's "plateau" is growth rate x horizon, and its 0 deaths at L >= 800 are
   uninformative** (~7,000 decisions at B8's 0.177/1k predicts ~1.2 deaths; P(0) ~ 0.3). Imitating it
   means staying small, which the mass objective does not want.
4. **Under the H5000 integral B8's growth is roughly break-even against its own late hazard**, so a
   growth governor is a pure mass loss that only serves the guard; it is legitimate only as an objective
   term and only if hazard rises with L within frame buckets.
5. **No serving-time planner or hand rule; training-time labels only; serving stays a bare network.**
6. **Reward-shaping proxies (tight cost, corpse-bait discount) are low priority.** My reason: they sit in
   Huber's quadratic regime so they do move behaviour (TIGHT moved tight_frac 2.8% -> 2.1%), but they
   shape a proxy. Partner's reason: they give the learner no new way to tell the fatal action from the
   safe one. Both reasons point the same way.
7. **Any screen is 5 paired seeds x >= 96 scripted worlds, reporting episode-level AND alive-frame
   survival on the same bank**, plus mass vs the incumbent. The M6 32-world probes (SE ~0.056 on a
   difference) had 63% power for the +0.11 they sought.
8. **The guard as run is weak in both directions and functions as a growth cap**; any fix is prospective,
   on fresh banks, and B8's look-2 data is never re-scored.
9. **Mechanism**: late-game neck sealing in crowded scripted worlds, with the kill pump (greedy snakes
   die on the hero's head path ~20x per episode and drop whole-body corpse lines beside it), pocket
   intrusion and respawn pollution explaining why scripted (0.53) < frozen (0.65) < mixed (0.87) for B8.
   I accept this as the leading mechanistic hypothesis; it is measurable from the census captures.

## 2. Disagreements, each with a resolution

**D1. Why M6 (TIGHT, DEATHW) and D20 were null.** Partner: harsher death penalties "give the learner no
new way to tell the fatal action from the safe one". Me: that is true but not the operative reason. Under
`smooth_l1` beta 1 at Q scale ~100, a death row's gradient is clipped to the same magnitude as a residual
of 1; for state classes the network cannot separate (which is exactly the partner's aliasing), the fixed
point is the median target, i.e. the survive value whenever P(death | class) < 0.5. Doubling the death
penalty changes a target whose gradient is already saturated. And M6 had 63% power for +0.11. So M6 does
not discriminate perception from value learning. *Resolution*: T2 below (loss-scale arm, zero code)
plus the Q-gap-at-commitment readout from T1. If the Q-gap at the last avoidable decision is already
large in favour of the fatal action AND the probe cannot separate, perception wins; if the Q-gap is
small (< 1-2 units) or the probe separates, value learning wins.

**D2. "Fix training at L >= 800: the regime is the late scripted world, stratifying by L under-samples
it."** Partially agree. Training worlds are 75% scripted (`--mixed-frac 0.25`), so late scripted states
are ~0.75 x 0.42 ~ 30% of transitions time-averaged: not starved in aggregate. The defect I see is cyclic:
all 128 worlds per actor reset together every 5,000 frames (`train_m3.py:259-300, 455-462`) and the
500k FIFO holds ~1,950 frames, so replay swings between ~0% and ~80% late-game with period 1.28M. If
reservoirs are built they should be keyed on (mix, frame-in-segment, L), not L alone. *Resolution*: one
training-telemetry histogram over a full 1.28M cycle (deaths and transitions by mix x segment-frame x L
x eps-flag) and the checkpoint-phase probe (8 checkpoints over one cycle, 96 worlds each).

**D3. Tail-topology inputs vs the M7 feature study.** Partner: AUC over all L >= 800 states is the wrong
test; a policy input pays at the commitment decision. Me: agreed on the statistics (a feature that fires
only at the ~1 commitment row per death cannot move an AUC computed over ~100 positive rows per death),
but M7's family A ("time-aware tail reachability T in {L/4, L/2, L}") and family B (frontier vacate
times) are near relatives of tail-in-region and boundary TTL and added only +0.019 / +0.018 AUC, and
M7's not-tight AUC at H100 is 0.587 -> 0.607, i.e. pre-seal death is nearly unpredictable from any
scalar or hand feature so far. Either the signal is in the raster (then B8's own features carry it and
the probe will show it) or it is genuinely absent from the inputs. A 5-seed 20M retrain (~7 h Mac plus
featurizer and parity work) should wait for the action-contrastive test. *Resolution*: at the last
avoidable decision from T1, does the surviving action differ from the fatal one on tail-in-region or
boundary TTL in >= 70% of cases (partner's own GO bar)? And does a logistic probe on B8's fuse features
separate commitment states from matched controls (AUC >= 0.75)?

**D4. Rollout labels "as the anchor stream in place of v8 labels".** Partner proposes feeding rollout
labels through the margin loss. Me: the current on-policy v8 labels at `--label-frac 1.0` equal the
student's own argmax on 99.995% of decisions, so the margin loss is a keep-your-argmax trust region on
the very states that must change; dense rollout labels would inherit that (and the S18 margin-vs-TD
conflict). Labels must be sparse and decisive (chosen action dies within 300 frames, some alternative
survives), weighted, with the lambda schedule audited. *Resolution*: audit the lambda M6 actually ran
(one grep); add an anchor-off arm to T2; in T3 label only decisive states.

**D5. Counterfactuals are exact only against scripted opponents.** Partner says "the same labels cannot
be produced against human players". True for Play mode, but frozen-pool opponents are argmax networks
and random_safe is seeded by (world, slot, frame), so rollouts are exact in all three eval mixes. T3's
label bank can and should include frozen and mixed worlds, which also guards the transfer risk the
partner raised. *Resolution*: code read of `fast_anchors.py` and the frozen policy path (done; both
deterministic); confirm `GridBatchSim` deepcopy fidelity with a 1,000-frame replay parity check.

**D6. Growth governor as a reward cap.** A cap on the mass credit above L_cap leaves eating free, so it
does not govern; the agent is indifferent, not deterred. A governor needs a negative term above L_cap
(or a survival-weighted objective). Minor, and both of us rank it last among legitimate levers.
*Resolution*: design note; only matters if the owner keeps a champion-level survival floor.

**D7. The 30-60-cell pocket.** Unmeasured. If the region at the trap flag is typically several hundred
cells and the hero dies by further self-partitioning, the "neck" story is half the mechanism and the
in-pocket play is the other half (where boost burn and tail-chasing matter). *Resolution*: region size
at onset and its time series from the census captures.

## 3. What changed my mind

1. **Horizon lever withdrawn.** Death is terminal in the TD target and both growth value and death cost
   scale with the effective horizon, so gamma does not change the growth-vs-death ratio. The M4
   gamma 0.995 -> 0.998 gain is confounded with the other three arm-C changes.
2. **The guard gap is smaller in clause units than I said.** My "census corroborates -0.10 to -0.15" was
   in episode units; the clause is alive-frame. A late death at frame ~3,500 costs ~0.3 alive-frame, an
   early one ~0.7. Rough reconstruction from the census: B8 loses ~(26 x 0.7 + 49 x 0.3)/160 ~ 0.21
   alive-frame per episode, the champion ~(10 x 0.5)/32 ~ 0.16: a gap of about -0.05, i.e. at the margin,
   with the look-2 -0.156 on seed 0's 32 worlds about 1.5-2 SE below it. The per-world `look-2.json`
   records settle this at zero cost and should be read before anything else.
3. **Credit horizon.** The trap flag is the seal. If "41 of 64 onsets unavoidable" holds, the commitment
   is earlier than 31 decisions for most deaths, through states whose region scalars read safe. n-step
   16-32 is demoted from lever 2 to a conditional arm (only if the commitment horizon is <= ~40); rollout
   labels (which supply the contrast at the commitment decision without bootstrapping) move up.
4. **Perception moves from "last" to "gated third".** M7's not-tight AUC of ~0.59 at H100 says pre-seal
   death is nearly invisible to every scalar and hand feature tried; the window/TTL-clamp/8x8 limits are
   real. What is still unproven is whether B8's raster towers already carry the signal; the probe decides.
5. **Counterfactuals are cheaper and more general than I assumed** (exact in all mixes), which makes the
   commitment/contrast study a half-day Mac job rather than a research project.

What I still hold: the Huber median regime is the reason death-penalty levers cannot work and is the
cheapest untested lever; M6 was underpowered; the on-policy margin anchor is a trust region; replay
content cycles with period 1.28M; `death_len_mean` is phase-confounded.

## 4. Updated lever ranking (RL view, integrating the mechanics memo)

1. **Commitment/contrast study + metric reconciliation (no training).** Decides between value learning
   and perception, sizes rollout labels, boost and tail-topology levers, and fixes what the guard gap is.
2. **Loss scale at the death delta** (reward x0.01 with an exact output-layer rescale, or Huber beta ~
   death delta, or MSE), 5 paired seeds x 8M x 96 worlds, ideally as a 2x2 with anchor off. Zero code.
3. **Rollout-labelled late-game training** (sparse decisive labels via the LabelBuffer path; snapshot bank
   of late states across all mixes), gated by lever 1 showing surviving alternatives within 150 decisions
   for >= 60% of deaths. Needs the owner's "training-time labels are not a veto" decision.
4. **Tail-topology inputs** (per-action tail-in-region, boundary TTL, free run, global TTL plane), gated by
   the action-contrastive test in lever 1.
5. **Anchor audit/ablation** (folded into lever 2 as an arm).
6. **(Mix, phase, L)-keyed replay reservoirs / segment desync**, after the checkpoint-phase probe.
7. **Calibrated death-probability head** (the probe in lever 1 is its zero-training version).
Conditional/withdrawn: n-step 16-32 (only if commitment horizon <= ~40); boost competence (falls out of
lever 3's labels if lever 1 shows >= 20% boost-saveable deaths); corpse-bait discipline (only if >= 50%
of traps began while eating corpse within 2 cells of own body); growth governor (only if the owner keeps
the champion floor and hazard rises with L within frame buckets); gamma 0.999 / terminal hero (withdrawn).

## 5. Joint top-3 (pre-registerable)

### T1. Commitment and contrast study on the census deaths (no training)
- **Data**: the 75 B8 deaths (5 finals x 32 round-52 scripted worlds, census-b8 code), plus the 11
  incumbent and 10 champion deaths as reference; eps = 0 throughout (eval policy).
- **Procedure**: replay each world deterministically to the death; for each decision d in the last 300
  frames, deepcopy the sim, play each legal action (incl. boost) then B8 greedy for 300 frames, record
  survival. Define LAD = last decision with an alternative surviving >= 300 frames; H_LAD = frames from
  LAD to death. Record at LAD and at the trap-flag frame: Q(chosen) - Q(best surviving alt); sealer
  identity of the neck cell (own body / opponent body / wall); tail-in-region and min boundary TTL for
  chosen vs surviving action; whether a surviving action is a boost; live opponent head inside the
  region; region size at onset and its series; frame of death. Fit a logistic probe (5-fold by world) on
  B8's 256-d fuse features: LAD states vs matched non-LAD states at similar L and frame.
- **Zero-compute part**: from Phase R `look-2.json`, per-world `surv5` and `mi5` for B8_s0 and champion
  on seed 0's scripted bank, and the incumbent on all banks; from the census, frame of death per death.
- **Decision rules**: (a) >= 60% of deaths have a surviving alternative within 150 decisions and median
  H_LAD <= 100 -> T3 GO; median H_LAD > 300 or < 30% avoidable within 300 -> T3 NO_GO (planner-as-
  teacher becomes long-rollout and expensive). (b) Q-gap at LAD < 2 Q units in a majority, or probe AUC
  >= 0.75 -> value-learning limited, T2 is favoured; Q-gap > 5 in favour of the fatal action AND probe
  AUC < 0.65 -> perception limited, tail-topology inputs move to the top. (c) Surviving action differs on
  tail-in-region/boundary TTL in >= 70% -> tail-topology GO; < 50% -> NO_GO. (d) Boost saves >= 20% ->
  include boost labels. (e) Guard: if the reconstructed alive-frame gap vs champion is within [-0.08,
  -0.02], the guard failure is reported as "at the margin, one bank" in any amendment memo.
- **Cost**: ~1 Mac day, no pods. Rollouts ~75 x 300 x 4 x 300 = 27M env-frames at ~0.4 ms per env-frame
  batched (feasibility doc figure; verify deepcopy fidelity with a 1,000-frame parity replay first).
- **Owner decision**: none.

### T2. Loss-scale 2x2 screen (training, zero code)
- **Arms**: CTRL (M6 CTRL recipe: 8M fine-tune from B8_s0..4 finals, `--gamma-start 0.998 --eps-end-at
  1`, anchor v8margin as run) x {Huber at Q scale as now, death at the knee: scale rewards by 0.01 and
  the output layer of the init checkpoint by 0.01 exactly, rescale the margin constant, re-base the qmax
  tripwire; fallback beta 30-50 on the old scale if unstable} x {anchor as run, `--anchor-floor 0
  --label-frac 0` from step 0}. Reuse the existing M6 CTRL_s10..14 finals for the CTRL cell (re-probe on
  the new 96-world bank), so 3 new arms x 5 seeds x 8M = 120M transitions ~ 6 h at 5.4k/s on the Mac.
- **Metric**: scripted episode survival and alive-frame survival on 96 fresh scripted worlds per seed
  (480 paired per arm; SE(diff) ~ 0.032 episode-level); mass on 32 worlds x 3 mixes vs frp3-s12 + v8;
  hold-out v8 agreement as the collapse tripwire; deaths per 1k decisions at L >= 800.
- **Stopping rule**: GO if the paired mean scripted-survival delta vs CTRL >= +0.06 with mass delta >=
  -30 and no other mix down > 0.05; KILL if delta <= 0 at 96 worlds; otherwise extend to 192 worlds once
  and stop. Interaction (loss x anchor) reported with its SE (~0.045); adopt the best single arm only if
  its own cell passes.
- **Value check**: rerun T1's Q-gap readout on the new checkpoints; the gap must open >= 10 decisions
  before death in a majority of the L >= 800 deaths for the mechanism to be credited.
- **Owner decision**: none for Mac-only; ~$3-5 of pod time if parallelised (4090 pinned, no fallbacks).

### T3. Rollout-labelled late-game training (training-time planner labels), gated by T1(a)
- **Design**: build a snapshot bank of late states (frame >= 2,500, L >= 700) from B8 rollouts in all
  three mixes (75% scripted / 25% mixed as in training, plus frozen); at a sparse subset of decisions
  (region drop, tight flag, or every 50th decision at L >= 700) roll out each legal action with B8
  greedy for 300 frames; keep only decisive states (chosen action dies within 300, some alternative
  survives); label = surviving action set. Fine-tune from the B8 finals for 8M with the TD loss at the
  T2-chosen scale and the margin loss on decisive labels only (lambda 1.0 for 2M then floor 0.1), v8
  labels off. Optionally regress Q(chosen) toward the realised 300-frame rollout return as a second arm.
- **Arms/seeds/worlds**: {CTRL at the T2 scale, LABELS, LABELS+return-regression} x 5 paired seeds; 96
  fresh scripted worlds per seed plus 32 frozen and 32 mixed for transfer.
- **Stopping rule**: GO if scripted survival delta vs CTRL >= +0.06 at mass >= -30 and no mix falls
  > 0.05; KILL if <= 0; else extend to 192 once. Report the label density, the fraction of labels equal
  to the student's argmax, and the S18-type conflict (hold-out agreement, margin loss trend).
- **Cost**: 2-3 days of code (sim clone, batched rollouts, labeler, LabelBuffer swap) + ~4 h Mac
  training + ~3-6 h rollouts; pods optional.
- **Owner decision**: required. Does a training-time rollout planner (labels only; serving stays a bare
  network) count as "no veto"? Both panelists recommend yes, with disclosure in the result doc.

## 6. Minimum context before the first experiment (deduplicated, prioritized)

MUST
1. `train_m3c.py` on branch m6: TD loss and beta; n in the n-step builder; replay capacity/sampling;
   whether all 128 worlds per actor reset together and whether actors are staggered; the exact mass
   reward (kappa(1-gamma)Phi(L) with or without the shaping term; death value); the anchor lambda the M6
   fine-tunes actually ran (defaults hold 1.0 to 8M); LabelBuffer sampling; the 30 bv8 feature names and
   caps (any tail-reachability or boundary-TTL signal?). [mac_artifacts]
2. Phase R `look-2.json` per-world `surv5`/`mi5` for B8_s0 and champion_a5+v8 on seed 0's scripted bank;
   incumbent per bank; frame of death per lost world. Settles the metric question and the true guard gap.
   [mac_artifacts]
3. Census per-decision captures: hazard per 1k decisions by L bucket x frame bucket for B8, incumbent,
   champion; region size at trap onset; frame of death for the 75; boost involvement. [mac_artifacts]
4. T1 itself (commitment/contrast rollouts, sealer identity, Q-gap, probe). [new_measurement]
SHOULD
5. CTRL_s10 `log.jsonl`/telemetry over 8M (qmax sawtooth, deaths_per_1k, death_len_mean per interval) and,
   if logged, training deaths by mix x segment-frame x L x eps-flag over one 1.28M cycle. [mac_artifacts]
6. Checkpoint-phase probe: 8 checkpoints over one 1.28M cycle of a B8_s0 fine-tune, 96 scripted worlds
   each. Decides reservoirs-only vs reservoirs + desync. [new_measurement]
7. champion_a5+v8 and incumbent episode and alive-frame survival on all five Phase R banks and the
   round-52 bank (96 worlds): the comparator estimate for any amendment and the sensitivity table.
   [new_measurement, cheap]
8. Seed-ensemble eval (mean-Q of the 5 B8 finals, 96 scripted worlds, ~1 h): near-independent seed
   deaths predict a large gain if errors are idiosyncratic. [new_measurement]
9. Whether n-step, replay capacity and the label source are flags in `train_m3c.py`; Mac RAM and
   `--obs-pack` bytes per transition. [mac_artifacts / mac_transcript_digests]
NICE
10. L60 intermediate probes (every 5M) with their checkpoint phases; D20 probe files. [mac_artifacts]
11. Body-shape and corpse-intake statistics at L >= 800 (parallel-lane fraction, corpse share of intake,
    head-to-own-body distance while eating corpse); fold into T1's capture if cheap. [new_measurement]

## 7. Joint view on the guard clause

As run (`example_rules.py`): one seed, one bank, 32 vs 32 paired worlds, alive-frame `surv5`, point
threshold -0.05 with tolerance 1e-9, comparator champion_a5 + v8. It is weak both ways (paired SE ~0.07
on the alive-frame metric, ~0.125 on episode-level; ~34% false-fail under equal safety on the latter,
~61% detection of a true -0.10) and, because the comparator never lives at L >= 800, it measures "did you
stay small" rather than "are you safe for your size". The report's "0.64 target" subtracts the margin
from an episode-level census number while the clause is alive-frame; late deaths cost ~0.3 alive-frame
each, so the census implies a true clause-metric gap vs the champion of roughly -0.03 to -0.08, i.e. at
the margin, with look 2's -0.156 about 1.5-2 SE below that on one bank. Read the per-world records.

Joint recommendation: (i) never re-score B8's look-2 data; (ii) if the owner's goal is "a bare network
that beats frp3-s12 + v8", amend prospectively and once: comparator = the served agent (goal-justified),
margin unchanged at -0.05, pooled over all five seeds' fresh banks with >= 96 scripted worlds per seed
(or the feasibility clause's cap), both metrics reported with the clause scored on alive-frame as before,
plus an absolute catastrophic floor (pooled episode-level survival >= 0.50 on every mix) and a per-seed
floor (no seed below -0.15 vs the served agent); publish the sensitivity table and disclose PARTIAL;
(iii) report length-matched hazard per decision as a secondary, not a clause (it is diagnostic gold but
needs per-decision captures to pre-register cleanly); (iv) if the owner's product goal is "rarely
visibly dies in Watch mode", say so explicitly, keep a champion-level floor re-registered on 96 worlds
and all seeds, accept that B8 probably fails it, and continue the campaign with T1-T3, whose aim is then
a learned long-snake skill, not a smaller snake.
