# Round 2, mechanics and game-theory panelist (role: mechanics)

Read-only design discussion, 2026-10-14/15. Inputs: my round-1 memo, the RL panelist's round-1 memo,
the Mac-session report, digest blocks 0-9, and the Oct-8 repository. Code citations are from
`/home/user/snake-dqn`; everything about `train_m3c.py` (Mac only) is inferred from `train_m3.py`
and the flags quoted in the digests, and is marked as such.

## 0. Where I moved

Three things in the RL memo changed my diagnosis, and two digest facts I had not weighed did too.

1. **The Huber-median mechanism is real and I had missed it.** `train_m3.py:430`
   `smooth_l1_loss(qa, y)` with the default `beta=1`; under the B8 mass reward a death row's TD error
   is ~100 Q units (V(alive at L=1000) = (1-gamma)Phi/(1-gamma) = Phi = 100; death target -3), so the
   death row and the five n-step rows before it carry a sign gradient only. For any class of states
   the network cannot tell apart, the fit converges to the median target, i.e. the survive value
   whenever the class's death rate is under 50%. A 20% risk is worth -20 in expectation and -0.25 at
   the Huber fixed point. This bounds what *any* input change can do: a new neck feature that lifts a
   class's death rate from 5% to 30% still moves Q by under half a unit. It also explains D20/DEATHW
   (bigger numbers on a saturated gradient) better than my "no new information" story alone.
2. **The guard's SE is ~0.12, not ~0.07.** I assumed world-level pairing bought a paired SD of ~0.41.
   The census says otherwise: B8 died where the champion survived in 51 of 160 worlds and survived
   where it died in 26; 77/160 discordant pairs is exactly what independence predicts
   (0.53x0.31 + 0.47x0.69 = 0.49). World-level correlation between two agents is ~0, so 32 vs 32 has
   SE ~0.12 whether or not the worlds are shared. The RL memo's 34% false-fail under equality stands.
3. **My tail-topology lever is mostly already in the inputs, and the AUC study says it does not
   help much.** v7's `landing_area` uses `tail_aware_reachable` with release (`safety_veto_v3.py:158`:
   a cell becomes passable once the BFS distance reaches its segment's vacate time), i.e. a
   time-aware flood; those per-action areas are among the 30 bv8 scalars B8 already sees (digest:
   "missing vs v8: post-boost landing area, per-action tail-aware landing_area, head risk" is
   exactly what bv8 added). The M7 families A (time-aware flood at T = L/4, L/2, L) and B (frontier
   vacate times) are my "tail-in-region" and "boundary TTL" in all but name; they added +0.019 and
   +0.018 AUC. I keep the objection that death-within-H AUC is not per-action separation at the
   commitment decision, but I no longer rank new static inputs high.
4. **Before the tight flag, nothing static predicts the death.** M7's not-tight rows: AUC at H=100
   0.587 -> 0.607 with 99 extra features; at H=300 0.457 -> 0.524. The states that will be traps in
   30-100 decisions look like the states that will not, on everything measured. With deterministic
   opponents and a near-deterministic hero, death within 100 is a deterministic function of the
   full state, so the information is there but it is "what happens if I keep going this way for 30
   steps", not a summary statistic. That favours exact-replay counterfactuals and rollout labels
   over any new scalar, and it says the Huber fix will mostly help in the separable middle zone.
5. **Traps are escapable 30% of the time** (census: 32 of 108 traps escaped, 74 fatal). Inside a
   trap P(death) ~0.69 > 0.5, so the median fit already values the inside of a trap correctly; the
   deficit is before the flag, where classes are barely separable, which is the zone the Huber fix
   addresses only partially (it penalises the whole class, a caution tax, rather than ranking the
   actions). I therefore expect the Huber lever to be positive but smaller than "the largest single
   change", and worth running first because it costs nothing.

What I did not move on: the mechanism of the deaths (neck sealing in a crowded late scripted world,
sealer unidentified), the scripted-mix explanation (greedy hugging, kill pump, pocket intrusion,
respawn pollution), the champion plateau being growth rate x horizon, and the H5000 arithmetic.

## 1. Agreements (real ones)

- The fatal choice precedes the tight flag; the census labels the fatal frame, not the decision.
  Both memos want the exact-replay counterfactual study on the 75 deaths before any training.
- Scripted worlds are exact under replay (`fast_anchors.py:111` keys `random_safe` by
  `(world_seed, slot, frame)`; greedy_food is a deterministic kernel), so counterfactuals and
  rollout labels are cheap and faithful here, and useless against humans.
- "Tried and null" overstates. M6 at 5x32 had SE 0.056 (63% power for +0.11); mechanistically, a
  bigger death number on a sign-saturated loss and a tightness cost on a one-step symptom were not
  expected to work.
- The v8-label anchor at `--onpolicy-labels --label-frac 1.0` is a keep-your-argmax trust region
  (v8 differs from B8 on 4.5e-5 of decisions, never at a fatal one). M6 argv shows no
  `--anchor-hold/--anchor-floor-at`, so `Schedules` defaults (hold 8M) imply lambda 1.0 for all 8M
  unless train_m3c offsets the clock. Audit before interpreting any more fine-tune screens.
- Replay: the problem is cyclic non-stationarity (all 128 worlds reset per 5000-frame segment,
  500k FIFO < 1.28M segment pair) plus rare-event supply, not aggregate starvation; `death_len_mean`
  is phase-confounded. Mechanics addition: training heroes respawn (`allow_respawn=True`), so
  training also holds "late crowded world, short hero" states that eval never has, and loses the
  rest of a world-segment's long-snake exposure at every death.
- Perception is not the first lever; the FC layer is global over the 31x31 crop; the limits are the
  window and the TTL clamp, and the AUC study says even those matter less than I argued.
- A serving-time governor or any deeper hand rule is a veto in spirit; the objective-level form must
  be screened with its mass cost reported.
- Screen design: pair by seed (fine-tunes from B8_s0..s4), 96 scripted worlds per seed, deaths per
  1,000 decisions at L >= 800 as the secondary endpoint, GO at +0.06 survival with mass >= -30.
- The guard is the weakest clause in the rule and a prospective, disclosed, one-time amendment with
  fresh banks and a pooled estimator is legitimate; re-scoring look 2 is not.

## 2. Disagreements, each with a way to settle it

1. **"The training objective is ~4x more reckless than the metric" (RL item (i), lever 6).**
   Under the mass reward V(alive at L) = Phi(L) regardless of gamma, so death costs Phi(L)+3 and one
   cell is worth 0.1: death/cell = L+30. Under the H5000 integral death costs L*tau/5000 and a cell
   tau/5000: death/cell = L. The trade-off at the moment of choice is the same. What gamma distorts
   is (a) the discount on the delay between commitment and death: 0.998^31 = 0.94, 0.998^100 = 0.82,
   0.998^300 = 0.55, and (b) future growth, which training under-weights ~2x at frame 2,900 (training
   is the more growth-conservative objective). So gamma 0.999 buys 0.97/0.90/0.74 on (a): modest.
   The "largest survival gain so far" from 0.995 -> 0.998 was bundled with the kill-reward removal,
   the mass reward and mixed worlds in arm C; no gamma-only arm exists. Resolve: a gamma-only arm in
   the screen if there is room; otherwise read the delay-to-death distribution from the
   counterfactual study and compute the implied discount.
2. **"The census corroborates a real -0.10 to -0.15 gap vs the champion."** On episode survival,
   yes. The clause uses `prefix_h5000.survival_fraction` = alive frames / 5000 (`hooks.py:140`,
   `metrics.py`), which prices a death at frame 3,800 as 0.76 alive. B8's 49 late deaths (frame
   >= ~2,850 at B8's growth rate) and 26 early ones put its census-bank alive-frame survival near
   0.82; the champion's 10 deaths at median L 520 (~frame 3,000) put it near 0.87: a gap of about
   -0.05 on the metric the clause actually scores. Phase R's -0.156 on seed 0's bank is then a
   ~1-SE draw at SE 0.12 against a pooled truth near the threshold. Resolve: compute alive-frame
   survival from the census per-episode death frames and read look-2's per-world surv5 for B8_s0
   and the champion (artifact reads, no compute).
3. **How much the Huber fix can do.** Inside traps p ~0.69, so both fits see them; before the flag
   classes are barely separable (not-tight AUC 0.59-0.61), so an expectation fit penalises whole
   classes uniformly and ranks actions only in the middle zone. I predict a positive but smaller
   effect, with a mass cost (caution tax). Resolve: the screen's secondary endpoint (deaths/1k at
   L >= 800) and the RL memo's own value check (Q-gap must open >= 10 decisions before death on a
   majority of the 49).
4. **"Perception last, and the AUC study says hand features add little."** Agreed on the ranking,
   not the inference. The decisive test is whether the fatal and surviving actions differ on any
   available input at the commitment decision; AUC over all L >= 800 states cannot say. And the
   not-tight AUC of 0.59 argues that the needed information is a rollout, not a feature, which moves
   rollout labels up, not down. Resolve: the counterfactual study records, per death, the inputs of
   the taken vs the best surviving action at the commitment decision, plus B8's Q-gap there.
5. **n-step 16-32.** Fine as an arm, but note the in-window reward is a near-constant 0.2/step, so a
   32-step return carries almost only the terminal; the eps contamination at n=32 is 28%, and for a
   long snake the most dangerous random action is a boost (two collision-checked cells). Run one
   actor at eps=0 or exclude eps windows. Resolve: report the eps-contaminated fraction per arm.
6. **Growth governor as a reward cap.** The RL memo calls its legitimate form gamma 0.999 / terminal
   hero; I call it a reward cap on L. Both are objective-level and both lose mass; under the
   measured hazard slope (h' ~2.7e-7 < 2/(L tau) = 1e-6) growth is still break-even against B8's own
   late hazard, so neither pays under the mass metric and both exist only for the guard. Resolve:
   hazard per 1k decisions by L bucket within frame buckets; a cap pays only if hazard rises with L
   at fixed frame.

## 3. Updated lever ranking

1. Exact-replay counterfactual/commitment study on the 75 census deaths (shared; no training).
2. Zero-code learner screen: Huber knee (beta ~100 old scale or reward x0.01), anchor off, their
   combination, n-step 24 (2x2 plus one), 5 paired 8M fine-tunes, 96 scripted worlds per seed.
3. Rollout action labels through the existing LabelBuffer plus late-game/pre-trap snapshot
   restarts (training-time only), gated by the ceiling test and by the owner's planner ruling.
4. Death-window and L-stratified reservoirs; desynchronised segment phases if the checkpoint-phase
   probe shows a swing.
5. Calibrated death-probability auxiliary head (also the representation diagnostic).
6. Boost competence via rollout labels where a boost at the commitment decision survives.
7. Tail-topology inputs / mid-resolution plane: only if the commitment study shows per-action
   separation on a quantity B8 does not already see.
Not ranked: corpse-bait discipline (conditional on the corpse share at trap onset), reward cap on L
(only if hazard rises with L at fixed frame), hand rules at serving time (excluded).

## 4. Joint top-3 (pre-registrable)

**J1. Measurement day, no training.** (a) Exact replay of the 75 census deaths in GridBatchSim:
for each of the 60 decisions before death, play each masked-legal alternative with B8 greedy for 300
frames; record the earliest decision with a survivor (commitment horizon), the surviving action
(boost or not, tail-side or open-side), the owner of the cell that sealed the final pocket, whether
the own tail bounded it, B8's Q-gap at that decision, and the bv8/region inputs of the taken vs the
surviving action. (b) From census records: alive-frame vs episode survival for B8 x5, incumbent,
champion; hazard per 1k decisions by L bucket x frame bucket. (c) Linear probe on B8's 256-d fuse
features with the M7 capture rows (death within 50/100/300 at L >= 800, out-of-fold by world).
Routing rule, written down first: >= 50% of the 49 late deaths with a survivor within 100 decisions
and a Q-gap under 1 unit there -> J2 first; survivors exist but no input separates them -> J3 first;
< 30% with any survivor within 300 -> deaths are environmental, the guard is the issue. Cost: a few
Mac hours; owner decision: none.

**J2. Learner-regime screen.** Arms: CTRL (M6 CTRL finals reused if the recipe is bit-identical,
else re-run), HUBER, ANCHOR0 (lambda 0, label_frac 0 from step 0), HUBER+ANCHOR0, NSTEP24 (+HUBER,
one actor eps 0). 5 paired seeds (fine-tunes from B8_s0..s4, 8M, batch-scale 2). Probe no veto on 96
scripted worlds per seed on a fresh dev bank plus 32 frozen and 32 mixed; score both alive-frame
and episode survival. Primary: paired scripted alive-frame survival delta vs CTRL; secondary:
deaths/1k at L >= 800, mass delta, hold-out v8 agreement (collapse tripwire < 0.6). GO: delta >=
+0.06 with mass >= -30; KILL: delta <= 0; else extend to 192 worlds. Cost: ~4 arms x 5 x 25 min
training plus probes, Mac only. Owner decision: none (no pod spend).

**J3. Rollout labels and snapshot restarts (training-time only).** Step 1, ceiling test: B8 plus a
depth-100 rollout shadow (choose the action whose B8-greedy rollout survives longest, ties to B8's
argmax) on 48-96 scripted worlds, 1-2 seeds, L >= 600 decisions only; GO to step 2 iff scripted
survival >= 0.65. Step 2: label L >= 600 decisions with the same rollouts (keep only decisive labels:
one action survives >= 300 frames while the taken action dies within 100), feed them through the
on-policy LabelBuffer as the margin anchor, and restart a fixed share of worlds from late-game /
pre-trap snapshots; 5 paired seeds, same probe and thresholds as J2, plus a transfer check (no mix
may fall > 0.05). Owner decision: yes, twice: whether training-time lookahead labels count as "no
veto" (the served artifact is a bare network), and whether the ceiling test's ~8-10 Mac hours or a
pod are acceptable before Track 3's feasibility numbers exist.

## 5. Minimum context before the first experiment (deduplicated, prioritised)

Must: (1) train_m3c.py on branch m6: TD loss and beta, exact mass reward (is the PBRS term kept?),
n-step parameter, replay capacity/sampling, segment reset/stagger, LabelBuffer and label_frac
semantics, margin constant, and the lambda column of M6's log.jsonl. (2) Census per-episode death
frames -> alive-frame vs episode survival for all seven arms, and look-2 per-world surv5 for B8_s0
and the champion on seed 0's bank. (3) Hazard per 1k decisions by L x frame bucket from the census
captures. (4) The counterfactual replay study (J1a). (5) Linear probe on B8 fuse features (J1c).
(6) Training death records by L x segment frame x mix x eps flag over one full 1.28M cycle.
Should: (7) checkpoint-phase probe; (8) seed-ensemble mean-Q eval on 96 scripted worlds;
(9) champion on all five Phase R banks; (10) the 30 bv8 feature names and caps; (11) corpse share of
B8's intake at L >= 800, kills per episode and live greedy snakes inside the hero's region.
Nice: (12) Mac RAM and obs-pack bytes per transition; whether the n-step builder and Replay
parameterise n and capacity; M5 L60 intermediate probes vs cycle phase.

## 6. Guard clause, joint view

Agreed facts: one seed, one bank, 32 vs 32, SE ~0.12 against a -0.05 margin (false-fail ~34% under
equality, ~39% false-pass at a true -0.10), a comparator that spends 5.6% of its decisions above
L=800, and a metric (alive frames / 5000) that prices late deaths lightly. Mechanics adds that the
champion's survival is bought with size (0.17 cells/frame net of boost burn, 800 reached around
frame 4,000) and that on the clause's own metric the census bank implies a pooled gap near -0.05,
not -0.156; the RL memo adds that on episode survival a real -0.10 to -0.15 gap exists. Both are
true and must be reported side by side on the same bank.

Joint recommendation: keep a scripted-survival protection; re-specify it prospectively, once, before
any new gate data: pooled over all five banks, comparator the served agent (the stated goal's
comparator) at the unchanged -0.05 margin on alive-frame survival, plus a per-seed catastrophic
floor (no seed below -0.15) and an absolute floor on episode survival (no mix below 0.50); measure
the champion on all five banks now as a diagnostic; publish the sensitivity table (B8 under old and
new clause, both metrics); never re-score look 2 as GO. If the owner's product value is "the Watch
snake rarely dies", keep the champion comparator, fix its power (five banks), accept that B8 as
trained probably fails on episode survival, and pursue J2/J3 as the learned fix rather than a
smaller snake. This is the owner's call and the single largest lever on the table.
