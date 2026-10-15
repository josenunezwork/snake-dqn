# Mac answers to the panel decision memo (2026-10-14)

Read-only verification of the panel memo (`PANEL_DECISION_MEMO.md`, round-1/round-2 memos and
the librarian answers, 2026-10-15) against the actual trainer source on `main` and the artifacts
that exist only on this Mac. Everything the panel marked *inferred* about `train_m3c.py` is
checked here with line numbers; the "needs_mac_artifacts" items in the librarian table are
answered from the files. Two load-bearing panel claims are wrong (sections 2 and 5); the rest
hold. Nothing here is gate evidence.

Sources: `research/redesign_m3c_20261008/train_m3c.py` at `dde6def`; artifacts
`snake-dqn-artifacts/{redesign-m6-mac-20261014, redesign-m4-phase-r-nv-20261013,
b8-death-census-20261014, redesign-m4b-mac-20261013}`.

## 1. Trainer facts (train_m3c.py on main)

| Panel question | Answer | Where |
|---|---|---|
| TD loss and beta | `smooth_l1_loss(qa, y)`, PyTorch default beta 1.0, mean over 384 agent rows (768 at `--batch-scale 2`). No Huber-beta, reward-scale or TD flag exists. | `train_m3c.py:1787`; flag grep empty |
| n-step | `N_STEP = 5` module constant; `NStepBuilder(n_step=N_STEP)`. Not a flag. Death flushes the pending window as terminal (`disc = 0`, `nxt = -1`). | `:164, :635, :1814, :632` |
| Replay capacity / sampling | `--replay` (500k in every run); `Replay.sample` is uniform over the ring, filtered to rows still in the obs ring. No PER, no stratification. | `:426-429` |
| LabelBuffer | FIFO ring of (obs id, v8 action); `sample` uniform. | `:1098-1119` |
| Segments / actor phase | Fresh `GridBatchSim` of all E worlds per `--segment-frames 5000`; actor `j` starts at segment index `j` and advances by K (= actors), all actors start at frame 0 together. Actors are **in phase**. | `:1246-1257, :2167, :2406` |
| Reward composition | Sim v2 reward `gamma*Phi(L') - Phi(L) + death_value*[died] + kill_scale*victims` with `Phi(death) = 0` (**PBRS kept**), plus `mass_term = kappa*(1-gamma)*Phi(L_after)` for heroes alive after the step. Death value -3, kill 0 in C/B8/M6. | `grid_sim.py:845-868`; `train_m3c.py:232-235, :2338-2342` |
| eps | `eps_start 0.05 -> eps_end 0.01` linearly by `eps_end_at`; `--eps-end-at 1` means 0.01 from the first transition. | `:178-200` |
| Anchor schedule defaults | hold 8M, floor-at 15M, floor 0.1 (lambda 1.0 until hold, exp decay to floor). | `:175-196` |
| 30 bv8 scalars | `FEATURE_NAMES = (area_g, head_risky, landing_ok, v5_eligible, spacious)` x 6 actions. `area_g` cap = `min(max(32, 2L), 4096, open)`: census row at L 1284 has `area_cap 2568`. No tail-in-region, boundary-TTL or multi-step tail-reachability input. | `src/simd_env/v8_nb.py:93` |
| Mac RAM | 68.7 GB, 18 CPUs. | `sysctl hw.memsize` |

**The qmax "inconsistency" (memo 2.5) is resolved.** `qmax` is `max |Q|` over the agent rows
(`:1804`). With PBRS kept, the alive steady-state value is near 0 (PBRS telescopes to
`-Phi(L_0)`, the mass term adds `+kappa*Phi`), and the death target is `gamma*0 - Phi(L) - 3`.
The large |Q| are *negative* near-death values: the census records `q_fatal` at L 1284 as
-92 to -99 for all six actions (target -131). So the death delta is `Phi(L) + 3` = 100-130 Q units
either way and the Huber-beta-1 linear regime is confirmed. For T2 the beta route is the right
one; no reward or output-layer rescale is needed.

## 2. Correction: M6 ran the anchor at lambda 0.1, not 1.0 (panel S8 is wrong)

Every M6 run's argv has `anchor_hold 0, anchor_floor_at 1, anchor_floor 0.1` (the B8 recipe's
hold/floor-at were overridden). The `lambda` column of all 15 `log.jsonl` files is 0.1 at every
100k row. The logged anchor loss (before lambda) averages 0.004-0.006 against a TD loss of
0.06-0.20, so `lambda * anchor` was about 0.5% of the loss. The DQfD margin loss is zero wherever
the student's argmax equals the label, which is 99.98% of on-policy rows, so the "keep your
argmax trust region" is numerically dormant in fine-tunes.

Consequences: the trust-region explanation for the M6 nulls does not hold; the T2 ANCHOR0 arm is a
cheap ablation, not a hypothesis-bearing one. B8's own 20M training did run lambda 1.0 for its
first 8M (`anchor_hold 8000000, anchor_floor_at 15000000`), where the argument still applies.
ANCHOR0 on the M6 recipe is just `--anchor-floor 0` (hold 0 / floor-at 1 leave no interior `t`,
so no `log(0)`).

## 3. Replay cycle confirmed from CTRL_s10

`qmax` per 100k: dips to 92-96 at 1.8M, 3.1M, 4.4M, 5.7M, 7.0M (autocorrelation peak at lag 13 x
100k, r 0.43): period 1.3M = 2 actors x 128 worlds x 5000 frames, as the panel inferred.
`death_len_mean` per interval cycles 86 to 1230 with the same period; the "200-235" reading is the
last two intervals (204, 235) at the young-hero phase. Cycle-averaged it is about 600-700, so
training deaths happen at roughly eval lengths: the phase-confound view is right, the
starvation view is not.

## 4. Phase R look-2 per-world records (the guard cell)

Seed 0's scripted bank, 32 paired worlds, hero B8_s0 (`M4-selected-noveto`) vs champion_a5 + v8:

| Metric | B8_s0 | champion | diff | paired SE |
|---|---|---|---|---|
| surv5 (alive-frame, the clause) | 0.727 | 0.883 | -0.156 | 0.067 |
| episode survival | 0.406 (19 deaths) | 0.594 (13) | -0.188 | |
| mass | 480.5 | 380.3 | +100.2 | |

World-level correlation between the two agents is -0.03, so pairing by world buys nothing. The
incumbent on the same bank: surv5 0.837, episode 0.656 (B8_s0 vs incumbent -0.110 / -0.250;
incumbent vs champion -0.046 / +0.062).

Per seed on its own bank, B8 vs incumbent, scripted (surv5 / episode): s0 -0.110 / -0.250,
s1 -0.013 / +0.031, s2 +0.074 / +0.094, s3 +0.145 / +0.094, s4 -0.109 / -0.219. Pooled over the
five banks: scripted -0.003 / -0.050; frozen +0.011 / +0.069; mixed +0.038 / +0.100.

## 5. Census bank (round 52): alive-frame survival, death frames, hazard crosstab

| Arm | episode surv | alive-frame surv | deaths | median death frame | median L at death |
|---|---|---|---|---|---|
| B8 s0 | 0.656 | 0.899 | 11 | 4178 | 1106 |
| B8 s1 | 0.375 | 0.728 | 20 | 2638 | 793 |
| B8 s2 | 0.625 | 0.893 | 12 | 4092 | 1218 |
| B8 s3 | 0.531 | 0.823 | 15 | 3283 | 972 |
| B8 s4 | 0.469 | 0.778 | 17 | 3024 | 934 |
| B8 pooled | 0.531 | 0.824 | 75 | | 974 |
| frp3-s12 + v8 | 0.656 | 0.855 | 11 | 2929 | 643 |
| champion_a5 + v8 | 0.688 | 0.863 | 10 | 2870 | 520 |

B8's 49 late deaths (L >= 800) fall at frames 3344 / 4051 / 4554 (quartiles), mean alive-frame
cost 0.218 each; the 26 early ones at 1139 / 1792 / 2192, cost 0.669. In clause units the
census-bank gap vs the champion is **-0.039 pooled** (B8_s0 alone +0.036), inside the panel's
rule R5 band [-0.08, -0.02]: "at the margin". B8_s0 across its two 32-world banks: 0.899 vs 0.727
alive-frame (11 vs 19 deaths); the champion on the same two banks: 0.863 vs 0.883. A one-bank,
one-seed guard cannot resolve a 0.05 margin.

**Hazard per 1,000 decisions, B8 pooled, L bucket x frame bucket** (deaths / decisions):

| L | frame 2000-3000 | 3000-4000 | 4000-5000 |
|---|---|---|---|
| 600-800 | 0.13 (9 / 69.0k) | 0.00 (0 / 0.7k) | |
| 800-900 | 0.16 (6 / 37.7k) | 0.15 (1 / 6.9k) | |
| 900-1000 | 0.10 (2 / 19.2k) | 0.14 (4 / 27.8k) | 0.00 (0 / 0.4k) |
| 1000-1100 | 0.00 (0 / 4.6k) | 0.20 (8 / 40.6k) | 1.05 (5 / 4.8k) |
| 1100+ | 0.00 (0 / 0.3k) | 0.07 (3 / 43.0k) | 0.22 (20 / 93.0k) |

At a fixed frame bucket the hazard does not rise with length (3000-4000: 0.15, 0.14, 0.20, 0.07
from L 800-900 to 1100+). At fixed length it rises with frame (L 1100+: 0.07 -> 0.22; L 1000-1100:
0.20 -> 1.05). **Rule R6 does not fire: hazard is a function of late-game crowding, not of L.
Drop every growth-governor form.** Incumbent and champion have 0 deaths at L >= 800 in frames
3000-4000 on 12k and 0.3k decisions; uninformative. Counts are small throughout.

**Trap onset** (census `t0`, 49 late deaths): onset-to-death quartiles 20 / 59 / 92 frames; best
one-step landing area at `t0` median 132 cells (IQR 32-159), one frame earlier median 230
(IQR 160-1872). The pocket at the seal is about 130 cells, not 30-60; in a quarter of the cases
the region drops from above 1800 to below 160 in one frame, which is a neck closing. 19 of the 49
late onsets had a one-step escape action.

**Census README**: the line "a v8 veto would not have changed any B8 death" is reworded. Shadow
v8 agreed at all 75 fatal decisions and in every trap run, but 12 of its 30 disagreements over
the whole run fall inside 11 episodes that later died, 70 to 3,380 decisions earlier.

## 6. What this changes in the panel's plan

- T2 arms: HUBER (new flag, beta ~100 at the current Q scale) is the hypothesis-bearing arm.
  ANCHOR0 stays as a cheap ablation. CTRL can reuse the M6 CTRL_s10-14 finals (all exited 0).
- Governor levers (reward cap, survival weight, smaller snake): dropped by R6.
- Guard: the census-bank gap vs the champion in clause units is -0.04 pooled; look-2's -0.156
  came from the one bank on which B8_s0 died 19 times out of 32. The amendment question is an
  owner decision; the data say "at the margin, one bank", exactly R5.
- Still open from T1: GridBatchSim deep-copy parity and timing (the census `clone_sim` exists at
  `research/b8_death_census_20261014/census.py:215`, not yet parity-tested over a long replay),
  the commitment/contrast rollouts (Part A), the fuse-layer probe (Part C), the checkpoint-phase
  probe, the champion on the other four banks, the seed ensemble.
