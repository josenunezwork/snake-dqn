# Round 1 judgment — big-lever research rounds

Judge: Fable (round-1 judge), 2026-10-21 ~00:15 PT. Inputs: `../CHARTER.md`, `../LIVE_FACTS.md` (23:45 PT entry),
`QUESTION.md`, `astra_01..10.md`. Repo verified read-only at `ab4fe3a8` (main). Sealed study not opened. No jobs run.

## (a) Bottom line

1. Ten independent advisors, asked for a map, converged more than a divergent round should: **5 of 10** (02, 08, 09, 10
   and 01's P3) name the same mechanism — **long-horizon on-policy/online RL from a B8 warm start, fixed served
   opponents, undiscounted H5000 mass-integral return, 10^8–10^9 hero transitions, optionally recurrent, served as a
   small bare net.** Its premise is that every prior fine-tune (T23, S1, FRP) trained 10^7 transitions with replay DQN,
   γ-restarts and safety anchors — an objective ~500 decisions deep against an endpoint 5,000 frames deep.
2. The second cluster (01, 08, partly 04/06/07) is **the GPU-resident exact engine** — an enabler, not a lever. I verified
   the spike as written cannot be fast: `research/gpu_sim_spike/ego_device.py:142` runs an **H×W = 12,035-iteration**
   flood per observation and `v8_device.py:144` up to 4,096; dispatching the frozen L4 manifest unchanged will time out.
3. The third cluster is **representation** (07, 10): B8's global raster is 8×8-cell mass blocks and local TTL clips at 255
   (`src/simd_env/ego_raster.py:25-57`, verified), so distant connectivity and release order are not in its inputs. No
   recurrent or full-resolution ego2s student has ever been trained (grep: no GRU/LSTM in `src/` or `research/`; the
   deleted June DRQN was on the vector61 line). This is untested, not refuted.
4. Search-as-signal with better labels (04, 06) is a legitimate variant but sits on top of the projection problem that
   just printed **Claim A D0 KILL** (`LIVE_FACTS.md`); it is deferred behind T34's Round 0 verdict and the D0 rerun.
5. World models (05) and leagues (03) are low-confidence by their own authors' admission; not funded as big levers.
6. The running shortlist is below. The three experiments to run now: a 2-arm PPO pilot (recurrent vs feedforward, from
   the B8 trunk, mass-integral return, CPU sim + GPU learner, ≤20 u), an offline information-value test of
   full-resolution geometry + history (≤16 u), and the flood-kernel fix followed by the frozen GPU-spike run (≤20 u).
7. All ten advisors reject the direction judge's "2+ weeks to first signal" as a blanket rule; I rule them right for
   bounded pilots on existing infrastructure and the direction judge right for a qualified GPU engine. Both proceed in
   parallel because the calendar, not units, is now the constraint.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | GPU-native sim engineer | **keep** | The one advisor who read the spike's kernels: **`_component_sizes` loops `range(H*W)`** and **`_flood` up to `_AREA_MAX`** (verified). Correctly separates throughput falsification (days) from exact-RNG parity (weeks). Its L4 20–80k world-steps/s target is a conjecture consistent with gpu_expert_1's 0.25–0.4M row-steps/s (×6 rows/world). Correct that `reference.py` rescoring replays actions without recomputing policies. |
| 02 | Large-scale on-policy RL | **keep** | Best-specified recipe: PPO, 256-unit GRU after fusion, zero-init residual policy head, per-frame alive-mass/1000 reward with γ=1, GAE λ=.995, KL-stop .01, B8 softmax warm start with calibrated temperature. **T23 three changing learner seats, S1 γ=.998/ε=.01/anchor=.1 still −12.5/−14.9** verified. Throughput floors (2k/s) are assumptions, flagged as such. |
| 03 | Self-play / league | **partly** | Clean H−S−F design isolating adaptation from distribution shift, and honest that B8 already dies against *scripted* anchors (**T23 scripted −101** verified). But the mechanism (changing opponents teach own-body routing) is the weakest fit to a fixed-roster target. Not a big lever; a diversity add-on if lever 1 works. |
| 04 | MuZero / sampled search | **partly** | Right that six actions need no Gumbel and that projection, not label volume, is the bottleneck (**LABELS 99.64 % decisive agreement yet −55 both-alive mass**, verified). Its full-horizon mass-return Q̂ with KL-bounded projection is a different label from T34's survival-only ranker. But **Claim A D0 KILL** (LIVE_FACTS) says the frozen loss cannot flip decisive rows; this lane waits for the rerun and Round 0. |
| 05 | World models | **partly** | Honest: sim stepping is 3.4 % of worker time, so a physics-only model misses the bottleneck (**v8 56.8 %**, verified); 0.99^300 arithmetic correct. The learned model is a fallback if the exact engine fails on cost. Not funded first. |
| 06 | Distillation of search | **partly** | Concedes the Claim A amendment already owns the loss fix; its addition (continuation-consistent, set-valued regret labels, 15 % precision arithmetic at 0.025 % prevalence) is sound but incremental on a lane that just KILLed at D0. Fold its auditing requirements (post-intervention continuations, precision not recall) into lever 4. |
| 07 | Opponent modelling / partial observability | **keep** | **8×8 global blocks, TTL clip 255, two global convs** verified; **T10 explicitly rejects a ceiling claim for other models** verified. Its offline information-value test (A: B8 inputs; B: +full-res geometry; C: +128-decision history; regret on T3-bank roots) is the cheapest representation falsifier proposed. Factual nuance: its "historical GRU failure" was the vector61 DRQN, never an ego2s student — its prior should be higher, not lower. |
| 08 | CUDA / Colab cost | **keep** | Placement table and arithmetic correct (**A100 host = same 12 vCPU**, verified; 504 runtime-h/week at 3 slots). Right that CUDA graphs cannot fix a host-bound teacher. PQN choice is defensible but the existing trainer **imports `RasterDuelingNetwork`/`BatchSim`** (verified) — a port, not a reuse. Note: the 3-assignment cap is a self-imposed assumption (LIVE_FACTS), test pending. |
| 09 | Evaluation / trust skeptic | **keep** | Most useful rules of the round: training RNG/float order may differ, evaluation semantics may not; audit consequential disagreements separately because **75/301,589 = 0.025 %** can hide inside 1 % disagreement; **T3's +122.8 is 32 scripted worlds** (verified); 1,374 paired worlds for 80 % power on +0.03 survival; fixed-N acceptance at a true +40 is ~50 %. Plan operating characteristics at +60/+80. |
| 10 | Slither.io mechanics | **keep** | Corrects the round's own fact sheet: the **150 is a board-wide cap on corpse-class pellets (kill corpses + boost trail), oldest evicted** (`mechanics_constants.py:30-64`, verified) — not 150 per corpse. Strategic picture (broad sweeps, exits preserved, harvesting not hunting) matches the offense judgment. Its pilot design (recurrent vs parameter-matched feedforward PPO, same B8 trunk, same reward) is the cleanest 2-arm test in the round. |

Factual errors found: none that change a conclusion. Corrections to the round's fact sheet: (i) the corpse cap is
board-wide (above); (ii) ego2s recurrence is untested, not failed; (iii) the spike's flood loops make its current
throughput manifests moot until rewritten.

## (c) Disagreements ruled

1. **PPO (02, 10) vs PQN (08, 09).** PPO for the pilot: it can target the undiscounted mass-integral directly through a
   critic, warm-starts from B8 logits with a KL trust region (the T23/S1 failure was losing B8's productive policy, not
   failing to be cautious), and needs no new return machinery. PQN is a backup arm if PPO's on-policy sample cost is the
   binding constraint after the engine exists; the existing PQN code targets the wrong raster and sim.
2. **"2+ weeks to first signal" (direction judge) vs all ten.** Both are right about different things. A bounded pilot on
   the existing CPU sim + GPU learner is a 2-day object; a qualified device-resident engine is 1–2 weeks. Since units no
   longer bind, run the pilot now and build the engine concurrently; do not gate one on the other.
3. **GPU throughput: gpu_expert_1 (0.25–0.4M row-steps/s) vs Astra 01 (20–80k world-steps/s).** Compatible units
   (6 rows per world-step), but 01 is right that the current kernels cannot reach either number: the flood loops must be
   replaced by bounded wavefronts or label propagation before any manifest is worth its units. Admission floor: 20k
   full world-steps/s on L4 with five real opponents and long bodies.
4. **Representation (07) vs "T8–T10 proved the ceiling".** T8 tested a 294k-parameter net on the *same compressed*
   raster (`t8_probes:31-38`); T10 disclaims ceilings for other inputs. 07 is right that full-resolution geometry is
   untested. The information-value test settles it offline for ≤16 u.
5. **Search labels (04, 06) vs the LIVE D0 KILL.** The KILL is about the frozen survival-teacher loss; 04's mass-return
   operator is different but shares the projection bottleneck. Ruling: no new distillation lane until the D0 rerun and
   Round 0 verdict; if the exact engine qualifies, the natural home for search is *inside* the RL loop (auxiliary
   targets, ExIt-style) rather than another offline fit.
6. **Hunting vs defense.** Unanimous with the offense judgment: defense (late large-body survival) is the lever; corpse
   harvesting is additive; hunting is capped by the shared 150-pellet pool. Nothing to re-litigate.

## (d) Running shortlist (new this round)

| # | Lever | Conf. it becomes the push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Long-horizon on-policy RL from B8** (PPO, γ=1 mass-integral return, fixed opponents, KL trust region, recurrent option) | 45 % / 25 % | 20 u + 2 Sol days; scale 150–400 u | E2 pilot: recurrent vs feedforward PPO from the B8 trunk, 2×8–20M, fresh 96/32/32 screen vs B8 |
| 2 | **GPU-resident exact engine** (enabler for 1, 4; and for T34 labels) | 35 % reaches ≥20k world-steps/s on L4 within a week | 20 u + Sol 1–2 days (floods) then 4–7 days (qualify) | E1: bounded flood kernels, then the frozen spike on L4 (10× cost, ≥99 % agreement, decisive-state audit) |
| 3 | **Representation: full-res geometry + opponent history student** (feeds lever 1's architecture) | 30 % / 20 % | 16 u + 1–2 Sol days | E3: offline regret test A/B/C on T3-bank roots; GO if ≥20 % regret reduction, LB90 > 0, ≤1 % unnecessary switches |
| 4 | Search-as-training-signal with full-horizon mass-return targets + measured projection regret | 15 % / 15 % | blocked | Wait for D0 rerun + Round 0; then 04's projection experiment (≤20 u) only if D0 flips to non-KILL or Round 0 GO |
| 5 | Learned world model for imagination | 10 % / 5 % | 20 u | Only if lever 2 fails on cost: 05's structured-model decisive-root probe |
| 6 | League / history curriculum | 5 % / 5 % | 20 u | Not funded now; revisit as opponent-diversity add-on after lever 1 shows a positive sign |

## (e) Still unresolved

- Whether a B8-warm-started learner can *gain* mass at all: every fine-tune since B8 lost (T23 −80.8, S1 −12.5/−14.9,
  T4c, T6). The pilot must distinguish "optimizer/objective damage" from "representation limit" — Round 2's question.
- Collector throughput on a 12-vCPU Colab host with the CPU sim and frozen opponents (v8 hook 57 % of worker time);
  pilot sizing assumes 2k hero transitions/s, unmeasured for PPO collection. S2 native v8 (1.4–2.2×) helps.
- The 3-concurrent-assignment cap (self-imposed; test pending) and whether T34's lanes leave two slots.
- Parity doctrine for GPU-trained artifacts: advisors agree training RNG/float order may differ and evaluation may not;
  the exact wording of a "qualified" GPU-trained candidate (matched-event fixtures, decisive-state audit, Mac strict)
  needs a written rule before any scale run.
- How a Round 0 GO (teacher acting beats B8) changes priorities: it would make lever 2 also T34's label accelerator.

## (f) Experiments Claude can run now (≤20 u each; dispatch via Sol between rounds; report in LIVE_FACTS.md)

**E1 — Flood-kernel fix + frozen GPU spike on L4.** Sol replaces `ego_device._component_sizes` (H×W loop) and
`v8_device._flood` with bounded wavefronts (cap at the v8 caps 160/4096 where the CPU kernel caps; for the uncapped
ego region size use iterative min-label propagation or a per-world CUDA-kernel BFS), keeps the CPU oracle, then Claude
dispatches the existing L4 manifest unchanged in rule. Cap: 15 u spike + 5 u probe, 48 h. Kill: frozen rule
(≤0.1× L4 CPU cost per dense decision, ≥99 % first-action agreement, 0 unexplained) plus 09's addition — report
agreement separately on decisive roots; below 20k full world-steps/s on L4 after the fix → STOP the engine thesis this
week. Precondition for levers 2 and 4.

**E2 — PPO pilot, recurrent vs feedforward, from the B8 trunk.** Two arms × one seed, 8M hero transitions each
(raise to 20M only if the E4 ruler shows ≥4k/s): B8 encoder copied, actor logits = calibrated softmax of B8 Q (≈1 %
non-greedy), zero-init residual policy head, critic refit before actor unfreeze, reward = alive hero mass/1000 per frame,
γ=1 to H5000 with zero bootstrap at death, GAE λ=.995, 2 epochs, clip .1, KL-stop .01, LR 1e−5 encoder / 1e−4 new.
Arm R adds a 128–256-unit GRU (reset on hero death, 32-step burn-in); arm F is parameter-matched feedforward. Fixed
equal mixes, CPU GridBatchSim collector + L4 learner. Screen both endpoints and B8 on fresh 96/32/32 paired worlds.
Cap: 20 u, 10 L4 allocation-h, 48 h after Sol's build (1.5–2 days). Kill: tripwires (NaN, >90 % one action, boost share
> 3×B8); either arm UB90 ≤ 0 vs B8 pooled mass → KILL that arm. GO to scale only if an arm shows ≥+10 pooled with
non-negative scripted no-death and survival band v2; record R−F separately. Nothing here is a recipe claim.

**E3 — Information-value test for representation (lever 3).** Reconstruct ≤256 unsealed T3-bank worlds / 1,536 roots
(half decisive, half counterweight; world-disjoint 192/32/32 split); fit three parameter-matched action-ranking
students (A: B8 inputs; B: + full-res 145×83 owner-relative occupancy and unclipped release order; C: B + 128-decision
hero/opponent history via 64-unit GRU); target = D300 mass-return regret under the recorded continuation. Cap: 16 u
(≈3 h CPU reconstruction + 4 L4 GPU-h fits), 48 h. Kill: INCOMPLETE if histories cannot be reconstructed; otherwise GO
only if B or C cuts held-out decisive regret ≥20 % vs A with paired whole-world LB90 > 0 and ≤1 % unnecessary switches;
keep history only if C beats B by ≥10 %. A GO changes E2's architecture for the scale tranche.

**E4 — Collector throughput ruler (precondition for E2 sizing).** One L4 runtime, 30 min, ≤2 u: hero transitions/s
for PPO-style collection with 128 worlds, frozen roster, S2 native v8 on, 6 vs 12 single-thread processes, long-body
fixtures. Report active and allocated rates. Kill: < 1k/s → shrink E2 to 4M per arm and say so before data.

**E5 — Colab assignment-cap test (already pending, ~1 u).** Confirms whether a fourth concurrent runtime is possible;
changes every week-plan's slot arithmetic.

Deferred (not now): 04's projection experiment (lever 4) until D0 rerun + Round 0; 05's model probe until E1 fails;
03's H−S−F curriculum until E2 shows a positive sign.
