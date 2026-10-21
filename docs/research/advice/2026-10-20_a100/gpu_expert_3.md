# GPU expert 3 — learning and the network (independent answer, 2026-10-20 PT)

Vantage point: large-batch RL on accelerators, distillation, torch.compile/AMP, A100 specifics. Read-only; one
Mac micro-probe (1 thread, nice 10, ~6 s). No Colab/RunPod jobs. Labels: **VERIFIED** (file:line / probe) or
**ESTIMATED** (arithmetic shown).

## Answer to the open question (≤ 15 lines)

1. **Pointing an A100 at *learning* does not move the calendar either.** The student is 1,638,263 params
   (VERIFIED probe); one 1-thread Mac CPU core already does 1,060 rows/s fwd+bwd+Adam at B=1,024 (VERIFIED probe),
   an L4 ~25k rows/s (M2b record). A full 800-world round (~13–16M rows × 5 epochs) is ~45–55 min on an L4, ~15–20
   min on an A100 — the whole R1+R2+R3 fit chain saves ≤ 2 h of lane time, zero Mac time (ESTIMATED §3).
2. **The data floor is the host, twice:** labels are 57 worlds/lane-h (VERIFIED CAMPAIGN_LOG 15:35), and the fit
   cache is a *CPU replay regeneration* of each world's rasters before epoch 1 (VERIFIED cache.py:1-6,287-295).
   An A100 would consume a lane-hour of labels in ~40 s. HBM/tensor cores/80 GB buy nothing at 12 KB/row, B=1,024.
3. **A GPU-resident PQN/PPO sim is the wrong bet for "soonest":** every bare-network RL/self-play lever since B8
   failed (T23 −80.8, S1 KILL, M-series), the sim's Python-ordered food/RNG and vector61 opponents have no torch
   port (gpu_featurizer.py is the *old* raster31v2 obs, not B8's ego2s-bv8), parity CI alone is 10–14 pd per the
   blueprint, and whatever it learns still pays the same Mac strict gate. Earliest credible: mid-November, not Oct 25.
4. **What a big GPU does make cheap, and the plan is wasting, is already on disk:** every teacher row exports
   per-action 300-step rollout outcomes `(survive, alive_steps, mean_mass)` for all 6 actions (VERIFIED
   exports.py:132-147) and the fit **discards them**, keeping only `action_star`/`q_prev`/`forbidden_mask`
   (VERIFIED data.py:135-137, cache.py:26). Regress a 6-way **H300 outcome head** (BCE survive + Huber mass) on
   them, B8 trunk frozen first (bounds T3's spillover), then unfrozen/3× wider as the L3 2×2 already allows. That
   gives the served net the 50–80-step horizon as a *calibrated value of each action*, not a 1-of-6 argmax copy.
5. **Calendar:** Sol ≈ 1.5–2 days (head + loss + cache columns + export of a bare Q-only student), no new labels
   needed for the probe. The strict gate (~1–2 Mac days) is unchanged, so the gain is **≤ 2–3 days only if** the
   Claim-A labels run on Colab in parallel with Path B retention (needs the §5 prospective ruling); otherwise none.
6. **Cheapest experiment (§6):** the 39 A5 export worlds already on disk, L4, frozen trunk, ≤ 1 h, ≤ 2 u.
   GO if held-out per-action survival AUC ≥ 0.80 on decisive rows and outcome-argmax agrees with the teacher ≥ 85%;
   KILL at AUC ≤ 0.77 (= the reactive ceiling T8/T5b; then the planner/Path B is the only route to the horizon).

## 1. What the student is, and what a fit costs (verified)

| Fact | Status | Evidence |
|---|---|---|
| Ego2sNet: local 3 convs (stride-2 ×2) → 64·8·8 → 256+LN; global 2 convs → 32·10·10 → 128+LN; fuse 256+128+60 → 256; dueling V + A(6) | VERIFIED | `src/model/ego2s_network.py:30-72`; BV8 = (7 ch, 60 scalars) `:27` |
| Parameters 1,638,263 | VERIFIED | probe (`Ego2sNet(7,60)`, `sum(p.numel())`) |
| Mac 1-thread CPU, B=1,024: **966 ms/update = 1,060 rows/s** fwd+bwd+Adam; fwd-only 571 ms = 1,792 rows/s | VERIFIED | probe, `OMP=1, torch.set_num_threads(1), nice 10`, random uint8 inputs |
| Fit recipe: 4–6 epochs, B=1,024, Adam 2.5e-5, CUDA path implemented but unqualified, one pinned prefetch batch | VERIFIED | `research/t34_exit_b8_fit/train.py:124-144`, README :28-31 |
| Cache = per-world mmap of `local/global/scalars/q_prev/forbidden_mask/full_sight`; built by **replaying** each world (`regen_seconds`, `regen_seconds_per_row`) | VERIFIED | `cache.py:1-6,26,128,287-295` |
| Loss uses only `q_prev`, `action_star`, `forbidden_mask`, `sample_weight` | VERIFIED | `loss.py:31-34`; batch fields `data.py:135-137` |
| Exports carry per action `{survive, alive_steps, mean_mass(fp16), horizon, censored, tested}` ×6 plus `forbidden`, `length`, `reason` | VERIFIED | `research/t34_exit_b8/exports.py:132-171` |
| A100 learner (old DQN line, bs2 fp32) 78–83 upd/s alone; bf16 117–118 (1.5×); bs4 linear (no gain) | VERIFIED | `t16_colab_smoke02_results_2026-10-17.md:187-194,210-212` |
| L4 learner 29–30 upd/s at 87–90 % GPU, ~2 u/h; A100 ≈ 3.2× the units | VERIFIED | same doc :497-505; brief |
| Labels: 57.2 / 54.4 completed worlds per L4 lane-hour, 16–20k rows/world, ~8.7 MB/world, ~0.036 u/world | VERIFIED | CAMPAIGN_LOG 10-20 ~15:35 |
| Serving headroom: Watch6 update p95 8.397 ms vs 11.688 ms bar | VERIFIED | `t12_b8_serving_qualification_2026-10-17.md:184`; SUCCESS_CONTRACT §4 |

Row bytes: local 7·31·31 = 6,727 + global 4·37·37 = 5,476 + 60·4 = 240 → **12.4 KB/row** (ESTIMATED from shapes).
An 800-world round ≈ 800 × 18k = 14.4M rows ≈ **180 GB** of uint8 rasters — the cache is SSD-streamed by design;
no GPU memory size matters. At 25k rows/s the input stream is 310 MB/s: ~1/6 of an L4's PCIe, ~0.01 of A100 HBM.

## 2. Why an A100 (or 3 L4s) on the fit does not change the date

ESTIMATED fit wall per round: 14.4M rows × 5 epochs = 72M row-passes. L4 at 23–29k rows/s (M2b) → 41–52 min;
A100 fp32 ≈ 3× L4 on this conv/GEMM mix → 14–18 min; bf16 autocast +1.5× on both (T16 ratio). CUDA graphs are
irrelevant at B=1,024 (kernel launch ≪ 1 ms per step of ~40 ms); `torch.compile` gives perhaps 1.2–1.4× on the
small convs. Across R1+R2+R3 (≤ 4 fits) the A100 saves ≤ 2 lane-hours and **zero Mac hours**. The brief's "fits are
minutes" is wrong (they are ~¾ h on an L4 incl. a CPU-bound cache build on epoch 1), but ¾ h is not a calendar item.

Data floor with a 12-vCPU host: labels 55 worlds/lane-h ≈ **1.0M rows/lane-h**; cache regen is a bare replay of the
same worlds (no search), ESTIMATED 1–2 ms/row single-thread → 12 procs ≈ 6–12k rows/s ≈ 20–40 min per round. A
GPU consuming 25k rows/s needs ~90 label lanes to be busy. So on this pipeline the GPU is ≥ 95 % idle regardless of
model; three L4 lanes for *labels* (3 × 57 worlds/h) are the only throughput lever, and they are already funded.

"What fit would you run if it could take 10 GPU-hours?" Nothing useful: with ~14M rows per round the model is
data-bound, not optimizer-bound; longer fits only overfit the bank (T3 already reached 99.66 % bank agreement and
lost, `t3_diagnostics_2026-10-16.md:26-28,67`). Ten GPU-hours would buy a 20–40× larger student or 200 epochs — neither
addresses why T3 lost (late-growth deficit on both-alive frames, `:181`).

## 3. The levers a big GPU *could* make cheap, judged for "soonest"

| Lever | Verdict for calendar | Why |
|---|---|---|
| **(a) Outcome/value head on exported per-action rollouts** | **DO** (L4 is enough) | Data exists (exports.py:132-147) and is discarded (data.py:135-137). Turns 6 argmax labels into 12 regression targets per row (survive logit, mean mass) → the net learns a *graded* 300-step value, which is what the 0.74–0.77-AUC reactive predictors lack (T8 `:56-66`). Serve `argmax(Q_B8 + λ·mass_300 − μ·(1−p_surv))` or the head alone; bare network, same serving cost (+1 linear layer). Addresses REVIEW_A #9 (wrong-ordered margins at stationarity): regression targets have no margin pathology. |
| (b) 3–5× larger student (width ×2, +1 conv per tower) | OK, after (a) proves signal | Serving headroom 8.4 → 11.69 ms. Fit cost ×3–5 → 2–4 h/round on an L4, still minutes on A100 — this is the single case where A100 beats L4 on *lane* time, worth ≤ 3 h/round; does not beat 3 L4 lanes in parallel. Only after a signal exists; capacity was never shown to be T3's limit. |
| (c) Recurrent / lookahead-distilled student | No (now) | Serving is server-authoritative per tick; a GRU adds state-carry to the serving contract + strict-gate identity work (weeks). The DRQN line was deleted in June for losing every benchmark. |
| (d) Direct RL on a GPU-resident sim (PQN/PPO at 10⁵–10⁶ sps) | No (for this goal) | No torch sim exists: `gpu_featurizer.py` ports the *raster31v2* featurizer, not ego2s-bv8 (its docstring :1-12); `GridBatchSim` keeps CPython `random.Random` food spawns and order-dependent collision resolution (grid_sim.py:28-36); opponents are vector61 (0.86 MFLOP/row) + scripted. Blueprint priced parity CI at 10–14 pd and PQN trainer 9–12 pd (`ml_redesign_blueprint_2026-07.md` §10). Transfer risks: food-tie and corpse mechanics may differ (fine for behaviour), but opponent policies must be the same frozen/scripted roster or the strict gate measures a different game. And the bare-RL track record since B8 is 0 for 6. Mid-November at best. |
| (e) Self-play with the search teacher as value oracle | No | T23 league: −80.8 mass, 0/5 seeds; teacher is host-bound (7.5–7.8 % B8 forward). |
| (f) Regress B8 onto its own H-step survival outcomes (no search) | Cheap hedge, not first | It is (a) with N=1 and the hero's own action only; T3's label bank was this family ("B8-continuation labels") and lost. Keep as a control arm inside (a): teacher outcomes vs own-continuation outcomes. |

What may differ between training env and the Mac reference sim for (a)/(b): nothing — rows are replayed from the
exact sim (cache.py validates replay hashes, `train.py:112-115` refuses > 2 % mismatch). That is the right choice;
keep it.

## 4. Concrete design for lever (a) — "outcome head" (for Sol; ESTIMATED sizes)

- **Targets per row:** for each action a ∈ 0..5 with `tested[a]`: `y_surv[a] ∈ {0,1}`, `y_mass[a] = mean_mass[a]`
  (fp16 in export; already horizon-normalised, exports.py:139-141); mask untested/censored actions.
- **Head:** `nn.Linear(256, 12)` on the fuse vector (6 survival logits, 6 mass regressions); +3 K params, +0.003 ms.
  Loss: masked BCE(surv) + Huber(δ=0.1 after per-bank z-scoring of mass) + the existing KL anchor on non-decisive
  rows (keeps B8 behaviour where nothing was searched). Drop hinge/advantage-MSE for this arm.
- **Arms (fits the registered L3 2×2 shape):** trunk frozen vs trainable × head-only decision vs blended decision.
  Serving rule frozen pre-data: `a* = argmax over advisory-legal a of  Q_B8[a]·(1−w) + w·(m̂[a] − κ·(1−p̂[a]))`,
  w, κ chosen on the development probe only.
- **Cache change:** add two `(N,6)` float16 columns + `(N,6)` bool mask to `_ROW_ARRAY_FIELDS` (cache.py:26); cache
  identity hash bumps; no export change.
- **Device:** L4, bf16 autocast, B=1,024 (or 4,096 with lr scaled — B8's recipe showed no bs gain, keep 1,024).
  Memory: 1.6M params + Adam ≈ 20 MB; activations at B=1,024 ≈ 1024 × (32·31·31 + 64·16·16 + 64·8·8)·4 B ≈ 210 MB.
  Fits 1 GB; an L4 is already 20× oversized. A100: not needed.
- **Tripwires:** FINAL_PLAN §8 set (one-action share, all-length disagreement, boost share) apply unchanged.

Engineering: Sol 1.5–2 days (head, loss, cache columns, probe hook, export-strip of the head for the bare
checkpoint, tests on the existing `FakeReplaySource`), Claude ½ day review. No Mac time until the screen.

## 5. Calendar, honestly

- Today's binding sequence (brief; HANDOFF): Mac Phase 0 → Wed 10-21 ~13:00, Path B retention ~27 h on the Mac,
  then strict gate. Claim-A labels (Round 0 running on Colab; R1 ~14 lane-h = ~5 h on 3 lanes) and fits are on
  Colab and can overlap the Mac work *if* the SUCCESS_CONTRACT §5 vs DESIGN_V2 :103 overlap ruling goes that way.
- With lever (a): probe (§6) by **Oct 21–22**; if GO, R1 labels + outcome-head fit **Oct 22–24** on Colab while
  the Mac runs Path B; Mac screen (32-world H5000 ×3 mixes, ~hours) **Oct 24–25**; strict gate Tier 1+2 **Oct 26–27**;
  serving qualification Oct 27–28. That is the same Oct 25–30 window's early edge: **~2–3 days sooner than the
  late edge, not sooner than Oct 25**, because the Mac gate is serial and cannot start before Path B finishes.
- With any GPU-sim/RL lever: ≥ 3 weeks of Sol before the first candidate → mid-November, after the units deadline.

## 6. Single cheapest experiment — "outcome-head probe"

- **Fixture:** the 39 A5 v2 export worlds already on disk
  (`snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s*-output/exports/`, ~0.65M
  rows, ~340 MB; VERIFIED paths in CAMPAIGN_LOG 15:35). World-level 5-fold split (no row leakage across worlds).
- **Shape/device:** B8 s3 trunk **frozen**, `Linear(256,12)` head, bf16 autocast, B=1,024, 5 epochs, Adam 1e-3 on
  the head; one L4 lane. Cache regen of 39 worlds ≈ 2–4 min on 12 procs; fit ≈ 3.3M row-passes / 25k ≈ 2–3 min.
  **Total ≤ 1 h incl. setup, ≤ 2 u.** Control arm in the same job: same head trained on the hero's *own-continuation*
  outcome only (lever f), to show the search outcomes carry information the bare continuation does not.
- **Thresholds (pre-registered, decisive rows, held-out worlds, world-bootstrap 95 % CI):**
  - **GO:** per-action survival AUC ≥ **0.80** (LB > 0.77) **and** outcome-argmax = `action_star` on ≥ **85 %** of
    decisive rows **and** teacher-arm AUC − control-arm AUC ≥ 0.03 → fund R1 with the outcome-head arm as S1c.
  - **KILL:** AUC ≤ **0.77** (indistinguishable from T8/T5b reactive ceiling 0.742–0.769, `t8_probes:56-66`) or
    argmax agreement < 70 % → a frozen-trunk net cannot read the rollout outcome from the state; try the trainable
    trunk once (same job, +3 min), and if still ≤ 0.77, conclude the horizon is not state-recoverable by this
    representation → serve the planner (Path B) and stop buying distillation labels.
  - Between: run the trainable-trunk arm and the 2× width arm (+10 min) before deciding.
- This probe needs no Mac, no new labels, no A100, and answers the question the whole A100 round skipped: is the
  50–80-step horizon *learnable from these states at all*, independent of how many argmax labels we buy.

## 7. Where an A100 specifically would matter (for the record)

Only when the fit itself becomes the lane bottleneck: ≥ 5× student **and** ≥ 3 rounds/day **and** labels arriving
faster than 25k rows/s — i.e. ≥ 90 label lanes. HBM2e bandwidth is irrelevant at 310 MB/s input; 40/80 GB is
irrelevant at < 1 GB resident; tensor cores help only under bf16, which T16 showed is 1.5× and a numerics change the
strict gate would have to re-identify. MIG would let one A100 host 7 small fits, but the host has the same 12 vCPU
to feed them (T16 :3-4). Verdict: no A100 for learning either; put the units into label lanes and the probe above.
