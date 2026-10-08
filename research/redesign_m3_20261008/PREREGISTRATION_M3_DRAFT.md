# M3 pre-registration DRAFT (rev. 2): RL fine-tune of the M2b ego2s-b student

**Status: DRAFT for the owner (2026-10-08).** Rev. 2 folds in the independent review of
rev. 1 (NO-GO as written; every finding addressed in §11). Nothing here runs until the
owner ratifies it and approves any pod spend. At ratification the frozen parts (§7 rule
spec JSON + plan, §6 seeds and namespaces, §3 schedules) are committed with sha256 pinned by
test BEFORE the first training step. Development / Tier-1 only: nothing here promotes; a
Phase R GO still buys LH-1, the strict sequential gate and Mac serving qualification.

Inputs: doc `docs/research/redesign_scope_2026-10-07.md` §6–§10, §14–§18; M2b PASS
(`research/redesign_m2_20261008/results_m2b/ni_check/verdict.json`); the ratified
sequential Phase R amendment and design on main (`22ba10b`).

## 1. Question

Does reinforcement learning in the gate world, starting from the distilled ego2s-b student
and anchored to its teacher, produce a policy that, wrapped in the served v8 veto, beats
the served champion **I = frp3-s12 + v8** on Tier-1 H5000?

## 2. Starting point

`student_m2b_final.pth`, sha `36a92948ff82618b1944431675414301ef27f22bc15c7e12329bf75fc7cac48f`
(M2b NI PASS +52.4 vs frp3-s12 without veto; +11.1 with v8 on both, reported). No boost
reweighting (doc §18.3: it raises overall teacher-Q regret); the boost rate is an M3
diagnostic.

## 3. Algorithm (DQN family; doc §6). All schedules are in TRANSITIONS.

* **Learner:** synchronous vectorized Double DQN, dueling `Ego2sNet` (ego2s-b), **n-step 5**,
  Huber TD, Adam 1e-4, batch 512, grad-norm clip 10, target sync every 1000 updates, no PER
  in the first run.
* **Replay ratio 4:** one update of 512 per 128 new agent transitions (20M transitions =
  156k updates).
* **Discount (γ 0.995 via a value re-bootstrap):** γ 0.99 for the first **2M** transitions
  (the teacher's Q is a γ 0.99 value), then linear to **0.995 by 6M**; γ 0.997 only if
  FRP-v5-H wins. The reward's potential term uses the learner's current γ (policy-invariant
  shaping; the reward drifts slightly during the anneal, accepted and disclosed).
* **Anchor (kept), demonstrations anchor-only:** the M2b demonstrations have no reward or
  next state, so they take **no TD loss**. On them: `λ_anchor · (advantage-Huber ×10 + KL
  τ 0.05 to the teacher Q + DQfD margin to the v8 action)`; the raw-Q regression term is
  **dropped** (it would fight γ 0.995 forever). Demonstrations are 25% of every batch.
  `λ_anchor` = 1 until 6M transitions, then decays exponentially to a floor of 0.1 at 15M.
* **Memory (Mac, 64 GB, shared with the Phase R jobs):** agent replay 500k transitions
  (uint8 obs 12.4 KB → ~6.2 GB); demonstrations streamed from the compressed shards (1.9 GB
  on disk) by a background decompress thread into a rotating 300k-sample buffer (~3.7 GB),
  one shard swapped per 20k updates; total ≤ 12 GB.
* **Exploration:** ε-greedy over the resolved mask, ε 0.05 → 0.01 by 5M.
* **Reward:** the champion's contract (`GridBatchSim` reward v2: potential length term, death
  −3, kill credit 0.3 × victim length).
* **Episodes:** hero death is terminal (no bootstrap); the hero respawns into a new episode;
  the world resets every 5000 frames (truncation: bootstrap).
* **Tripwires:** HALT on NaN/inf or action collapse (one action > 90% of a 100k-decision
  window). FLAG only (reported, no halt): max |Q| above 5× the demonstrations' max |Q|
  (kills legitimately pay up to +0.3 × victim length), held-out v8 agreement on the
  demonstration hold-out below 0.6.

## 4. Opponents

* **Training worlds:** 75% scripted (`scripted-anchor/v1`, decision-identical fast policy),
  25% frozen-pool (the strict rosters' checkpoint pool: champion_a5 and the three older pool
  members, vector61 batched forwards). frp3-s12 never trains against itself.
* Training world seeds: namespace `redesign-m3-train/v1` (disjoint from every NI, probe,
  distillation and Phase R namespace; tested at ratification).

## 5. Compute plan

Measured on the Mac (doc §14.4, §18.2; one thread per process, loaded host):

| Quantity | Value |
|---|---|
| Env, scripted worlds, ego2s-b, MPS acting (E = 256) | 12.1k hero transitions/s |
| Env, frozen-pool worlds, ego2s-b (E = 64) | 1.6k–2.7k/s (+31 µs acting) |
| **Env, training mix 75/25** (0.75 × 83 µs + 0.25 × 500 µs ≈ 187 µs per world-frame) | **≈ 5.3k transitions/s** (this is also G1 for the training mix: **not met**, 4× below 20k) |
| Learner, `Ego2sNet` DQN update on MPS, batch 512 | ≈ 20k samples/s ≈ 200 µs per 128 transitions at ratio 4 |

**Mac design: one synchronous process (doc §6), MPS for every forward and update.** Per 128
transitions: env ≈ 128 × 187 µs ≈ 24 ms + update ≈ 26 ms (incl. ~1–2 ms CPU batch
gathering) → **≈ 2.6k transitions/s**. M3-A (5M) ≈ **32 min**; M3-B (5 seeds × 20M =
100M) ≈ **10.7 h**. The second allowed CPU process runs the dev probes (§6) concurrently.
A two-process actor/learner split (shared-memory replay) would roughly double the rate; it
needs new code and its own review, so it is not assumed.

**Pod (owner's spend decision).** One 4090-class pod (vCPU count and price to be verified
with the runpod skill; unverified here): env actors in 20 processes ≈ 20 × 5.3k × 0.78
(fast EPYC 9655P vCPU ≈ 0.78× a Mac slot; slow hosts ~10× slower are refused by the
identity pre-check) ≈ 83k/s; the 4090 learner is unmeasured (at batch 2048, if ≥ 80k
samples/s → ≈ 20k transitions/s, learner-bound). That needs the multi-process actor code too.

* **G2 as defined in doc §8 is ≥ 50k transitions/s end-to-end AND the ≥ 4× rule.** The
  projection (≈ 20k/s) would fail the 50k clause; ≥ 4× the Mac (2.6k/s) would pass easily.
  **Owner decision:** keep G2 at 50k (likely no pod for M3-B), or amend G2 to "≥ 4× the
  measured Mac M3-A rate". Cost if a pod runs: 100M / 20k/s ≈ 1.4 h + 0.5 h smoke ≈ 2 h ×
  ~$0.7–1.2/h ≈ **$1.5–2.5**.
* The Mac↔pod float-tie finding (argmax ties resolving differently across CPU models)
  affects only cross-platform identity checks, not training validity.

**Recommendation:** M3-A on the Mac now (≈ 32 min, free); decide M3-B's platform after
M3-A's measured rate and the owner's G2 ruling.

## 6. Stages and the learning-slope criterion

* **M3-A (Mac smoke):** 1 seed, 5M transitions; dev probes at 0 / 2.5M / 5M on reserved
  distillation-namespace worlds (round index 52, 16 per mix, student + v8 and student
  alone). HALT only on a §3 tripwire; the probe trend is FLAGGED if the 5M student + v8 mean
  is below the 0M mean by more than 50 (≈ 2 SE on 48 worlds) and the owner decides.
* **M3-B:** 5 training seeds (0–4), 20M transitions each, checkpoints every 2.5M.
* **Learning slope (doc §10 M3 exit, part 1):** per seed s, the dev H5000 mass of student +
  v8 at checkpoints 0, 5, 10, 15, 20M on the same 48 round-52 worlds; per world, the
  mix-stratified per-checkpoint mean; **one slope per seed** = OLS slope of the per-checkpoint
  world-paired means on transitions; then a **one-sided 90% Student-t lower bound across the
  5 seed slopes (df 4) > 0** → proceed to Phase R. Otherwise stop and report.
* **Dev probe cost:** 5 checkpoints × 48 worlds × 5 seeds ≈ 1200 H5000 v8 episodes ≈ 4 h
  on one thread (+ ≈ 0.3 h for M3-A), run in the second CPU process alongside training.
* **Phase R candidates:** each seed's final (20M) checkpoint, fixed in advance.

## 7. Tier-1: sequential Phase R vs I (method `sequential-phase-r-obf-hk-v1`)

**Owner rulings needed first:** (i) the ratified amendment covers "future FRP-family Tier-1
screens"; applying it to a new architecture needs an explicit ruling; (ii) doc §10's
intermediate Tier-1 screen (pooled UB > 0) is replaced by this sequential Phase R, whose
KILL / NO_GO stops play that role.

* **Arms:** C = M3 student (seed s, 20M) + v8 (λ 8) vs I = frp3-s12 + v8. Reported-only:
  the M2b student + v8 control on seed 0's bank only (96 episodes, the RL contribution).
* **Worlds:** **one 32-world bank per training seed** (FRP-v3 practice: 160 distinct
  worlds), each bank played in all 3 mixes (strict rosters), from the fresh namespace
  `redesign-m3-phase-r/v1`; H10000 records carrying their exact H5000 prefixes.
* **Looks:** 11 / 22 / 32 worlds per (seed, mix), look-major schedule; the prefix identity
  controls run in look 0; create-only `plan.json` before any shard; a look receipt before
  any look-k+1 work; the independent `python -I research/sequential_phase_r/audit.py` must
  PASS.
* **Rule spec:** the FRP v4 spec of `research/sequential_phase_r/example_rules.py`
  (`sequential-phase-r-rule/v1`) with: N 32, 5 seeds, GO OBF one-sided 0.10 on the HK
  pooled lower bound (nominal 0.0050 / 0.0457 / 0.0860 at df 4); KILL Pocock repeated HK
  upper bound < +20 (spending 0.10; non-binding for GO); NO_GO opt-in **G = 50**; the
  ratified margins (protective clauses √(1−t) at z 1.645; other points (1−√t) at z 1.2816).
  Its JSON and sha256 are pinned in the ratified pre-registration (condition 1).
* **GO (full rule at the stopping look):** (1) pooled H5000 point ≥ +20 AND the OBF HK lower
  bound > 0; (2) scripted mean ≥ 0 AND scripted survival ≥ −0.03; (3) frozen / mixed
  survival ≥ −0.03; (4) ≥ 4 of 5 seeds positive; (5) prefix identity controls pass AND
  pooled MI10 point ≥ +20; (6) champion guard: C vs **champion_a5 + v8** (the pre-frp3
  served champion, FRP's reference) point > 0 and scripted survival ≥ −0.05 on the reference
  seed's 96 episodes; (7) gate feasibility (N_max ≤ cap). GO_UNGATEABLE / PARTIAL are
  final-only, with the fixed rule's precedence; INCOMPLETE / INVALID_ANALYSIS first.
* **Amendment conditions quoted (binding):** stops are final (a GO, KILL or NO_GO at an
  interim look ends the study; an override is a recorded protocol deviation; no resume,
  rerun or relabel of a look); a HALT ends the analysis and the owner decides; an incomplete
  look is never analysed (INCOMPLETE); the audit must PASS (UNCLOSED / HALTED / FAIL →
  INVALID_ANALYSIS); after an early stop the report states that point estimates are biased
  upward and descriptive, final-only labels are unresolved, and candidate selection used the
  stopping look's worlds.
* **OC applicability:** the design matches the simulated FRP v4 family (5 seeds, N 32, the
  same thresholds), so the committed OC table applies. The variance is comparable: per-world
  SD of (student + v8 − I) on the M2b NI worlds was 192–318 per mix vs the pool's SD* ≈ 290.
* **Engine and its prerequisites (code to write and review before ratification):**
  * **P1 — live ego2s + v8 policy:** a `tournament_eval` agent spec for ego2s checkpoints
    and a live v8 wrapper around the ego2s serving policy (the web serving policy is
    deliberately unwrapped). LH-1, the strict gate and serving qualification need it anyway.
  * **P2 — record identity with the student acting:** M1/M2b established observation
    identity with the teacher acting; before ratification, ≥ 3 worlds per mix of C + v8 must
    give identical records on the SIMD grid engine and on the live engine (same platform).
  * **Engine rule fixed in advance:** if P2 passes, Phase R runs on the SIMD grid engine;
    if any world differs, the WHOLE study runs on the live engine. No per-world substitution.
* **Platform:** Mac only. Pod Phase R needs D4 (the per-look gate in
  `research/pod_phaser`), which is deferred, so the CPU-pod option is not available until
  that reviewed change lands.
* **Compute:** per H10000 v8 world ≈ 25–30 s on one Mac thread. Fixed maximum: 960 (C, I) +
  96 (control) + 96 (champion guard) + 15 (prefix controls) ≈ 1170 episodes ≈ **9.2 h on one
  thread → ≈ 4.6 h on 2 Mac processes**; ≈ 2–2.5 h expected near the null with NO_GO. Doc §8's
  "≥ $35" was a pod Phase R of the vector61 line on the live engine; this Mac plan is free.

## 8. Decision rule

| Outcome | Next step |
|---|---|
| M3-A HALT (tripwire) | stop; diagnose (owner) |
| M3-A probe FLAG | owner decides whether to run M3-B |
| Learning-slope LB ≤ 0 | stop M3; report; no Phase R |
| Phase R **GO** | pre-declared candidate = the seed with the highest per-seed lower bound (FRP-v3 practice) → LH-1 → strict sequential gate (live engine; 2 Mac slots or the strict-RunPod amendment) → Mac serving qualification (25 Watch + 25 Play, parity, ≤ 8 ms for 12 snakes with v8) |
| **KILL** / **NO_GO** | stop the M3 line; report; owner decides between capacity / horizon changes and retiring |
| PARTIAL / GO_UNGATEABLE | owner decision |

## 9. Owner decisions at ratification

1. Apply the sequential Phase R amendment to this new architecture; replace doc §10's
   intermediate screen with it.
2. G2: keep 50k (no pod for M3-B) or amend to "≥ 4× the measured Mac M3-A rate" (pod ≈ $1.5–2.5,
   needs the multi-process actor code).
3. NO_GO opt-in (recommended) and G = 50.
4. M3-B platform after M3-A (Mac ≈ 10.7 h free vs pod).

## 10. Work before ratification (code, Mac-only)

The synchronous DQN trainer for ego2s-b (`train_m3.py`) with the §3 schedules and the
streamed demonstration buffer; P1 and P2 (§7); the rule-spec JSON + test pin; tests and an
independent review.

## 11. Review of rev. 1 (2026-10-08): findings and resolutions

| Finding | Severity | Resolution |
|---|---|---|
| Schedules in updates never reached in 20M transitions (γ never annealed, anchor never relaxed) | blocker | All schedules now in transitions (§3) |
| Live-replay check / post-GO path need a live ego2s + v8 policy that does not exist; precondition overstated; post-hoc per-world substitution | major | P1/P2 prerequisites; whole-study engine rule fixed in advance (§7) |
| G2 quietly redefined | major | Doc §8 G2 restated; amendment is an explicit owner decision (§5, §9) |
| Replay/demo memory and IPC undefined; demos have no TD targets; single-process rate is ~2.6k/s, not 5k | major | Anchor-only demos, streamed buffer, memory budget, single-process rates and run times (§3, §5) |
| Same world bank across seeds contradicts FRP practice and DL/HK independence | major | One bank per training seed (§7) |
| Amendment scope (new architecture), D4 (no pod Phase R), missing rule-spec JSON, conditions not quoted, variance comparison | major | Owner ruling listed; Mac-only; spec named and pinned at ratification; conditions quoted; SD comparison stated (§7) |
| Learning-slope OLS anti-conservative; unit undefined | major | Per-seed slopes, df 4 t bound, per-world unit (§6) |
| Dev-probe and Phase R extras not budgeted; $35 vs $0.5 | major | Budgeted (§6, §7); reconciled |
| |Q| > 50 halt mis-calibrated; agreement halt; weak non-collapse test | minor | Flags with scale-relative rule; probe flag at ~2 SE (§3, §6) |
| Raw-Q anchor fights γ 0.995; γ in the shaping; "champion" in clause 6; training namespace; skipped intermediate screen | minor | Raw-Q term dropped; shaping γ stated; champion_a5 + v8; namespace named; owner ruling (§3, §4, §7) |
