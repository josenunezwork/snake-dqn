# M3 pre-registration DRAFT: RL fine-tune of the M2b ego2s-b student

**Status: DRAFT for the owner (2026-10-08).** Nothing here runs until the owner ratifies it
and approves any pod spend. When ratified, the frozen parts (§7 rule spec, §6 seeds and
namespaces) are committed with a sha256 pinned by test BEFORE the first training step, as
for every pre-registration in this program. Development / Tier-1 only: nothing here
promotes. A Phase R GO still buys LH-1, the strict sequential gate and Mac serving
qualification, exactly as for the FRP family.

Inputs: doc `docs/research/redesign_scope_2026-10-07.md` §6–§10, §14–§18; M2b PASS
(`research/redesign_m2_20261008/results_m2b/ni_check/verdict.json`); the ratified
sequential Phase R amendment (`docs/research/governance_amendment_sequential_phase_r_2026-10-07.md`,
design `docs/research/sequential_phase_r_design_2026-10-07.md`, code
`src/evaluation/sequential_phase_r.py`, on main since `22ba10b`).

## 1. Question

Does reinforcement learning in the gate world, starting from the distilled ego2s-b
student and anchored to its teacher, produce a policy that (wrapped in the served v8 veto)
beats the served champion **I = frp3-s12 + v8** on Tier-1 H5000?

## 2. Starting point

* **Student:** `student_m2b_final.pth`, sha `36a92948ff82618b1944431675414301ef27f22bc15c7e12329bf75fc7cac48f`
  (M2b NI PASS: +52.4 vs frp3-s12 without veto; +11.1 with v8 on both, reported).
* **Optional boost-fixed variant (owner choice at ratification):** the same student
  fine-tuned 2 epochs with boost-label weighting (doc §18.3). Chosen only on held-out
  distillation metrics; never on any evaluation world.

## 3. Algorithm (DQN family; the doc's §6 recommendation)

* **Learner:** synchronous vectorized Double DQN, dueling `Ego2sNet` (ego2s-b, 1.6M params),
  **n-step 5**, Huber TD, Adam 1e-4, batch 512, grad-norm clip 10, target sync every 1000
  updates, no PER in the first run (added only if M3-A shows sample starvation).
* **Discount:** **γ 0.995** (the doc's horizon choice, matching the FRP-v5-H arm), reached
  through a **value re-bootstrap**: the teacher's Q is a γ 0.99 value, so the first 200k
  updates run at γ 0.99 with the policy held by the anchor (below), then γ anneals
  linearly to 0.995 over the next 400k updates. γ 0.997 only if FRP-v5-H wins.
* **Replay:** 1M agent transitions (uint8 obs ≈ 12.8 KB each: ~13 GB host RAM) + the M2b
  **teacher demonstration set** (4.16M labelled decisions, data rounds 10–12, already on disk)
  sampled at **25% of every batch** (DQfD style).
* **Anchor (kept):** on demonstration samples only, `λ_anchor(t) · (advantage-Huber ×10 + KL
  τ 0.05 to the teacher Q + DQfD margin to the v8 action)` (the M2b distillation loss);
  `λ_anchor` = 1.0 for the first 600k updates (re-bootstrap + γ anneal), then decays
  exponentially to a floor of 0.1 by 2M updates and stays there.
* **Replay ratio:** 4 sampled transitions per new agent transition (≈ 1 update of 512 per
  128 transitions).
* **Exploration:** ε-greedy over the resolved mask, ε 0.05 → 0.01 over 5M transitions (the
  student is already competent; no exploration from scratch).
* **Reward:** the champion's contract (`GridBatchSim` reward v2: potential-based length term
  at the learner's γ, death −3, kill credit 0.3 × victim length), so the training objective
  matches the frp3 line's.
* **Episodes:** hero death is terminal for the TD target (no bootstrap); the hero respawns
  into a new episode; world reset every 5000 frames is a truncation (bootstrap).
* **Tripwires (halt-and-flag):** NaN/inf, max |Q| > 50, action collapse (one action > 90% of
  a 100k-decision window), held-out v8 agreement on the demonstration hold-out < 0.6.

## 4. Opponents

* **Training mix:** 75% of worlds scripted (`scripted-anchor/v1`, the fast decision-identical
  policy) and 25% frozen-pool worlds (the strict rosters' checkpoint pool: champion and the
  three older pool members, vector61 batched forwards). Gate mixes stay evaluation-only.
  Rationale (doc §14.4, §18.2): frozen-pool rows cost 240–590 µs per world-frame against
  ~2 µs for scripted rows; FRP-v3 showed scripted opponents in training help.
* frp3-s12 is NOT an opponent in training (it is the incumbent being evaluated against).

## 5. Compute plan (measured numbers; Mac CPU rules as 2026-10-08)

Measured on the Mac (§18.2; one thread per process, loaded host):

| Quantity | Value |
|---|---|
| Actor, scripted worlds, ego2s-b, MPS acting (E = 256) | 12.1k hero transitions/s |
| Actor, frozen-pool worlds (E = 64), no acting forward | 1.6k–2.7k/s (+31 µs/world-frame acting) |
| Training mix 75/25 (per world-frame ≈ 0.75 × 83 µs + 0.25 × 500 µs) | **≈ 5.3k transitions/s per actor process** |
| Learner, `Ego2sNet` DQN update on MPS (batch 512) | ≈ 20k samples/s (M1 network probe) |
| Learner capacity at replay ratio 4 | ≈ 5k transitions/s |

**Mac (GPU free; CPU limited to ≤ 2 processes × 1 thread while FRP-v5-H then FRP-v5-S2
Phase R hold the slot locks, ~20 h):** one actor process + one learner process (MPS):
**≈ 5k transitions/s, balanced** (actor ≈ learner).

* M3-A smoke (1 seed × 5M transitions): **≈ 17 min**.
* M3-B (5 seeds × 20M transitions): **≈ 5.6 h** serially (one seed at a time; the GPU is
  shared), free.

**Pod (needs the owner's spend approval; the ≥ 4× per-step rule applies):** one 4090-class
pod with ≥ 24 vCPU.

* Actors: 20 processes. Per-vCPU speed depends on the host: a fast EPYC 9655P vCPU ran the
  gate world at ≈ 0.78× a Mac slot (0.0054 vs 0.0042 s per frame-world); slow hosts were
  ~10× slower and are refused by the identity pre-check. On a fast host: ≈ 20 × 5.3k × 0.78 ≈
  **80k transitions/s** of env.
* Learner: unmeasured on a 4090. At batch 2048 a 4090 is expected to do several times MPS;
  if it reaches ≥ 80k samples/s the run is learner-bound at **≈ 20k transitions/s ≈ 4× the
  Mac**, i.e. the 4× rule is marginal and **G2 (30-min smoke) decides**: proceed on the pod
  only if the measured end-to-end rate is ≥ 4× the Mac's measured M3-A rate.
* Cost (if G2 passes): M3-B 100M transitions / 20k/s ≈ 1.4 h + smoke 0.5 h ≈ 2 h ×
  ~$0.7–1.2/h (4090 secure, vCPUs included) ≈ **$1.5–2.5**. Verify current prices with the
  runpod skill before any request; the $50 budget is untouched otherwise.
* The Mac↔pod float-tie finding (argmax ties resolving differently across CPU models) affects
  only cross-platform identity checks, not training validity: training runs are not
  replayed bit-for-bit, and evaluation (§7) is platform-pinned per world.

**Recommendation:** run M3-A on the Mac now (free, 17 min of GPU + 2 CPU processes); decide
the pod only on the measured M3-A rate and G2.

## 6. Stages and checkpoints

* **M3-A (Mac smoke):** 1 seed (seed 0), 5M transitions; dev probe at 0 / 2.5M / 5M on
  reserved distillation-namespace worlds (round index 52, 16 per mix; student without veto
  and with v8). Continue to M3-B iff no tripwire fired and the 5M probe (student + v8)
  is ≥ the 0M probe − 15 (non-collapse; not an efficacy claim).
* **M3-B:** 5 training seeds (seeds 0–4; seed 0 continues from M3-A's checkpoint only if
  the run is bit-identical in config, otherwise restarts), 20M transitions each,
  checkpoints every 2.5M.
* **Learning-slope criterion (doc §10 M3 exit, part 1):** dev H5000 of student + v8 on the
  round-52 worlds at checkpoints 0, 5, 10, 15, 20M per seed; OLS slope of mass on
  transitions pooled over the 5 seeds with a per-seed intercept; **one-sided 90% lower bound
  > 0** → proceed to Phase R. Otherwise stop and report (no Phase R).
* **Candidates for Phase R:** each seed's **final (20M) checkpoint**, fixed in advance (no
  selection on the dev probes, which would bias Phase R).

## 7. Tier-1 evaluation: sequential Phase R vs I (method `sequential-phase-r-obf-hk-v1`)

* **Arms:** candidate C = M3 student (seed s, 20M) + v8 (λ 8, served variant; `Ego2sV8Policy`)
  vs incumbent I = frp3-s12 + v8. Reported-only control arm: the M2b student + v8 (the RL
  contribution).
* **Worlds:** fresh namespace `redesign-m3-phase-r/v1`, **N = 32 worlds per (seed, mix)**,
  3 mixes (frozen / scripted / mixed, strict rosters), the same world bank across seeds as
  the FRP family's bank discipline, H10000 records carrying their exact H5000 prefixes.
* **Looks:** 1/3, 2/3, 1 → 11 / 22 / 32 worlds per (seed, mix), look-major schedule, prefix
  identity controls in look 0, create-only plan and look receipts, look gate per shard,
  independent `python -I` audit must PASS.
* **GO (full rule, judged at the stopping look):**
  1. pooled H5000 point ≥ +20 AND HK pooled lower bound > 0 with **O'Brien-Fleming spending
     of one-sided 0.10** (N = 32, df 4: nominal 0.0050 / 0.0457 / 0.0860);
  2. scripted mean ≥ 0 AND scripted survival ≥ −0.03;
  3. frozen and mixed survival ≥ −0.03;
  4. ≥ 4 of 5 seeds positive;
  5. prefix identity controls pass AND pooled MI10 point ≥ +20;
  6. champion guard: C vs champion + v8 point > 0 and scripted survival ≥ −0.05 on the
     reference seed's 96 episodes;
  7. gate feasibility (N_max ≤ cap at the strict gate's design).
  Interim looks use the ratified margins (protective clauses at √(1−t), z 1.645; other points
  (1−√t), z 1.2816).
* **KILL:** Pocock-type repeated HK upper bound < +20 at any look (spending 0.10;
  non-binding for GO; followed by protocol).
* **NO_GO (gate futility):** **opt in, G = 50** (the ratified default): repeated upper bound
  < 50 → stop. Saves ~46–61% of evaluation compute near the null (design doc §9).
* **Final-only labels:** GO_UNGATEABLE / PARTIAL with the fixed rule's precedence;
  INCOMPLETE / INVALID_ANALYSIS first.
* **OC report:** this design matches the simulated FRP family (5 seeds, N = 32, same
  thresholds), so the committed OC table applies; any change re-runs `simulate.py` before
  ratification.
* **Engine:** the SIMD grid engine (`run_simd_eval(sim_engine="grid", vector61=True,
  hero_ego2s=..., hero_safety_veto="v8")`). The governance precondition for counting SIMD
  Tier-1 numbers of a new architecture (doc §9 step 1: serving harness + live identity) is
  met for ego2s-b (§17.1, §18.1). Pre-registered live-replay check: 2 worlds per mix of look 0
  replayed on the live engine through the web serving policy on the same platform; records
  must be identical (an argmax float-tie flip is reported and, if it occurs, the affected
  worlds' live records are the ones used).
* **Compute:** per H10000 world with v8 ≈ 25–30 s on one Mac thread (from the M2b v8 arms:
  10–13 s per H5000 world in 48-world batches). Fixed maximum: 5 × 3 × 32 × 2 arms ≈ 960
  episodes + controls (+15%) ≈ **9 h on one thread → ≈ 4.5 h on 2 Mac processes**; expected
  ≈ 2–2.5 h with NO_GO near the null. On a fast 32-vCPU CPU pod (≥ 4× the 2 Mac processes):
  ≈ 20–25 min, ≈ **$0.3–0.5**.

## 8. Decision rule (what each outcome buys)

| Outcome | Next step |
|---|---|
| M3-A tripwire or collapse | stop; diagnose (owner) |
| Learning slope LB ≤ 0 | stop M3; report; candidates do not go to Phase R |
| Phase R **GO** | pre-declared candidate = the seed with the highest per-seed lower bound (FRP-v3 practice) → LH-1 → strict sequential gate (live engine, 2 Mac slots or the strict-RunPod amendment) → Mac serving qualification (25 Watch + 25 Play, parity, ≤ 8 ms for 12 snakes) |
| **KILL** or **NO_GO** | stop the M3 line as specified; report; owner decides between capacity/horizon changes or retiring |
| PARTIAL / GO_UNGATEABLE | owner decision |

## 9. What the owner decides at ratification

1. Mac-only (≈ 5.6 h, free) vs pod for M3-B (G2 decides the 4× rule; ≈ $1.5–2.5).
2. Starting student: M2b final vs the boost-weighted variant (§18.3).
3. NO_GO opt-in (recommended) and G.
4. Phase R platform: Mac (≈ 4.5 h max on 2 processes, after FRP-v5 Phase R frees the
   slots) or a CPU pod (≈ $0.5).
