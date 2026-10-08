# M3 pre-registration: RL fine-tune of the M2b ego2s-b student

**Status: FINAL, pending ratification (§9).** Supersedes `PREREGISTRATION_M3_DRAFT.md`
(rev. 2 + re-review fixes), which keeps the full rationale and the two review rounds; this
file is the binding text. Pinned code is identified by commit in §8. Development / Tier-1
only: nothing here promotes; a Phase R GO buys LH-1, the strict sequential gate and Mac
serving qualification.

## 1. Question and arms

Does RL in the gate world, from the M2b student and anchored to its teacher, give a policy
that wrapped in v8 beats **I = frp3-s12 + v8** on Tier-1 H5000? Start:
`student_m2b_final.pth` sha `36a92948…c48f`. Boosting is left to RL (doc §18.3); the greedy
boost rate is an M3 diagnostic.

## 2. Algorithm (frozen; `train_m3.py`)

Exactly the draft's §3–§4 as implemented in `research/redesign_m3_20261008/train_m3.py`:
synchronous single-process Double DQN, dueling `Ego2sNet` (ego2s-b), n-step 5, Huber TD,
Adam 1e-4, batch 512 = 384 agent + 128 demonstration rows, grad clip 10, target sync 1000
updates, one update per 128 agent transitions, 500k agent replay; γ 0.99 to 2M transitions
then linear to 0.995 at 6M (the sim's shaping γ follows the learner's γ per 5000-frame
segment); anchor on demonstrations only (advantage-Huber ×10 + KL τ 0.05 + DQfD margin;
no raw-Q term), λ 1 to 8M then exponential to 0.1 at 15M; demonstrations = M2b rounds
10–12 minus hold-out worlds, 300k rotating buffer, one shard per 1000 updates; ε 0.05 →
0.01 by 5M over the resolved mask; worlds 75% scripted / 25% frozen-pool (strict rosters),
E = 64, seeds `redesign-m3-train/v1` key `seed<s>`; HALT on NaN/inf or action collapse
(> 90% of 100k decisions), FLAG |Q| > 5× demo max and hold-out v8 agreement < 0.6.

## 3. Prerequisites (done)

* **P1** live student + v8 player: `tournament_eval._attach_agent` serves an ego2s hero via
  `Ego2sServingPolicy(forward="rowwise")`; the vetoes accept ego2s heroes (`8e4799a`).
* **P2** grid-vs-live record parity with the student acting (`student_record_parity.py`,
  `results/p2_v8.json`, `results/p2_none.json`): **student + v8: 9/9 worlds identical**
  (3 per mix, H5000) → per the fixed engine rule, Phase R runs on the SIMD grid engine.
  Reported: the student WITHOUT veto differs on 6/9 worlds (frozen / mixed, boost-frame
  counts): the SIMD `Ego2sSimdPolicy` acts on the resolved mask (as pre-registered for
  M2/M2b) while the live AISnake uses its advisory mask with the normal-only fallback. No
  Phase R arm uses that path (both C and I carry v8, which uses the advisory mask in both
  engines).
* **Rule spec + plan pinned:** `rule_spec.py` (FRP-v4 sequential form; seeds 0–4; NO_GO
  G = 50; cap 600; N 32) pinned by `tests/test_m3_rule_spec.py`: rule sha256
  `fd42a5d5…4267`, plan sha256 `5d1ee49d…304e`, looks 11 / 22 / 32, GO nominal
  0.0050 / 0.0457 / 0.0860, KILL nominal 0.0464 / 0.0508 / 0.0528.

## 4. Measured rates (Mac, one thread + MPS; loaded host)

`train_m3.py` smoke (120k transitions, 64 worlds): **≈ 1.8k transitions/s** end-to-end
(env ≈ 4.6k/s; learner ≈ 42 ms per update of 512). M3-A (5M) ≈ **46 min**; M3-B (5 × 20M)
≈ **15.4 h** on the Mac (≈ 3.1 h per seed). Dev probes run in the second CPU process.

## 5. M3-A (Mac smoke; runs after ratification)

Seed 0, 5M transitions, checkpoints at 2.5M and 5M. Dev probes (16 worlds per mix,
distillation-namespace round index 52, grid engine, student + v8 and student alone) at 0
(= the M2b student), 2.5M, 5M. HALT only on a §2 tripwire; FLAG if the 5M student + v8 mean
is below the 0M mean by more than 50 (≈ 1.4–2.3 SE); report the rate, the probes, the
learning-curve diagnostics (TD, anchor loss, hold-out agreement, greedy boost rate, deaths
per 1k decisions).

## 6. M3-B and the learning-slope gate

5 seeds (0–4) × 20M transitions, checkpoints every 2.5M (seed 0 restarts unless M3-A's
config is identical, in which case it continues). Probes (student + v8) at 0, 5, 10, 15,
20M on the same 48 round-52 worlds; per seed the OLS slope of the world-paired
mix-stratified means on transitions; **proceed to Phase R iff the one-sided 90% t lower
bound across the 5 seed slopes (df 4) > 0**. Candidates: each seed's 20M checkpoint.

## 7. Tier-1: sequential Phase R (method `sequential-phase-r-obf-hk-v1`)

As draft rev. 2 §7 with the pinned rule spec: C = M3 seed s + v8 vs I = frp3-s12 + v8;
one 32-world bank per seed from `redesign-m3-phase-r/v1`, all 3 mixes; looks 11/22/32;
prefix controls in look 0; GO (OBF on the HK LB + clauses 2–7, guard vs champion_a5 + v8
on seed 0's bank), KILL (Pocock HK UB < +20), NO_GO (UB < 50); create-only plan before any
shard, look receipts, look gate, `python -I` audit must PASS; stops are final; disclosures
after an early stop; control arm M2b student + v8 on seed 0's bank (reported). Engine: SIMD
grid (P2 passed), Mac only (D4 deferred). ≈ 1180 episodes ≈ 4.6 h max on 2 Mac processes.

## 8. Pinned code (commit `a6244da` + this file's commit)

`train_m3.py` sha256 prefix `aef08446e4c77b12`; `rule_spec.py` `04a2d7f69ced2756`;
`src/simd_env/ego2s_policy.py` `247686a44008a794`; `web/backend/ego2s_policy.py`
`591473721bde0ffc`. Any change to these before the M3-B / Phase R step it governs is a
recorded deviation.

## 9. Owner decisions and ratification

Owner decisions (2026-10-08, relayed by the research loop):
(a) the ratified sequential Phase R applies to this architecture and replaces doc §10's
intermediate Tier-1 screen; (b) the pod gate is "≥ 4× the measured Mac M3-A rate
end-to-end" (the per-step 4× rule); pod spend still needs the user's approval when
proposed; (c) NO_GO opt-in at G = 50 confirmed; (d) the 5-seed platform is decided after
M3-A. Boosting is left to RL; the boost rate is reported.

**Ratification:** pending the independent review of this file and the pinned code; recorded
below when done.
