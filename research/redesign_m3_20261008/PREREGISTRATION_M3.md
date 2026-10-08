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
  engines). The review confirmed the cause: each differing world differs only by +1 boost
  frame on the trapped death frame (the resolved mask falls back to the legal mask, boost
  included; the live snake falls back to normal moves).
* **P2 is repeated on the real candidates** before Phase R's first shard: `--arm v8`, 3
  worlds per mix at H5000 plus 1 world per mix at H10000, for every seed's 20M checkpoint.
  Any difference triggers the fixed engine rule (the whole study on the live engine; the
  platform choice returns to the owner first).
* **Rule spec + plan pinned (amendment condition 1):** `rule_spec.py` (FRP-v4 sequential
  form; seeds 0–4; NO_GO G = 50; cap 600; N 32); canonical JSON in `rule_spec_m3.json`;
  pinned by `tests/test_m3_rule_spec.py`: rule sha256
  `fd42a5d5ba898e41c2f4b251ac7833a461ed7773e5c91290f4d5015c00e04267`, plan sha256
  `5d1ee49d0ce8ce0f764f61909ad1803b542e825c70a08c01c12b83636651304e`, looks 11 / 22 / 32,
  GO nominal 0.0050 / 0.0457 / 0.0860, KILL nominal 0.0464 / 0.0508 / 0.0528. The rule's
  hero is a label; the candidate checkpoints' sha256 are written into `plan.json` before the
  first Phase R shard.
* **World namespaces** (`tests/test_m3_namespaces.py`): training `redesign-m3-train/v1`
  key `seed<s>`; Phase R banks `redesign-m3-phase-r/v1` key `seed<s>` (32 worlds each);
  disjoint from every NI, probe, distillation, P2 and M1 world used so far.

## 4. Measured rates (Mac, one thread + MPS; loaded host)

`train_m3.py` smoke (120k transitions, 64 worlds): **≈ 1.8k transitions/s** end-to-end
(env ≈ 4.6k/s; learner ≈ 42 ms per update of 512). M3-A (5M) ≈ **46 min**; M3-B (5 × 20M)
≈ **15.4 h** on the Mac (≈ 3.1 h per seed). Dev probes run in the second CPU process.

## 5. M3-A (Mac smoke; runs after ratification)

Seed 0, 5M transitions; checkpoint at 2.5M (`ckpt_2500k.pth`), the 5M model is `final.pth`.
Dev probes (16 worlds per mix,
distillation-namespace round index 52, grid engine, student + v8 and student alone) at 0
(= the M2b student), 2.5M, 5M. HALT only on a §2 tripwire; FLAG if the 5M student + v8 mean
is below the 0M mean by more than 50 (≈ 1.4–2.3 SE); report the rate, the probes, the
learning-curve diagnostics (TD, anchor loss, hold-out agreement, greedy boost rate, deaths
per 1k decisions).

## 6. M3-B and the learning-slope gate

5 seeds (0–4) × 20M transitions, checkpoints every 2.5M, the 20M model is `final.pth`.
Seed 0 restarts (the trainer saves weights only, no optimizer / replay / RNG state). Probes (student + v8) at 0, 5, 10, 15,
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
grid (P2 passed; repeated on the candidates, §3), Mac only (D4 deferred). Forward modes are
pinned and asserted by the Phase R harness: `vector61_forward="rowwise"` and
`ego2s_forward="rowwise"` (the modes P2 verified). ≈ 1180 episodes ≈ 4.6 h max on 2 Mac
processes.

## 8. Pinned code (commit `a6244da` + this file's commit)

`train_m3.py` sha256 prefix `aef08446e4c77b12`; `rule_spec.py` `04a2d7f69ced2756`;
`src/simd_env/ego2s_policy.py` `247686a44008a794`; `web/backend/ego2s_policy.py`
`591473721bde0ffc`; `src/simd_env/eval_engine.py` `48b46a2cd22e193b`;
`src/scripts/tournament_eval.py` `ce43be867cbe2357`; `student_record_parity.py`
`46f842f048d0c417`; `research/redesign_m2_20261008/dev_probe.py` `0ac3b9673e842a81`. Any change to these before the M3-B / Phase R step it governs is a
recorded deviation.

## 9. Owner decisions and ratification

Owner decisions (2026-10-08, relayed by the research loop):
(a) the ratified sequential Phase R applies to this architecture and replaces doc §10's
intermediate Tier-1 screen; (b) the pod gate is "≥ 4× the measured Mac M3-A rate
end-to-end" (the per-step 4× rule); pod spend still needs the user's approval when
proposed; (c) NO_GO opt-in at G = 50 confirmed; (d) the 5-seed platform is decided after
M3-A. Boosting is left to RL; the boost rate is reported.

**Independent review (pre-ratification, 2026-10-08):** GO-with-fixes; `train_m3.py`
verified correct (every one of 19,936 replay entries of a logged smoke recomputed exactly:
full 5-step windows, terminal flushes, truncation bootstraps; Double-DQN target with the
next state's mask; schedules; anchor = M2b loss minus raw-Q; demo stream; tripwires);
rates reproduced (1.73k/s). Its fixes are folded in above: P2 repeated on the real
candidates incl. H10000 (§3), forward modes pinned (§7), full rule / plan sha, rule JSON,
bank key, candidate hashes into `plan.json`, namespace disjointness test (§3), seed 0
restarts and `final.pth` naming (§5, §6), the pinned-file list extended (§8). Known and
accepted: the action-collapse check costs ≈ 5% (M3-A ≈ 48 min); demo prefetch makes runs
non-bit-reproducible; training never sees the "mixed" mix.

**RATIFIED 2026-10-08 by the research loop owner** (owner decisions (a)–(d) above), effective
at this file's commit. M3-A may start; M3-B's platform and any pod spend are decided after
M3-A; Phase R needs M3-B's slope gate and the candidate P2 re-run.

**Platform decision (d), recorded 2026-10-08 after M3-A** (relayed by the research loop as
the user's in-chat approval). M3-B uses 5 seeds (0–4) × 20M transitions on ONE RTX 4090
pod, secure cloud (≈ $0.74/h), run by `gpu_run.py` / `gpu_agent.py` at a clean commit;
everything else in §2 and §6 is unchanged. It uploads the committed tree, the sha-pinned
M2b student (`36a92948…c48f`), the pinned pool checkpoints and the M2b demonstration
rounds 10–12 (≈ 2.2 GB, sha-checked part by part). Money safety: a hard spend cap of $8
(stop when the account balance has fallen by ≥ $7.50 since launch); a 7 h pod lifetime,
with jobs killed 15 min before it; pod self-delete at the lifetime and a dead-man switch
(20 min without runner contact); a detached local watchdog; delete-and-verify (`GET /pods`
shows none of ours) on every exit; the shared ledger reservation (project cap). **Gate G2**
(the per-step 4× rule): after 30 min of training the 5 seeds' combined rate must be
≥ 5.6k transitions/s (≥ 4 × the M3-A rate of 1.39k/s), and the projected finish plus
45 min must fit in the remaining lifetime. Otherwise the pod is deleted and M3-B falls
back to the Mac (≈ 20 h, 2 processes), with the same seeds, budget and slope gate. The
learning-slope gate (§6) is evaluated on the Mac (`dev_probe`, round index 52, veto v8)
whatever the platform. The training device (CUDA, not MPS) is not part of the pinned
algorithm; rates and any CUDA-vs-MPS differences are reported. The launch itself
(`gpu_run.py launch --confirm`: spend plus upload) runs only on the user's direct
confirmation.

**M3-B launch attempt 1 (2026-10-08 05:25 PT, commit `601571d`): aborted safely, no pod,
$0.** Two runner bugs, both fixed before the relaunch (no change to the pinned algorithm,
seeds, caps or gates): (1) the local watchdog never ran, because a literal newline in its
generated source was a SyntaxError; the source is now a raw string that is byte-compiled
before launch, and the runner refuses to reserve or create anything unless the watchdog
process is alive and has logged "armed". (2) `POST /pods` failed and RunPod's error body was
not recorded; rp.py's output is now logged (key-safe) per attempt. At the time, RTX 4090
secure stock was "Low" and listed only in EU-CZ-1 and EUR-IS-2, with no data center pinned
in the request. A definite HTTP refusal (the pod was not created) is now retried up to 6
times: any data center first, then each stocked data center, checking for a pod of the
same name before each retry. An unknown outcome is never re-POSTed. Host minimums stay at
≥ 72 GB RAM and ≥ 10 vCPU: the cheapest secure 4090 offers list 12 vCPU and 83–100 GB, so
the minimums do not exclude stock, and 5 trainers need about 62–65 GB (review estimate),
so the earlier ≥ 48 GB figure would be unsafe for 5 seeds.
After re-review: only 4xx refusals, or a RunPod 5xx whose body says no instances are
available, are retried. Proxy 5xx responses such as 502 or 524 are unknown outcomes and are
never re-POSTed. Each launch's watchdog deletes only its own pod (exact name). A launch that
ends with its pod verified deleted, or aborts before creating one, stops its watchdog with a
stop file plus SIGTERM.

**M3-B launch attempt 2 (r2, commit `7add8c4`): aborted correctly, $0.06.** A secure 4090
pod came up in EU-CZ-1 (125 GB, 256 CPUs visible). The runner refused to train because the
pod's own RunPod key got HTTP 403 on its self-delete probe, so pod self-delete is not
available. The pod was deleted and `GET /pods` was empty. **User decision (2026-10-08, in
chat): "Run with Mac-side stop only".** The runner gets an explicit opt-in,
`--accept-no-self-delete "<note>"`, which records the note and the probe result in
`state.json` and the progress log. Without it, the run still refuses. What stops spend
without pod self-delete: the Mac runner (balance cap $8, 7 h lifetime); the detached Mac
watchdog (deletes the pod by exact name at lifetime + 5 min); the Mac stays on AC; the pod
still SIGKILLs every job's process group at the deadline and on the 20-min dead-man, so the
GPU idles but the pod is not deleted. Worst case if the Mac side fails entirely: the pod
idles until the watchdog fires (≈ 7.1 h × $0.74 ≈ $5.3, inside the cap). Seeds, gates and
the algorithm are unchanged.

**M3-B launch attempt 3 (r3, commit `cc85ec2`): no pod, $0.** The create got HTTP 500
"There are no instances currently available". The runner's no-capacity pattern did not
match this wording, so it did not retry. The pattern now matches it, case-insensitively.
**Capacity handling (within the approved $8 cap, secure cloud, same safety):** the runner
polls. Every ~5 min, for up to 3 h, it makes one complete attempt with its own pod name,
watchdog, ledger run and delete-and-verify, as long as every create try is a definite
no-capacity refusal. Nothing is reserved between attempts, and attempts without a pod
book $0. Results go to `<run-dir>/attemptNN/`; `<run-dir>/CURRENT_ATTEMPT` names the live
one. **GPU fallbacks, in order, all secure:** RTX 4090, RTX 3090 Ti, RTX 3090, RTX A6000,
A40, RTX A5000, L4. These are whole cards with ≥ 24 GB VRAM, a secure list price ≤ $0.80/h
(checked read-only on 2026-10-08), and an architecture that torch 2.5.1 / CUDA 12.4 in the
pinned image supports. Blackwell cards and MIG slices are excluded. Each try names exactly
one GPU type, and rp.py refuses any try whose secure estimate is above $0.80/h. Host
minimums (≥ 72 GB RAM, ≥ 10 vCPU) and the G2 gate (≥ 5.6k transitions/s combined at
30 min) are unchanged, so a slower GPU simply fails G2 and M3-B falls back to the Mac. The
GPU used is recorded in `state.json` and in the M3-B report.
After re-review: the GPU fallbacks and the stock wait go beyond the recorded approval
("ONE RTX 4090"), so both are explicit opt-ins that need the user's own approval. The
default is the 4090 only and a single attempt. `--gpu-fallbacks` allows the fallback list;
`--stock-wait-hours H` (at most 3) polls for capacity. Whichever is used is recorded in
`state.json` (`gpus_allowed`) and in this section when the run is reported.

**User approval (2026-10-08, directly in chat, relayed by the research loop): "Yes,
fallbacks + 3 h wait".** Secure fallback GPUs at ≤ $0.80/h, with 5-min stock polling for up
to 3 h, the same $8 cap and the same safety. Launch r4 runs from commit `0f6a12a` with
`--accept-no-self-delete … --gpu-fallbacks --stock-wait-hours 3`. Discrepancy, noted for
the record: the relayed approval names the 3090, A6000, A40, A5000 and L4. The runner's
list also includes the RTX 3090 Ti (secure list price $0.46/h, same class and caps). If a
3090 Ti is the GPU allocated, that is reported explicitly.

**Phase R harness disclosures (2026-10-08, before the study plan is written).**
`phase_r.py` is a fork of FRP-v5-S2's unit runner on the generic `research/sequential_phase_r`
package. It was smoke-tested on the out-of-namespace bank `redesign-m3-phase-r-smoke/v1`
(N = 4) and independently reviewed. Deviations from §7 as written:
(i) prefix controls: the candidate and the incumbent on every seed, plus champion_a5 + v8 on
seed 0. That is 33 control episodes, not 30; with them the study holds at most 1,185
episodes (≈ 1,180 in §7). (ii) Compute: up to 3 Mac processes, one per shared CPU slot
lock under the compute policy, instead of the 2 in §7's estimate. Results do not depend
on the process count: each unit is a deterministic, single-thread episode batch.
(iii) P2 at H10000 uses the `p2_h10000.py` wrapper, so the pinned P2 tool is unchanged.
Records carry each world's bank index, its roster member sha256s (rows always built from
the full bank) and the hero checkpoint path + sha256. A stdlib-only `phase_r_verify.py`
(run with `python -I`) checks the shard start markers and record bindings. It runs next to
the package's `audit.py`, and both must PASS before the decision is reported.

**M3 Phase R OUTCOME (2026-10-08): NO_GO_EARLY at look 1 (n = 22 worlds per seed × mix).
`phase_r_verify` PASS, `audit.py` PASS (both recompute the record keys, the deltas and the
decision; 33/33 prefix controls identical).** Plan `5d1ee49d…` was written at commit
`aeffbdd`.
* Look 0 (n = 11): CONTINUE. Primary HK (candidate + v8 − frp3-s12 + v8, H5000 mass
  integral) was +25.7, SE 23.1.
* Look 1 (n = 22): HK +11.4, SE 13.7. Seed effects: +24.9, +34.4, +14.2, −34.3, +16.9.
  The NO_GO upper bound +40.5 is below G = 50, so the study stops (KILL needs UB < +20,
  so it is not a KILL; GO needs LB > 0, and the LB is −19.0).
* Stops are final. Under §7 the point estimates below are biased by the early stop and
  are descriptive only; final-only labels are not resolved.
* Descriptive: c4 4/5 positive seeds; c5 mi10 HK −18.0 (SE 36.6); frozen survival −0.057;
  guard (vs champion_a5 + v8, seed 0) +111.9; control arm M2b + v8 vs the incumbent on seed
  0: stratified +14.2 (the candidate on the same bank: +24.9).
* Record: `results/phase_r/`.
