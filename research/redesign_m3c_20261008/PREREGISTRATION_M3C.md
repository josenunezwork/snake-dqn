# M3-C pre-registration

**Status: FINAL, ratified in §11.** Supersedes `PREREGISTRATION_M3C_DRAFT.md` (kept for the
record). Design and evidence: `docs/research/m3c_design_2026-10-08.md`.
Development / Tier-1 only: nothing here promotes; a GO buys LH-1, the strict sequential gate
and Mac serving qualification.

## 1. Question

Can an RL fine-tune of the M2b ego2s-b student (`student_m2b_final.pth`, sha `36a92948…c48f`)
produce a policy that **without any veto** beats **I = frp3-s12 + v8** on Tier-1 H5000
(primary), and, wrapped in v8, beats I (secondary)? Which of the M3-C changes (metric-aligned
reward, short anchor, eval-matched worlds) are responsible, with long RL in every arm?

## 2. Algorithm

`research/redesign_m3c_20261008/train_m3c.py` at the pinned commit. Base = M3's algorithm
exactly (PREREGISTRATION_M3.md §2; default flags reproduce `train_m3.py` bit for bit,
tested), run with `--actors 2` (learner + 2 actor processes, actor weights refreshed every 25
updates, bounded queues of 8 frames, replay ratio 4 enforced by the learner, file-backed
observation rings under `--ring-dir`) and **20M transitions**, checkpoints every 2.5M,
`final.pth` = 20M; a HALTed run writes `aborted.pth` instead and is ineligible. Short anchor =
`--anchor-hold 2000000 --anchor-floor-at 6000000` (λ 1 to 2M, exponential to 0.1 at 6M).
Training worlds: namespace `redesign-m3-train/v1`, key `seed<s>` (M3's), seeds 0–4 in every
arm. Tripwires as M3 (HALT on non-finite loss or action collapse; FLAG hold-out agreement
< 0.6). The M3 \|Q\| FLAG is kept but reported as uninformative (design §2.1); the held-out
calibration (`holdout_u_mean`) is reported.

## 3. Arms (5 seeds each)

| Arm | Flags beyond `--actors 2` |
|---|---|
| A | short anchor (v2 reward) |
| B | A + `--reward mass --mass-kappa 1.0` |
| D | B + `--frozen-frac 0.34 --mixed-frac 0.33 --segment-frames 10000` |
| L | `--reward mass --mass-kappa 1.0` (M3's long anchor) |
| E (conditional, §4) | B + `--anchor-boost-weight 10` |

Seed s of every arm starts from the same training-world stream and RNG seeds (seed-paired
contrasts; in multi-actor mode scheduling and policy lag still differ by arm, and D's world
settings change its stream). The run plan is `plans/m3c_pod_plan.json` (20 runs, seed-major queue). **Arm E is not run**
(§4 result).

## 4. Stage 0 (Mac, before any spend): boost screen B0 — done

M3-B finals (seeds 0–4) as is vs with the exact boost bias β = 0.035 (`boost_bias.py write`;
calibrated so the greedy boost rate on held-out states ≈ the teacher's), dev probes on
round-52 worlds, no veto and v8. **Amended before ratification (2026-10-08):** 16 worlds per
mix with M3-B's probe settings (batched forward, resolved mask), so the β = 0 side is the
existing M3-B probe files (`stage0_report.py`); the draft's 32 worlds per mix in Phase R
modes would have cost ~3 h of contended Mac slots for a screen already negative in the
design's mini-screen. Statistic: per seed, the paired
mix-stratified Δ; across the 5 seeds a one-sided 90% t lower bound (df 4). **Arm E is added
iff the lower bound is > 0 for BOTH no-veto and v8.**

**Result (`results/diag/stage0_report.json`):** no veto Δ = −20.9 (seed SE 11.2, LB −38.2;
per seed +2.4 / −22.1 / −27.7 / −58.7 / +1.5); v8 Δ = −0.5 (SE 8.7, LB −13.9). **Arm E is not
added**; boosting stays a reported diagnostic (greedy boost rate in every arm) and the mass
reward is the boosting treatment. The references the contrasts are read against (the M2b
student and the M3-B finals, 32 worlds per mix, Phase R's hero modes) are re-probed on the
Mac as descriptive data, not launch-blocking.

## 5. Platform

ONE GPU pod per `gpu_run_v2.py` (secure RTX 4090 preferred; secure fallbacks 3090 Ti / 3090 /
A6000 / A40 / A5000 / L4 at ≤ $0.95/h, opted in by the user), lifetime 10 h, spend cap $10,
Mac-side stop (`--accept-no-self-delete`, user decision 2026-10-08), host minimums 96 GB RAM
and 12 vCPU. G2 at 30 min: combined
rate ≥ 4 × `mac_rate` (the Mac's best measured end-to-end rate for this job at production
settings, pinned in the plan), the pod's whole wall time for the step (incl. upload and
setup) ≤ the Mac's / 4, and the queue's projected finish + 45 min inside the lifetime; else
delete and run M3-C on the Mac (same arms, seeds and rules; ≈ 38 h of the Mac's GPU at 2.94k/s).
`max_hourly` 0.95 (secure 4090 list $0.89; also caps the fallback GPUs, replacing M3-B's
$0.80), lifetime × max_hourly ≤ the cap. Concurrency (owner decision): start at
min_vcpu / 3.2 = 3 runs, then the adaptive fill adds one run per ≥ 5 min (cap 10) only while
the container memory headroom (minus the rings of runs younger than 10 min), VRAM, disk, CPU
(the cgroup quota if any) and GPU < 85% guards allow, the per-run rate fell ≤ 15% and the
combined rate rose ≥ 3% since the last fill; fills freeze 10 min before G2, which measures
the combined rate runner-side over that fixed-layout window. The training device / platform is not part of the
algorithm; rates and utilisation are reported.

## 6. Dev probes and selection (Mac)

`probe_watch.py` starts probes as checkpoints land: no-veto at 5M, 10M, 15M (16 worlds
per mix, round 52; the no-veto score is the training-progress metric), and at `final` 32
worlds per mix for no-veto and v8, always with Phase R's hero modes (`--ego2s-forward rowwise`;
without the veto `--ego2s-mask advisory`) (two watchers: `--schedule 5000k,10000k,15000k
--worlds 16 --vetoes none` and `--schedule final --worlds 32 --vetoes none,v8`, both
`--slots`); the incumbent I = frp3-s12
+ v8 once on the same 32 worlds per mix (`dev_probe.py --arm baseline --veto v8`; done in design: 434.9, 31 deaths in 96). Probes are
development information; **no early stopping** of training runs on probes.

Selection (`m3c_select.py`, at `final`, on the worlds every arm and I share):
* a seed is eligible iff its `summary.json` has `halt = null` and ≥ 20M transitions and it is
  fully probed; arms are compared on the seeds eligible in EVERY arm (≥ 4, else nothing is
  selected and no Phase R runs); a Phase R candidate arm needs all 5 seeds eligible;
* score(arm, veto) = mean over those seeds of the mix-stratified mean (per mix, the worlds
  every eligible file and I share);
* **primary arm** = argmax no-veto score (ties: order A, B, D, L, E);
  **enters the primary Phase R iff score_nv ≥ I_dev − 50 AND, in every mix, the
  world-paired dev survival difference (share of the arm's seeds alive at H5000 on a world
  minus I alive) has mean ≥ −0.03 − 1.28 SE** (a noise-aware form of the Phase R survival
  clauses; expected outcome today: NOT ENTERED);
* **secondary arm** = argmax v8 score; **enters iff score_v8 − I_dev ≥ +30 AND the same
  survival check**;
* reported (descriptive): B − A, D − B, B − L, E − B with seed-paired df-4 intervals,
  no-veto survival / deaths, greedy boost rate, held-out calibration, H10000 is in Phase R only.

## 7. Prerequisites for each Phase R

* **P2-NV** (primary): `p2_noveto.py --arm none` (3 worlds per mix, H5000) and
  `p2_noveto.py h10000` (1 world per mix, H10000) on every candidate seed's checkpoint: all identical (grid with `ego2s_mask="advisory"` == live). Any difference →
  the whole primary study on the live engine (platform decision returns to the owner).
* **P2-v8** (secondary): M3's `student_record_parity.py --arm v8` + `p2_h10000.py`, as M3 §3.
* Candidate sha256s into each `plan.json` before the first shard.

## 8. Tier-1: sequential Phase R (method `sequential-phase-r-obf-hk-v1`)

Rule specs `rule_spec_m3c.py` (FRP-v4 rule in sequential form, NO_GO G = 50 opted in, cap 600,
N = 32 per seed × mix, looks 11 / 22 / 32; pins in `tests/test_m3c_rule_spec.py`):

| Study | Hero | Baseline | Bank namespace |
|---|---|---|---|
| `redesign-m3c-phase-r-nv` (primary) | selected arm, no veto, `ego2s_mask="advisory"`, `ego2s_forward="rowwise"` | frp3-s12 + v8 | `redesign-m3c-phase-r-nv/v1` |
| `redesign-m3c-phase-r-v8` (secondary) | selected arm + v8 | frp3-s12 + v8 | `redesign-m3c-phase-r-v8/v1` |

Harness: `phase_r_m3c.py --study nv|v8` (fork of M3's harness; the candidate and the M2b
control arm play without a veto in the nv study; the incumbent and the champion always with
v8). A real plan refuses unless this file (as `PREREGISTRATION_M3C.md`) has a line starting
`**RATIFIED <date>` and the `--candidates` are the 5 `final.pth` of the arm the
`--selection` JSON entered for that study. Guard: the hero vs champion_a5 + v8 on seed 0's
bank; prefix controls as M3; create-only
plan / receipts, look gate, `python -I` audit must PASS; stops final. The two studies are
separate decisions about two deployment configurations (policy alone; policy + veto); each
has its own α as ratified for the method (no claim is made that the pair controls a joint
error rate). The primary runs first when both enter. Mac only, ≤ 3 slots.

Pins: primary rule sha256 `5103b413…84b6c0`, plan `70e4b4d7…2350f2`; secondary rule
`1f50018e…b97488`, plan `4a715656…a8211` (full values in `rule_spec_m3c.py` and the test).
**Harness smoke (2026-10-08, `results/phase_r_smoke_nv.json`):** the nv study on the
out-of-namespace bank `redesign-m3c-phase-r-smoke/v1` (N = 4, M3-B finals as stand-in
candidates): look 0 complete, 105 records (candidate and M2b control without a veto and with
the advisory mask; incumbent and champion with v8), prefix controls 33/33 identical, the
decision KILL_V4 (dev only, expected for a bare student vs I + v8), package audit PASS.

## 9. What each outcome means

* Primary GO_R → the redesign beats frp3-s12 + v8 without a veto at Tier 1: LH-1, strict gate,
  serving qualification of the student alone.
* Primary NOT ENTERED / KILL / NO_GO → the goal is not met; the arm contrasts say which lever
  moved the no-veto score; next round (M3-D) designed from them (candidates: longer RL,
  online veto-label distillation, κ sweep).
* Secondary GO_R → a with-veto improvement over I (a deployable fallback), not the goal.

## 10. Pinned code

The ratification commit (the commit of this file) pins every file of
`research/redesign_m3c_20261008/` (`train_m3c.py`, `gpu_run_v2.py`, `gpu_agent_v2.py`,
`probe_watch.py`, `m3c_select.py`, `rule_spec_m3c.py`, `phase_r_m3c.py`, `p2_noveto.py`,
`boost_bias.py`, `stage0_report.py`, `plans/m3c_pod_plan.json`), `src/simd_env/ego2s_policy.py`,
`src/simd_env/eval_engine.py` and `research/redesign_m2_20261008/dev_probe.py`; the pod runs
`git archive` of that commit (sha recorded in the run's `state.json`). Any later change to
these before the step it governs is a recorded deviation.

## 11. Ratification

Owner decisions (2026-10-08, relayed by the research loop): arms A / B / D / L at 20M as
designed, E only if §4 passes (it did not); `max_hourly` $0.95 incl. the fallbacks; the fill
beyond the guaranteed vCPUs only while the container memory / VRAM guards allow (adaptive, as
designed); entry gates as designed.

**User approval (2026-10-08, in chat, relayed by the research loop): "Yes, prep then pod
run"** — finish the $0 pre-launch steps (independent re-review of the round-2 fixes, Phase R
harness smoke + audit, stage 0, this file), then ONE secure GPU pod run (RTX 4090 preferred,
fallbacks ≤ $0.95/h), spend cap $10, lifetime 10 h, Mac-side stop
(`--accept-no-self-delete`, recorded), G2 gate (≥ 4× the Mac rate and wall clock),
per-checkpoint pulls + telemetry + Mac probes as checkpoints land.

**RATIFIED 2026-10-08 by the research loop owner**, effective at this file's commit.
