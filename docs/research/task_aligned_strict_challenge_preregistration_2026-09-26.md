# Strict challenge of original 6102 against Apex: pre-registration and feasibility (draft)

Status: **DRAFT PRE-REGISTRATION. NOT ADMITTED. NOT AUTHORIZED. NOT EXECUTED.**

This document consolidates the strict-challenge design that was already fixed
prospectively in `research/task_aligned_challenger_20260924/protocol.md`
("Fixed scientific choices for the later strict challenge", source sha bound in
the bridge intents). It adds three things:

- a feasibility estimate made before any pilot data exists;
- an explicit kill criterion for the challenger line;
- an execution envelope.

It does **not** change any fixed margin, MDE, seed namespace, count bound or
behavioral band.

## 1. Question

Does the original nominated checkpoint **6102/global35613**
(`8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f`) beat the
operational incumbent **Apex/vector61** in the deployed task, under the shared
strict gate (`src/scripts/eval_stats.py` `strict_promotion_decision`)?

Prior evidence is thin:

- **No comparison against Apex.** The three-lineage screen compared each
  candidate with its own parent, not with Apex, and its package result was
  `INCONCLUSIVE_SCREEN`.
- **Selected after the screen.** 6102 was chosen after the screen as the only
  lineage passing every check, so a winner's-curse discount applies.
- **Continuation got worse.** Its continued child was `NEGATIVE_CONTINUATION_SCREEN`.
- **Bridge qualification is not a performance result.** The bridge
  (audit-only PASS, 2026-09-26) establishes serving and accounting readiness
  only.

## 2. Fixed design (unchanged from protocol.md)

| Element | Fixed value |
|---|---|
| Profile | `promotion-v2-watch-rect` (see below) |
| Candidate serving | Restricted raster31v3, Q scale 0.1 |
| Mixes | `frozen`, `scripted`, `mixed`, with balanced roster rotations |
| Namespaces (already materialized) | development 16, pilot 16, final 120 (ordered), serving 50 |
| D | 16 Apex-only development worlds per mix |
| P | 16 paired pilot worlds per mix |
| N | `max(40, max per-mix pilot-required count)` using `paired_delta_pilot_size` (per-mix alpha 0.05/3 one-sided, marginal 80% power) |
| MDE | 0.10 × positive development incumbent mean mass, per mix |
| Noninferiority margin | 0.05 × positive development incumbent scripted mean mass |
| Superiority | One-sided paired t, Holm over three mixes, family alpha 0.05; at least 2 of 3 must pass |
| Behavioral bands | Survival fraction within [dev incumbent mean − 0.02, mean + 1] per mix; food and boost descriptive |
| Serving | 1 Watch + 49 Play episodes under actual serving rules |
| Ceiling | 3D + 6P + 6N + 50 ≤ 914 episodes, ≤ 4,570,000 world frames |

Profile details: six snakes, mechanics v2, 1450 × 830 rectangle, H5000, and
`deployment.yaml` with initial food 300 and capacity 10400.

Forbidden after any data is seen:

- repeating or extending the pilot;
- relaxing the MDE or margins;
- choosing a prefix of the final bank;
- substituting a checkpoint (including the continued child);
- resampling worlds.

## 3. Feasibility estimate (planning only, not a decision input)

The fixed design stops if the pilot requires N > 120. Using the gate's own
sizing function, N depends on the ratio r = SD(paired delta) / incumbent mean
mass:

| r | 0.30 | 0.35 | 0.40 | 0.50 | 0.60 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Required N | 85 | 115 | 150 | 234 | 337 | 526 | 935 |

So **N ≤ 120 requires r ≲ 0.36**. Proxies from saved data:

- **H3000 continuation study** (32 worlds × 3 profiles; 6102 against its child
  and against scripted heroes):
  - Paired-world correlation is near zero (ρ ≈ −0.2 to 0.3), because
    trajectories diverge.
  - 6102's own per-world mass SD is 27–35.
  - r is 0.35–0.45 in solo and 0.57–0.67 in both opponent profiles.
  - Proxy N is 114–415, and above 120 in 8 of 9 proxy pairings.
- **H5000 bridge shakedown** (2 shared worlds × 3 mixes; interface evidence,
  n far too small to size anything):
  - Apex mass per cell ranged 1.3–109.2 and 6102 ranged 8.6–102.5.
  - Apex's per-world spread is plausibly as large as its mean (r ≈ 1), which
    implies N ≈ 900.

**Expectation: the pre-registered pilot will most likely return N > 120 and
stop as `STOP_INFEASIBLE_AT_FIXED_MDE`.** A pass would require much lower Apex
per-world variance than either proxy suggests. These estimates are not
authoritative. Only the admitted pilot decides feasibility.

## 4. Kill criterion (new; fixed before any pilot data)

The challenger line is **retired** on any of these outcomes:

1. The pilot requires N > 120 (`STOP_INFEASIBLE_AT_FIXED_MDE`).
2. Wall caps cannot fit the required N plus audit and handoff.
3. The final gate fails or is inconclusive: fewer than 2 of 3 Holm superiority
   tests pass, or scripted noninferiority fails, or a behavioral, serving,
   source or artifact gate fails.

"Retired" means:

- no MDE relaxation, larger bank, re-pilot, alternative checkpoint, continuation
  or tuning of 6102 or its siblings;
- task-aligned PQN work stops competing for the incumbent slot;
- effort returns to the redesign levers (mechanics v2 and throughput).

A reopening requires a new, separately justified hypothesis, not a rescue of
this one.

A pass produces only a strict receipt. Promotion is a separate, explicit release
action that preserves Apex rollback assets. Nothing is copied, defaulted or
deployed by the challenge.

## 5. Execution envelope

Stages, each gated so that the next cannot start unless the previous one passes:

1. **Development:** 48 Apex episodes.
2. **Pilot:** 96 paired episodes, then freeze N.
3. **Final:** 6N episodes (≤ 720).
4. **Serving:** 50 episodes.
5. **Independent saved-evidence audit.**

Time estimates come from the v4 shakedown, which ran 12 H5000 episodes in
334 s, about 28 s per episode:

| Stage group | Estimate |
|---|---|
| Development + pilot (144 episodes) | ≈ 70 min |
| Worst-case full chain (914 episodes) | ≈ 7–8 h |

Proposed caps:

| Stage | Cap |
|---|---|
| Development | 2,400 s |
| Pilot | 4,800 s |
| Final | 30,000 s |
| Serving | 3,600 s |
| Audit | 900 s |
| Handoff reserve (separate) | 120 s |

The final cap is only reserved if the pilot yields N ≤ 120. If it does not, the
run closes after the pilot.

Resources:

- CPU at 2 intraop / 1 interop threads; RSS ≤ 8 GiB; MPS ≤ 24 GiB; available
  RAM ≥ 9.6 GiB.
- Shared locks and the Mac supervisor.
- Zero optimizer updates and zero checkpoint saves.
- One create-only controller with no retry and a fresh prospective deadline.

Implementation should reuse the qualified bridge serving path and the existing
strict CLI (`tournament_eval.py --strict-promotion-request`, profile
`promotion-v2-watch-rect`). Keep the controller simple:

- one intent and one runner;
- an audit that consumes the same saved per-world records the decision used;
- no stacked recovery admissions.

## 6. Decision needed before admission

Two honest options, both consistent with the kill criterion:

- **A. Run development + pilot only (~70 min).** Get the protocol-compliant
  feasibility answer. Expected outcome: stop and retire.
- **B. Retire now on the feasibility evidence above,** without spending pilot
  compute. This is a judgment call, not a strict-gate result.

Either way, do not modify the fixed MDE to make the challenge feasible: that
would be post hoc tuning.
