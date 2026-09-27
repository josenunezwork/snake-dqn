# Strict challenge pilot: original 6102 vs Apex, 2026-09-26

Status: **INVALID_STOP_AUDIT_FAILED**. The unaudited producer outcome is
`PILOT_STOP_INFEASIBLE_AT_FIXED_MDE`.

Decision: **the 6102 / task-aligned challenger line is retired** under the
pre-registered kill criterion in
[the pre-registration](../../research/task_aligned_strict_challenge_preregistration_2026-09-26.md).
That criterion retires the line when the pilot requires more than 120 worlds
*or* when anything fails. Both conditions occurred, so the decision does not
depend on the audit.

Apex/vector61 remains the incumbent. Nothing was promoted, and no final bank,
serving episode or retry was run.

## What ran

The only authorization is the user's reply "u pick", given when choosing
between A (run development and pilot only) and B (retire now).

The run used one create-only controller at source `56547b4`, branch
`task-aligned-subtest-report-boundary`. That controller was built, reviewed by
three independent lenses, fixed and verified GO by a workflow.

- **Development:** 48 Apex-only episodes on 16 worlds × three mixes (frozen,
  scripted, mixed) under `promotion-v2-watch-rect` at H5000. Natural exit 0,
  853 s.
- **Pilot:** 96 paired episodes (16 worlds × 3 mixes × candidate/incumbent).
  Natural exit 0.
- **Audit:** failed with an exit-1 FAIL, described below.
- **Totals:** 3,828 s elapsed. 720,000 completed frames, 2,315,909 model
  forwards, 1,000 checkpoint loads, 0 optimizer updates, 0 checkpoint saves.
  Peak child RSS was 0.41 GB (development). The deadline was met, with 6,876 s
  remaining at handoff.

Root: `snake-dqn-artifacts/ongoing-research-20260913/task-aligned-strict-challenge-20260926/pilot-v1`,
intent `54da9625…`.

## Result (unaudited)

| Mix | Apex dev mean mass | MDE (10%) | Paired Δ SD | Required N |
|---|---:|---:|---:|---:|
| frozen | 52.61 | 5.26 | 56.3 | 1,072 |
| scripted | 38.82 | 3.88 | 118.3 | 8,676 |
| mixed | 72.65 | 7.26 | 55.8 | 552 |

Required final worlds: **8,676**, against a pre-registered cap of 120.

The feasibility estimate made before the run predicted a stop. The actual
noise-to-mean ratio, 1–3, was even worse than the proxies suggested.

Descriptive pilot means (candidate − Apex; not a strict-gate test):

| Mix | Mean mass difference | Candidate wins |
|---|---:|---:|
| frozen | −30.2 | 5 of 16 worlds |
| scripted | −66.4 | 5 of 16 worlds |
| mixed | −34.9 | 5 of 16 worlds |

## Audit failure

The audit failed with `development: lifecycle_frames 0 != 240000 for a
completed stage`.

This is a defect in the audit rule. `lifecycle_frames` is charged only by the
bridge's live-parity probe, which the strict pilot does not install. The saved
physical reports show exactly 48/96 episodes, 240,000/480,000 completed
frames, and `lifecycle_frames` 0.

A reviewer had flagged this as a NIT. The integration pass then tightened the
rule incorrectly. The failure is preserved and is not relabeled.

A separately authorized audit-only correction on the saved evidence could
upgrade the N = 8,676 result to audited. It would not change the retirement
decision.

## Implication

At the deployed profile, the per-world outcome noise is far larger than a 10%
effect. A strict comparison at this MDE would need thousands of worlds per mix.

This applies to any future challenger at similar variance, not just 6102.
Future strict gates should plan on either:

- much higher simulation throughput (the vectorized-sim direction in the
  redesign blueprint); or
- a variance-reducing evaluation design, pre-registered before data rather
  than tuned after it.
