# Colab CLI pricing notes — 2026-10-18

## Purpose and scope

This note records the first live CPU rate-calibration evidence for the Colab
CLI supervisor. It is for operators setting conservative job budgets. It does
not qualify a training result, prove the remaining acceptance gates, or change
an approved spending cap.

## Observed CPU calibration

At 19:41 PT on 2026-10-18, the first live CPU acceptance attempt recorded an
observed CPU rate of **0.08 units per hour** and passed its rate-calibration
gate. The evidence directory is:

```text
/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance/acceptance-1792377693
```

The same attempt recorded `stop_verified: true`, and its final account query
found no assignments left. These facts establish one successful rate
calibration and cleanup observation. They do not establish a full acceptance
pass because the job stalled while delivering the former whole-commit source
archive before runtime readiness.

## Pricing decision

Keep the CPU `rate_units_per_hour` ceiling at **0.5 units per hour**. The
observed 0.08 units/hour is below that ceiling, but one calibration sample is
not a sufficient basis for lowering the rate used for admission, reserve
calculation, and the hard whole-session cap. The supervisor therefore records
both values:

| Value | Rate | Use |
| --- | ---: | --- |
| Observed CPU rate | 0.08 units/hour | Calibration and later drift detection for this allocation. |
| CPU priced ceiling | 0.5 units/hour | Admission, reserve, and session-cap calculation. |

The ceiling may be reconsidered only after reviewed evidence establishes a
stable lower rate across the relevant account and hardware conditions. Until
then, use 0.5 units/hour in manifests and acceptance planning, including
setup, export, idle time, and cleanup.

## Source-delivery implication

The stalled archive was approximately 166 MiB (171 MB of tracked source),
including large research evidence blobs. This was a transport failure mode,
not evidence that the CPU rate was lower or higher than the priced ceiling.
The new source plan fetches the pinned public GitHub commit on the runtime and
uses a <=32 MiB sparse fallback only when GitHub is unreachable. Refer to
[`colab-cli.md`](colab-cli.md) for the identity checks, fallback allowlist, and
upload bounds.
