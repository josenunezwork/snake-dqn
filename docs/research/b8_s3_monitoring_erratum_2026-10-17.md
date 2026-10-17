# B8 release-monitoring erratum — 2026-10-17

This is a sibling erratum to the unchanged
[B8 release record](b8_s3_release_2026-10-17.md). It corrects the scope of its
death-rate threshold. It does not change serving defaults, release identity,
checkpoint pins, veto scope, or Play behavior.

## Watch death thresholds by driven roster size

The release record's 0.848 deaths-per-1,000 trigger belongs to **six-snake Watch**.
It is not a global threshold for all Watch population sizes. Each row below uses
the highlighted B8 hero's actual decisions and authoritative death events.

| Driven Watch snakes | Baseline deaths / 1k | Rollback when deaths / 1k exceed | Source |
|---:|---:|---:|---|
| 6 | 0.424 | 0.848 | T12 default-six Watch, 25 H5000 episodes |
| 8 | 0.496 | 0.992 | T13b all-B8 Watch8, 25 H5000 episodes |
| 12 | 0.768 | 1.536 | T13 all-B8 Watch12, 25 H5000 episodes |

All triggers are exactly twice the recorded baseline. Equality does not trigger;
strictly exceeding the matching row does. The six-snake baseline remains the T12
release baseline; T13b's fresh six-snake descriptive result does not silently
replace it. T13/T13b are roster measurements, not new serving qualifications.

[`scripts/monitor_b8_release.py`](../../scripts/monitor_b8_release.py) now groups
Watch hero decisions and deaths by recorded `roster_size` (all driven AI in Watch),
prints each group's baseline/trigger, and independently evaluates its rate. The
aggregate rate is informational only. Mixing traffic cannot dilute a failing
roster or apply the six-snake threshold to a twelve-snake population. Unregistered
roster sizes return `NOT_HEALTHY` for a missing death baseline, at any speed.

This erratum does not assert a production monitoring window is complete or healthy:
the study suppressed product-monitor telemetry. The original fallback/ego2s
exception triggers and seven-day/100,000-decision observation requirement remain.

## Latency remains separately qualified

The existing T12B default-speed latency baselines and 1.5× triggers are unchanged:
Watch6, Watch12, Play with five driven AI, and Play with eleven driven AI.
The new Watch8 **death** baseline does not manufacture a Watch8 latency baseline.
Default-speed eligible Watch8 latency samples therefore still report a missing
T12B-qualified update baseline and cannot produce an overall healthy verdict on
that basis. This is distinct from the roster-specific death rollback test.

## Roster decision and verification

The [T13b result](t13b_six_snake_roster_2026-10-17_results.md) applies the rule
committed before running: **PLAN_MIXED_PENDING_ROSTER_QUALIFICATION**. Serving defaults remain unchanged.
Any planned mixed Watch default requires its separate roster qualification,
including identity, opponent-only veto scope, fallbacks and latency; this job
does not perform or waive that qualification.

The [final review](../../research/t13b_six_snake_20261017/FINAL_REVIEW.md) records
raw evidence/interval verification and focused monitoring tests. The original
release record's bytes are preserved.
