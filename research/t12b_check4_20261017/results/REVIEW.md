# Independent T12B evidence review — 2026-10-17

Outcome: **SERVING_PASS for check 4; no remaining findings.** A read-only reviewer
independently inspected the frozen producer, raw results, bindings, prior evidence,
and validation logs. No gameplay was repeated during review.

| Case | Recomputed update p95 (ms) | Server ceiling | Default overruns | Ceiling overruns |
|---|---:|---:|---:|---:|
| Watch6 | 8.048327 | 24 fps | 0 / 120 | 0 / 120 |
| Watch12 | 15.417011 | 12 fps | 0 / 120 | 0 / 120 |
| Play6 | 5.789862 | 24 fps | 0 / 120 | 0 / 120 |
| Play12 | 11.726538 | 12 fps | 0 / 120 | 0 / 120 |

Each case retained exactly 1,500 update and tick samples. Each of eight paced probes
retained 120 samples. All ceiling probes requested 120 fps and recorded the actual
server clamp; no retry was used. The isolated stdlib audit returned SERVING_PASS,
and every new receipt hash matched.

Intent `ff4a522f6b96bd8c42b31c5c37b629051ce3e43f9067a5dfdc4a6e622ca3bb34`
binds clean producer commit `23c571de289cc2d16029489935fbff5e9e6bbe6a` and amendment
SHA `ce753f69c162f036a66187ac8eac384ff84e30abcb2f13dc2849e316785559c1`.
The amendment was committed before intent creation and measurement.

Every recorded session/probe selected the exact B8 checkpoint, used the expected
roster and existing batched forward, and recorded no fallback, active veto, or
`last_error`. All nine raw quiet-machine observations showed AC power, open lid,
the owned qualification Python process only, and idle foreign git-sync/t3-run
tmux panes. No foreign session or process was modified.

All 137 inspected protected bindings—`src/`, the ego2s serving policy, registry,
and pin file—match the original T12 intent. The released default remains frp3-s12.
The prior audit still records checks 1–3 and 5–8 PASS and original check 4 FAIL;
all 61 files in its receipt remain hash-identical.

Validation logs confirm 104 backend tests, 15 frontend tests, 16 harness/audit
tests, the frontend build, and flake8 passed. Pre-run review findings in the new
harness/audit were resolved before the pinned commit and before timing; they
did not result in discarded qualification attempts.

Limits: the qualification covers these registered frame counts and observed Mac
conditions. Processing timing excludes socket transmission and client rendering.
The fps caps are processing-budget settings; actual cadence is lower under the
unchanged sleep-after-work loop. This check-4 pass carries earlier checks forward
and does not change the released default. The report retains the prior failure
and the diagnostic-only status of the additional timing breakdowns.
