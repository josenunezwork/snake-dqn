# T13b six-snake roster measurement — 2026-10-17

## Pre-registered decision rule

At six snakes, keep all-B8 if the all-B8 hero's deaths per 1,000 are within 0.10
of the mixed roster's AND its p95 length is within 100 cells of the FRP3 hero's;
otherwise the mixed roster (b8-s3 hero, frp3-s12 + v8 opponents) becomes the
planned Watch default, subject to a separate roster qualification (not in this job).

This rule is committed and pushed before running. Apply it to the pooled point
estimates from the fixed 25 episodes; bootstrap intervals remain descriptive.
Both conditions must hold. Serving defaults remain unchanged in either outcome.

## Fixed measurement plan

Authority: [T13b spec](specs/t13b_six_snake_spec_2026-10-17.md) and binding
[advice steps 1–3 and decision rule](advice/2026-10-17_after_t13.md).
Reuse the T13 real GameSession step/snapshot harness and metric definitions.

- Four six-snake rosters: all-B8 without veto; all-FRP3 with v8 on hero only;
  all-champion with v8 on hero only; B8 hero with FRP3 + v8 on every opponent.
- Also measure all-B8 at eight snakes. Each row has 25 H5000 episodes.
- Allocate 25 fresh paired seeds, disjoint from T7/T12/T13 registered banks,
  and a separate smoke seed. The same seeds pair the four six-snake rosters
  and the eight-snake supplemental measurement.
- Six snakes uses the mechanics_v2.yaml default. Eight uses a constructor-local
  research override only. Record identities, wrapper scopes and effective counts.
- Preserve T13's hero/arena table and paired percentile bootstrap method:
  10,000 resamples, NumPy RNG seed 20261017, pooled metrics recomputed per draw.
- CPU only, at most four single-thread episode workers plus one coordinator.
  No retries, no tmux operations, no operations on processes not started here.
- Add a separate monitoring erratum: Watch6 baseline 0.424, trigger 0.848;
  Watch12 baseline 0.768, trigger 1.536; Watch8 baseline from this measurement,
  trigger twice that baseline. Apply death triggers separately by roster size.
  Preserve the release record and existing latency-qualification boundaries.

Results and rule application will be appended after the fixed run completes.
