# B8 seed 3 extended Mac serving qualification (2026-10-17)

**Outcome: `SERVING_FAIL` / HOLD — not a release record.** The independent audit passed
checks 1–3 and 5–8 but failed check 4's paced high-speed validation. The released default
remains **frp3-s12 + v8** throughout this work:
`CHECKPOINT_RELEASED_DEFAULT = NAME_FRP3_S12`. B8 is selected for every candidate session
only with `SNAKE_SERVE_CHECKPOINT=b8-s3`.

The qualification is governed by the [T12 specification](specs/t12_b8_serving_qual_spec_2026-10-17.md)
and its immutable sibling [protocol](../../research/t12_b8_serving_20261017/protocol.md), SHA-256
`3d8b51675a2cdecbc476af56c48cfb4fa3ac6566eac1d4faf21c1870174bd406`.
The real run intent is `t12-serving-intent/v1`, SHA-256
`e03db4339b043e004575be7cbb30282243a8a7ebebb439e69455224361a6bd4b`, on clean commit
`70534958a6363c645d882c539912ef721c070501`. It binds B8 seed 3 checkpoint SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`, release environment
`SNAKE_SERVE_CHECKPOINT=b8-s3`, and the exhaustive seed audit
`SEED_AUDIT_T12_V2.json` SHA-256
`798402fbe40986047d2a69e5d45547b55c412ae2bb964c2d66d4e923d4de5015`.

Raw create-only evidence is rooted at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t12-b8-serving-20261017/run-v1`.
Portable copies are under [`research/t12_b8_serving_20261017/results/`](../../research/t12_b8_serving_20261017/results/):
the [intent](../../research/t12_b8_serving_20261017/results/intent.json),
[receipt](../../research/t12_b8_serving_20261017/results/receipt.json),
[audit](../../research/t12_b8_serving_20261017/results/audit.json),
[summary](../../research/t12_b8_serving_20261017/results/summary.json), and
[publication manifest](../../research/t12_b8_serving_20261017/results/publication.json).
An independent [evidence review](../../research/t12_b8_serving_20261017/results/REVIEW.md)
verified 61 receipt files and 1,024 source hashes, and preserves the latency-reporting
limitations and HOLD recommendation.
The [evidence.tar.gz](../../research/t12_b8_serving_20261017/results/evidence.tar.gz) archive
contains 98 files / 1,431,616 bytes and has SHA-256
`bd9fac2e663af2497dc1fdf6ab9e9f1c6fda40a882c2e4cd85b1716cb175fe3d`.

## Qualification state

All completed work was checked by the independent audit. Its `t12-serving-audit/v1` verdict
is `SERVING_FAIL`, SHA-256 `f77c5c33490f4749f6722e4de43fff89fc6b2caabb578446b7d6464703c627f9`.
The receipt SHA-256 is `36f3bba6ef25597615707118a62f24b6305fac2ac0e7f95d1ee55fb9fa609550`.

| Check | State | Evidence and current result |
|---|---|---|
| 1. Live observation identity | PASS | `p2_h5000.json` reports 9/9 identical H5000 records (SHA-256 `d51813cf…8bbe2`); `p2_h10000.json` reports 3/3 identical H10000 records (`75da79e5…b7cdf`). Served-session versus SIMD traces passed at H5000 for frozen and mixed rosters: `parity-000.json` (`281e0276…40255`) and `parity-001.json` (`8270367f…893d7`), each with no differing frame. The bound preflight suite includes `test_ego_live_bv8.py` and `test_ego_live_identity.py`. |
| 2. No serving veto | PASS | Candidate Watch resolution reports `active=False`, no wrappers or counters, and the reason `the safety veto applies only to a vector61 Apex policy`; Play is likewise unwrapped. The pin binds `veto_variant: none`, and B8's SHA is absent from `v8_gated_checkpoint_sha256s`. The 1,500-frame confidence sample observed zero active-veto frames. |
| 3. Registry, pin, and rollback | PASS | `defaults.json` (`14dc7f90…8fb05`) covers the empty/default, champion, frp3-s12, b8-s3, unknown, unfilled-pin, corrupt-bytes, and no-name-v7 matrix. It records B8 failure fallback through verified frp3-s12 and then champion when both pins are deliberately unfilled. Both live rollbacks stepped 100 frames with their expected checkpoint and v8 Watch wrapper. |
| 4. Latency and UI-speed ceiling | **FAIL** | All four update p95 values meet the `< 66.667 ms` threshold, but the full-tick speed validation fails at Watch12 after the one allowed lower retry. The audit therefore returns `SERVING_FAIL`; no additional pacing attempt was made. See [latency and failure detail](#latency-and-failure-detail). |
| 5. Served Watch roster and monitoring baseline | PASS | 25 all-B8 H5000 Watch episodes completed: 125,000 stepped hero decisions, 53 deaths, 124,947 alive frames, no `last_error`, and no veto wrappers. This all-B8 self-play roster is descriptive and was not a T7 gate roster. |
| 6. Play | PASS | All 25 real-control Play episodes completed: 17,878 frames total, each ending on natural human death. Human rows never entered AI observations or dispatch. `controls.json` (`20e9068f…475e3`) verifies opponent counts 1/5/11, isolated `POST /api/scores` submission plus score retrieval and `/api/players` lookup, and persisted served checkpoint identity `b8-s3`; legacy rows remain nullable/distinguishable. Its separate forced-terminal API fixture has zero survival frames; the 25 Play episodes establish gameplay coverage. |
| 7. UI and confidence | PASS | `confidence.json` (`448488af…461ef`) collected 1,500 usable Q vectors in 1,501 attempts: 1,177 confident (78.47%), 158 moderate (10.53%), and 165 close (11.00%); one dead/missing inspector frame and no non-finite Q values. `ui.json` (`90db3459…a1d60`) records a browser observation at `http://127.0.0.1:8812/#inspector`: narrator, steering wheel, both ego2s-bv8 inspector groups, and no veto counters. Portable [screenshot](../../research/t12_b8_serving_20261017/results/screenshot.jpg), [features view](../../research/t12_b8_serving_20261017/results/features.jpg), [DOM capture](../../research/t12_b8_serving_20261017/results/dom.txt), and [backend identity](../../research/t12_b8_serving_20261017/results/backend.json) are retained. The `v8 veto features (5 per action)` group is an input feature group, not an active serving veto. |
| 8. Train refusal | PASS | `controls.json` records both Train requests dropping to Watch with `policy_training=false`, epsilon zero, and `ego2s students are served forward-only; dropped to watch mode.` Train is therefore refused and remains unqualified for release. |

Because check 4 failed, this run is a **HOLD**. The release default was not changed.

## Latency and failure detail

Latency ran only after the read-only check found `t11-lanes` absent and after the owned
qualification workers and UI server had stopped. AC power and open lid were recorded. The
pre-run process observation showed unrelated `coreaudiod` activity at about one core; it was not
modified, and the raw timing data cannot establish it as the cause of any overrun.

Every case measured 1,500 advancing frames with a 200-frame warm-up after each construction
or Play reset. Actual warm-up totals were 200/200/600/1,045 for Watch6/Watch12/Play6/Play12;
an early human death restarts warm-up. All four update p95 values are below 66.667 ms.
Tick timing includes step and snapshot processing and is reported separately. Even the
largest update p95 uses only 20.52% of the default 12 fps budget (83.333 ms).

| Case | Update p50 / p95 / max (ms) | Tick p50 / p95 / max (ms) | Play restarts |
|---|---:|---:|---:|
| Watch6 | 6.375980 / 8.397673 / 31.864458 | 7.020750 / 9.229475 / 33.473875 | 0 |
| Watch12 | 12.507541 / 17.095975 / 39.566667 | 13.280667 / 33.051342 / 40.560584 | 0 |
| Play6 | 4.762625 / 5.526006 / 31.655292 | 4.839812 / 5.631202 / 32.001416 | 2 |
| Play12 | 9.394979 / 11.190676 / 37.428666 | 9.504209 / 11.408103 / 37.598208 | 5 |

The maximum full Watch12 tick implied an initial 24 fps processing bound. Its 120-frame
paced probe had one 55.9117 ms overrun against the 41.6667 ms budget. The protocol permits
one lower retry only. At 17 fps, the retry still had one 60.94329 ms overrun against the
58.82353 ms budget. Those two attempts are preserved, and no third timing attempt was made.
The final retry's achieved cadence was 11.53028 fps; it did not validate the 17 fps request.

| Case | Initial request / overruns | Permitted retry / overruns | Validated processing ceiling | Final measured cadence |
|---|---:|---:|---:|---:|
| Watch6 | 29 fps / 0 | None | 29 fps | 18.51102 fps |
| Watch12 | 24 fps / 1 | 17 fps / 1 | **Not established** | 11.53028 fps |
| Play6 | 31 fps / 0 | None | 31 fps | 19.84498 fps |
| Play12 | 26 fps / 1 | 15 fps / 0 | 15 fps | 10.59497 fps |

Each attempt contains 120 advancing ticks. These are processing-budget ceilings, not
achieved frame rates: the server sleeps a full `1 / speed` interval after doing its work.
Network transmission and client rendering are outside the measurement. The slider was not changed.

The raw latency JSON retains a known reporting defect: the aggregate post-retry
`estimated_achieved_fps` fields for Watch12 and Play12 remain their initial-bound values (24
and 26). They are not final measurements. This report uses the per-attempt
`actual_achieved_fps` values instead; Play12's final retry was 10.59497 fps. Likewise,
`shared_ui_speed_fps = 15` is the minimum attempted speed, not a validated global UI speed
ceiling, because Watch12 failed. The audit's generic failure wording, `paced retry count
exceeds the one permitted retry`, records the failed branch; the factual basis is the final
allowed retry's overrun.

The independent audit can be reproduced from the frozen evidence with:

```bash
./venv/bin/python -I research/t12_b8_serving_20261017/serving_audit.py \
  --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/t12-b8-serving-20261017/run-v1
```

Expected result: exit 1 with `SERVING_FAIL` for criterion 4. This is evidence validation,
not authorization to repeat the timing measurement.

## Validation before the real run

The bound preflight validation record is SHA-256
`c4d0021d221c7c91070e5ac6a5c4c3ca9d29ad1c3a53af12b2b9085a013161ef`.
It records all of the following successful checks:

- 149 core pytest cases, including live-observation, identity, web serving, score store,
  Play, and control coverage.
- 98 harness-and-serving pytest cases.
- 21 frontend confidence/narrator/steering-wheel tests and `npm run build --prefix web/frontend`.
- flake8 on the changed product and T12 Python files with an empty output log.

The 149 and 98 pytest counts overlap; they are not a combined count of unique tests.

The preliminary `SEED_AUDIT_T12.json` was retained as a limited hidden-coverage record only.
Before the real intent, it was superseded by the authoritative V2 audit above, which scans
the registered T7 banks/rejections/exclusions and repository/artifact JSON/JSONL evidence
with `--hidden --no-ignore`; it found no collision among 73 newly registered world seeds.
Earlier smoke parity v1 exposed a missing frame observer. That smoke evidence was preserved;
the repaired v2/v3 smoke probes passed. Neither smoke record is qualification evidence.

## Candidate, fallback, and rollback boundary

`b8-s3` is `Ego2sServingPolicy` over `ego2s-bv8`, with epsilon zero, a forward-only serving
path, and pin metadata `veto_variant: none`. Its selected bytes are pin-verified. A selected
B8 whose bytes or pin fail resolves to verified `frp3-s12 + v8`, then to champion if that
fallback cannot verify, with the reason recorded. The qualification does not alter the
released default.

The exercised rollback controls are:

```bash
SNAKE_SERVE_CHECKPOINT=frp3-s12  # serves frp3-s12 + v8, then restart
SNAKE_SERVE_CHECKPOINT=champion  # serves champion + v8, then restart
```

The prospective release change is exactly one line and has **not** been applied:

```python
CHECKPOINT_RELEASED_DEFAULT = "b8-s3"
```

Expected stderr after that future change begins with:

```text
served-checkpoint: name=b8-s3 requested=None sha256=3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8
safety-veto-serving: active=False ... reason=the safety veto applies only to a vector61 Apex policy
```

## Draft release record — conditional on final `SERVING_PASS`

**Claim:** **more mass at equal survival against the served agent, with no veto.** This is
not a safety claim.

The eventual release record must disclose that B8 was trained with v8 labels and receives
live v8 input features, while serving has no veto; Tier 2 margin was 0.034 above threshold
at SE 0.023; the all-B8 self-play serving roster was not gated; late deaths were at parity;
Train is not qualified; and scored Play rows are only comparable when their persisted served
checkpoint identity is considered. The claim derives from T7's gate evidence, not this
serving qualification.

## Post-release monitoring — conditional on release

For the first week **or** first 100,000 Watch-hero decisions, whichever ends later, roll
back on any recorded fallback reason or ego2s exception; on Watch hero deaths per 1,000
decisions exceeding `2 × qualification_baseline`; or on default-speed update p95 exceeding
`1.5 × qualification_latency_p95`. The measured Watch baseline is 53 deaths / 125,000
decisions = **0.424 deaths per 1,000 decisions**, so its prospective 2× trigger is
**more than 0.848 deaths per 1,000 decisions**. Logical length distribution: minimum 1,
p05 14, p25 75, median 172, p75 304, p95 528, maximum 867, mean 208.219784.
The measured default-roster Watch6 update p95 is 8.397673 ms; its prospective 1.5× trigger
would be above 12.596509 ms under matched measurement conditions. These are recorded
baselines, not authorization to start release monitoring: this failed qualification supplies
no approved release or global speed ceiling. Review late-death complaints without automatic rollback, because late death
is a disclosed parity behavior. The owner checks the counters after seven days.

## Hold boundary and final evidence

The run is complete, and its result is `SERVING_FAIL` rather than `SERVING_PASS`. No default
flip, source-code repair, or third timing attempt belongs in `run-v1`. The next investigation
is limited to the Watch12 full-tick tail and the stale post-retry summary metadata. Timing
alone identifies no root cause, including no causal conclusion about `coreaudiod`.

Any producer or serving-code change requires a sibling erratum, a newly registered producer
and seed namespace, and a fresh qualification. The frozen qualification source commit is
`70534958a6363c645d882c539912ef721c070501`; the publication commit is reported by the final
repository handoff rather than self-referenced here.
