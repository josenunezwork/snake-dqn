# Veto v7: v5 + space-preference re-rank, 2026-10-02

**Design.** v7 starts from v5's eligible actions (when v5 kept or vetoed) and
picks the one maximizing

    Q_norm + λ · min(area, cap) / cap

- `Q_norm` is min-max normalized Q over the eligible actions.
- `area` is the tail-aware reachable count after the move.
- `cap` = min(max(32, 2·length), 4096, open cells).
- In v5's no-spacious path, v5's choice is kept.
- λ = 0 reproduces v5 exactly.

**Motivation.** The v5 death census found that most self deaths are
enclosures sealed 20–60 frames before death.

## Dev λ sweep: **SELECTED λ = 4**

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-v7-lambda-sweep-20261002/run-v1` |
| Summary | sha256 `d665663d1ce5844b…` |
| Source | branch `900385f` |
| Namespace | `apex-veto-v7-dev-v1` |
| Size | 8 worlds per mix, paired vs v5; replay R = L200, 3/3 identical |

The selection rule was pre-declared, including an activity gate and a
clear-loser bound.

| λ | frozen | mixed | scripted | Pooled | Status |
|---|---:|---:|---:|---:|---|
| 1 | +29.5 | 0.0 | +10.8 | +13.4 | inactive |
| 2 | +178.7 | +48.9 | +84.9 | +104.2 | qualifies |
| **4** | +185.0 | +68.8 | +97.7 | **+117.1** | **selected** |

## Tier-1 screen (λ = 4): **RECOMMEND_STRICT_GATE**

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-v7-screen-20261002/run-v1` |
| Receipt | sha256 `b6596634fcdc13af…` |
| Source | `900385f` (bound to the sweep) |
| Namespace | `apex-veto-v7-screen-v1` |
| Episodes | 276; determinism and replay PASS |

| Mix | v5 | v7 | Δ (95% CI) | Holm-adj p | Better / equal / worse | Survived to H5000 v5 → v7 | Self deaths v5 → v7 |
|---|---:|---:|---|---:|---|---|---|
| frozen | 176.0 | 310.8 | +134.8 [90.1, 179.5] | 2e-7 | 27 / 12 / 1 | 5 → 25 | 31 → 10 |
| scripted | 161.3 | 315.2 | +153.9 [105.0, 202.8] | 2e-7 | 25 / 14 / 1 | 2 → 22 | 32 → 12 |
| mixed | 202.4 | 356.3 | +153.9 [110.6, 197.2] | 2e-8 | 29 / 10 / 1 | 5 → 27 | 33 → 10 |

This is a Tier-1 result, not a promotion.

**Next:** the strict Tier-2 gate against the released v5. The pilot paired SD
is about 135–153, so MDE 20 would need N = 547 > Nmax. The strict gate
pre-declares MDE = 40 per mix (N = 137). This is disclosed: chosen after
seeing this screen, and far below the observed effect.

**Amendment (before the strict run).** The MDE of 40 above was replaced, before
the run, by a rule: use the smallest MDE in steps of 5 that fits Nmax = 300.
That gives MDE 30 and N = 243 per mix (MDE 25 would need N = 350). The value
was still chosen after the screen, and this is disclosed in the intent
(branch `df86c32`).

## Strict Tier-2 gate (λ = 4 vs released v5): **STRICT_PASS**

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-v7-strict-20261002/run-v1` |
| Source | branch `df86c32` (merged into local `main`) |
| Intent | sha256 `3e39e0699ac1ba27…` |
| Receipt | sha256 `86ee36791e33ecad…` |
| Audit report | sha256 `9cfc7dbb7eaa6d33…`; independent audit PASS; decisions agree |
| Final | N = 243 worlds per mix, H5000, fresh namespace `apex-veto-v7-strict-final-v1` |
| Authorization | standing user authority |

The incumbent is champion + v5 (released). The candidate is champion + v7
(λ = 4). Both are hero-only wrappers against unwrapped opponents.

| Mix | v5 mass | v7 mass | Δ (95% CI) | Holm-adj p | Survival v5 → v7 | Better / equal / worse |
|---|---:|---:|---|---:|---|---|
| frozen | 191.8 | 322.1 | +130.3 [113.6, 146.9] | 1e-37 | 0.626 → 0.840 | 167 / 66 / 10 |
| scripted | 191.5 | 310.6 | +119.1 [100.1, 138.1] | 7e-28 | 0.562 → 0.741 | 137 / 93 / 13 |
| mixed | 211.4 | 334.4 | +123.0 [105.9, 140.2] | 2e-33 | 0.664 → 0.859 | 155 / 80 / 8 |

- **Combined:** mean delta CI95 [113.9, 134.3], n = 243 per mix.
- **Superiority:** Holm superiority in 3 of 3 mixes (2 required).
- **Scripted noninferiority:** lower bound +103.2 against a margin of −5.79.
- **Survival bands:** PASS in all three mixes.

The strict gate's own serving stage was a rollout-harness self-play
compatibility check, not the web path. Its closeout therefore records
`serving_path_qualified=false`. The web serving qualification below fills
that gap.

## Web serving qualification: **SERVING_PASS** (2026-10-03 UTC)

| Item | Value |
|---|---|
| Lane | `research/apex_veto_v7_serving_20261002` (copy of the v5 lane, with v7 identities) |
| Root | `snake-dqn-artifacts/apex-veto-v7-serving-20261002/run-v1` |
| Source | `main` `2f95e03`, clean detached worktree; protocol sha256 `6028193ecda2…` |
| Records | intent `074a3781…`, receipt `cf287fe7…`, audit `ab2bd395…` |
| Design | 25 Watch + 25 Play episodes at H5000, plus 2 parity probes at H5000, through the real `GameSession` and `app._apply_control` with `SNAKE_SERVE_VETO_VARIANT=v7` |
| Seeds | fresh domain `apex-veto-v7-web-serving-v1`; `seed_report.disjoint=true` |
| Compute | 1,650 s wall, one shared CPU slot (`cpu-slot-2`), on AC power |
| Result | 0 failures; S1–S6 all PASS; `serving_path_qualified=true` |

- **Watch:** the v7 wrapper was on the hero only in all 25 episodes.
  - 125,000 decisions; 257 actions replaced (0.21%).
  - 53 of those were v7 re-rank changes of a base action that v5 kept (3 driven by the tail-release model).
  - The other 204 were v5 vetoes, including 47 boost-landing vetoes.
  - Mean wrapper cost was 0.58 ms per decision.
- **Play:** unwrapped in all 25 runs; every run ended in a human death.
- **Parity:** both probes matched the v7 strict gate's rollout frame for frame. Traces, counters and v7/v5 diagnostics were identical. Label `rerank-exercised`: 6 re-rank changes and 21 replacements across the 2 probes.
- **Released default (S6):** an empty environment still serves v5. The `v5` and `v2` variants and `WATCH_HERO=0` all behave as documented.

This run checks serving correctness only. It makes no skill claim. The
default is **not** changed: v7 is served only with `SNAKE_SERVE_VETO_VARIANT=v7`.

## Release step (user-approved, not done here)

1. Set `VARIANT_RELEASED_DEFAULT = VARIANT_V7` in `web/backend/safety_veto_serving.py`.
   Update the tests that pin v5 as the default:
   - `tests/test_web_safety_veto_variant.py` (`TestVariantParsing`, `TestReleasedV5Default`)
   - `tests/test_web_safety_veto_v7.py` (`TestSelection`, `TestReleasedDefaultStillV5`)
   - the S6 pre-release precondition tests in the v5 and v7 serving run tests, which should pin the old default with monkeypatch, as the v5 release did.
2. Restart the server and check that stderr shows
   `safety-veto-serving: active=True scope=watch_hero … strict_checkpoint_match=True wrapper_source_sha256=56ff7009… variant=v7 variant_requested=v7 wrapper_sources_match=True wrapper_method=free-space-veto/v7-space-preference(lambda=4.0) reason=None`.
3. Rollback needs no code change. Restart with:
   - `SNAKE_SERVE_VETO_VARIANT=v5` to return to v5
   - `SNAKE_SERVE_VETO_VARIANT=v2` to return to v2
   - `SNAKE_SERVE_VETO_WATCH_HERO=0` to turn the veto off

## Release (2026-10-03)

v7 (`free-space-veto/v7-space-preference(lambda=4.0)`) is now the released Watch-hero default.
`VARIANT_RELEASED_DEFAULT = VARIANT_V7` in `web/backend/safety_veto_serving.py`.

- **Evidence:**
  - STRICT_PASS, from `apex-veto-v7-strict-20261002/run-v1` (receipt `86ee3679…`).
  - SERVING_PASS, from `apex-veto-v7-serving-20261002/run-v1` (receipt `cf287fe7…`, audit `ab2bd395…`).
- **Live check** on the released tree, with no environment set:
  - `variant=v7`, `active=True`, `scope=watch_hero`.
  - `strict_checkpoint_match=True`, `wrapper_sources_match=True`.
  - `wrapper_source_sha256=56ff7009…`, `reason=None`.
- **Tests:**
  - Web and serving tests: 458 passed, exit code 0.
  - Full suite before the flip: 4472 passed. The only 2 failures were the known ones in `tests/test_eval_controlled_parity.py`.
- **Rollback (no code change):** restart with one of:
  - `SNAKE_SERVE_VETO_VARIANT=v5`
  - `SNAKE_SERVE_VETO_VARIANT=v2`
  - `SNAKE_SERVE_VETO_WATCH_HERO=0` (veto off)
- **Unchanged:** Play AI wrapping stays opt-in and v2-only.
