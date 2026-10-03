# Veto v8 (v7 space preference + v6 head avoidance), 2026-10-02/03

v8 (`src/evaluation/safety_veto_v8.py`, opt-in) = v7's space-preference decision at `lambda`,
then v6's opponent-head layer: if v7's choice is head-risky and a masked-legal, v5-eligible,
non-risky alternative exists, the choice is replaced by v7's own re-rank over those
alternatives (same speed mode first). With the head layer off, v8 is v7 exactly. Incumbent
throughout: **v7 at `lambda = 4`** (Tier-2 STRICT_PASS vs v5,
`apex-veto-v7-strict-20261002/run-v1`). Sections 1-2 are **Tier 1 (non-authoritative)**; section 3 is the Tier-2 strict gate and
section 4 the web serving qualification. No default, champion, profile or served veto changed.

## 1. DEV lambda sweep: `SELECTED lambda = 8` (also the 3-slot calibration: PASS)

| Item | Value |
|---|---|
| Package | `research/apex_veto_v8_lambda_sweep_20261002` (commit `81a5453`) |
| Root | `snake-dqn-artifacts/apex-veto-v8-lambda-sweep-20261002/run-v1` |
| Summary | `merged/summary.json`, sha256 `2a3eb8b3…4720b2` |
| Namespace | `apex-veto-v8-dev-v1`, 8 worlds/mix, H5000 (now burned) |
| Episodes | 99 in 3 shards; self-check PASS; R = H800 replay 3/3 identical |

Paired mass-integral delta vs v7(4), per mix (wins/losses of 8), one-sided 90% upper bound:

| Arm | frozen | mixed | scripted | pooled (SD) |
|---|---|---|---|---|
| H400 (v8, lambda 4 = head layer only) | -37.1 (0/1), UB 15.4 | -5.6 (0/1), UB 2.3 | +137.1 (4/0) | +31.5 (137.8) |
| **H800 (v8, lambda 8)** | -14.1 (0/2), UB 1.4 | +19.5 (1/1), UB 48.4 | +137.1 (4/0) | **+47.5 (123.5)** |
| H1600 (v8, lambda 16) | -15.6 (0/2), UB 0.3 | +19.5 (1/1), UB 48.4 | +137.1 (4/0) | +47.0 (123.8) |

All three were active and none a clear loser, so the rule picked the highest pooled mean,
`lambda = 8`. Most worlds were exact ties; the frozen dip was the open question for the
screen. Calibration (3 concurrent shards vs the 2-slot v7 baseline): `CALIBRATION_PASS`,
arm-A wall ratio 1.07 pooled (frozen 1.02 / mixed 1.27 / scripted 0.93), 99 guard checks,
0 not-ok, 0 pauses, 0 stops, largest in-run slowdown ratio 1.14.

## 2. Tier-1 screen v8(8) vs v7(4): **ADVANCE**

| Item | Value |
|---|---|
| Package | `research/apex_veto_v8_screen_20261002` (`protocol.md`, `screen.py`, tests `tests/test_apex_veto_v8_screen.py`) |
| Source | commit `e260617` (clean tree, no untracked files); protocol Amendment 1 (review fixes) before any real episode |
| Root | `snake-dqn-artifacts/apex-veto-v8-screen-20261002/run-v1` |
| Summary | `merged/summary.json`, sha256 `ecd205e6…c0c3c4` |
| Namespace | `apex-veto-v8-screen-v1`, **60 worlds/mix** (180 pairs), H5000, `promotion-v2-watch-rect` |
| Disjointness | fail-closed preflight PASS: first 1000 seeds of every earlier domain (incl. v8 DEV sweep, v7 strict banks, v7 screen/sweep, v7 web serving) plus observed seeds from saved v7-strict rosters, v8-sweep intents/summary, v7 sweep/screen and v7 serving intents |
| Episodes | 372/372 (A 180, B 180, C 6, D 6); self-check PASS (0 failures, 0 warnings); C=A 6/6, D=B 6/6 across processes |
| Compute | 3 shards behind a start barrier on slots 2/1/3, launched after `slots-free` (the v7 web serving run had released its slot); 03:57:58-05:13:08 UTC (75 min wall); mean episode 33.1 s (A) / 36.8 s (B) |
| Thermal guard | 372 checks, 0 not ok, 0 pauses, 0 stops; largest in-run slowdown ratio 1.36 (threshold 1.60); AC power throughout |

Decision rule (pre-registered): ADVANCE iff pooled mean > 0, pooled one-sided 90% lower
bound > 0 and no mix with one-sided 90% upper bound < 0.

| Mix | A v7(4) | B v8(8) | Δ mean | 90% one-sided LB / UB | wins / ties / losses |
|---|---:|---:|---:|---|---|
| frozen | 304.2 | 342.8 | **+38.6** | 15.8 / 61.4 | 20 / 31 / 9 |
| scripted | 314.1 | 333.6 | +19.6 | -3.3 / 42.4 | 16 / 34 / 10 |
| mixed | 343.2 | 344.6 | +1.4 | -12.3 / 15.0 | 8 / 43 / 9 |
| **pooled** | | | **+19.8** (SE 9.0, df 154) | **LB +8.2** | 44 / 108 / 28 |

No mix shows a loss; pooled LB > 0 → **ADVANCE**. The sweep's frozen worry did not
replicate: frozen is the strongest mix here (its LB alone is positive). Scripted and mixed
gains are not established on their own.

Reported only:

| Mix | head-on deaths A → B | survived A → B | mean survival Δ | B head vetoes / risky decisions | B decisions differing from v7(4) |
|---|---|---|---|---|---|
| frozen | 5 → 1 | 28 → 38 | +0.068 | 75 / 79 | 87 (29 episodes) |
| scripted | 5 → 1 | 27 → 31 | +0.047 | 41 / 44 | 57 (26 episodes) |
| mixed | 2 → 1 | 38 → 41 | +0.007 | 17 / 19 | 25 (17 episodes) |
| total | **12 → 3** | 93 → 110 | | 133 / 142 | 169 of 770,590 decisions (2.2e-4) |

Self deaths 68 → 58; one wall death appeared in B (scripted). Informational pools (not the
decision): stratified two-sided 90% CI [+4.9, +34.8]; the crossed mixes x worlds estimate
(models the shared seeds) +19.8, 90% CI [-9.9, +49.6] (df 2.4, only 3 mix rows) and the
random-effects pool 90% CI [-16.8, +49.4] both include 0. The lower bound used by the rule
treats mixes as independent strata; with the shared-seed covariance modelled, the pooled
gain is not established at 90%.

## 3. Strict Tier-2 gate v8(8) vs released v7(4): **STRICT_PASS**

| Item | Value |
|---|---|
| Package | `research/apex_veto_v8_strict_20261003` (branch `v8-strict` `4ff58d4`, merged into local `main` as `a285b48`) |
| Root | `snake-dqn-artifacts/apex-veto-v8-strict-20261003/run-v1` |
| Intent | sha256 `ca4aa97074456fc1…` (binds the champion and all 7 veto source shas) |
| Receipt | sha256 `29b1f7f6f095cac1…`; decision `STOP_PASS` at look index 2 |
| Audit report | sha256 `72410ec575301def…`; independent audit PASS; decisions agree |
| Method | `strict-sequential-obf-bonferroni-v1`: 4 OBF looks (63/125/187/249 per mix), family alpha 0.05, 2 of 3 mixes required, scripted noninferiority, survival bands at stop |
| Final | 187 of N_max 249 worlds per mix (561 paired worlds), H5000, fresh namespace `apex-veto-v8-strict-final-v1` |
| Authorization | standing user authority |

The incumbent is the champion with the released v7 (λ = 4). The candidate is the champion
with v8 (λ = 8, head layer on). The screen's diagnostic `reference_lambda` was never
installed. Both are hero-only wrappers against unwrapped opponents.

| Mix | v7 mass | v8 mass | Δ (sd) | t at look 2 (boundary 2.543) | crossed at look | Survival v7 → v8 | Better / equal / worse |
|---|---:|---:|---|---:|---:|---|---|
| frozen | 307.1 | 337.2 | +30.1 (136.4) | 3.02 | 2 | 0.815 → 0.874 | 62 / 95 / 30 |
| scripted | 312.5 | 355.5 | +43.0 (165.0) | 3.57 | 1 | 0.764 → 0.833 | 61 / 91 / 35 |
| mixed | 332.1 | 368.6 | +36.5 (124.1) | 4.02 | 2 | 0.854 → 0.918 | 50 / 118 / 19 |

- **Efficacy:** boundary crossed in all 3 mixes (2 were required).
- **Scripted noninferiority:** lower bound +18.5 against a margin of −8.69.
  The margin is 0.03 × the calibration incumbent's scripted mean.
- **Survival bands:** PASS at every look.
- **Means are descriptive:** the naive means above are not bias-adjusted for the early stop.

## 4. Web serving qualification: **SERVING_PASS** (2026-10-03 UTC)

| Item | Value |
|---|---|
| Hook | `SNAKE_SERVE_VETO_VARIANT=v8` (opt-in, Watch hero only) in `web/backend/safety_veto_serving.py` (`6b5643e`) |
| Hook binding (fails closed) | champion sha, all 7 source shas, method `free-space-veto/v8-space-and-head(lambda=8.0)`, head layer on, no reference lambda; receipt, intent and audit shas pinned |
| Lane | `research/apex_veto_v8_serving_20261003` (copy of the v7 lane with v8 identities) |
| Root | `snake-dqn-artifacts/apex-veto-v8-serving-20261003/run-v1` |
| Source | `main` `bf70735`, clean detached worktree; protocol sha256 `4bdb4c0ab3cb…` |
| Records | intent `31ab8e30…`, receipt `5ac0836d…`, audit `bd7eec98…` |
| Design | 25 Watch + 25 Play episodes at H5000, plus 2 parity probes at H5000, through the real `GameSession` and `app._apply_control` with `SNAKE_SERVE_VETO_VARIANT=v8` |
| Seeds | fresh domain `apex-veto-v8-web-serving-v1`; `seed_report.disjoint=true`. Checked against the v7 lane's set, the v8 strict exclusion set and banks, and every seed the v8 strict run played |
| Compute | one shared CPU slot (`cpu-slot-1`), on AC power. Paused from 13:56 to 15:36 UTC while the lid was closed (see below). 6,927 s monotonic, which includes the pause |
| Result | 0 failures; S1–S6 all PASS; `serving_path_qualified=true` |

- **Watch:** the v8 wrapper was on the hero only in all 25 episodes.
  - 125,000 decisions; 360 actions replaced (0.29%).
  - 96 head vetoes (95 of them replaced a v7-kept choice). 99 head-risky decisions; 3 kept with no alternative.
  - 66 v7 re-rank changes; 199 v5 vetoes, including 40 boost-landing vetoes.
  - Mean wrapper cost was 0.65 ms per decision over 24 episodes. `watch_hero-007` is excluded: it was mid-decision during the pause, so its one apply call recorded about 5,088 s. Timing is reported only, never gated.
- **Play:** unwrapped in all 25 runs; every run ended in a human death.
- **Parity:** both probes matched the v8 strict gate's rollout (`spec.install_candidate`) frame for frame. Traces, counters and v8/v7/v5 diagnostics (including v7's probe counters) were identical. Label `head-exercised`: 9 head vetoes, 2 v7 re-rank changes and 17 replacements across the 2 probes.
- **Released default (S6):** an empty environment still serves v7. The `v7`, `v5` and `v2` variants and `WATCH_HERO=0` with `v8` all behave as documented.

**Execution note.** The launcher held the slot lock and checked AC power at launch. It did not
check the lid, which was closed. The Mac was in sleep and dark-wake cycles, and one sleep was
logged as a "Dark Wake Thermal Emergency". The run was therefore paused with SIGSTOP while
the lid was closed and resumed when it opened (`lidguard.log` next to the root). Results are
deterministic and do not depend on wall time. Only the reported timing of the episode that
straddled the pause is affected. Future launchers should also require the lid open.

This run checks serving correctness only. It makes no skill claim. The default is **not**
changed: v8 is served only with `SNAKE_SERVE_VETO_VARIANT=v8`.

## Release step (user-approved, not done here)

1. Set `VARIANT_RELEASED_DEFAULT = VARIANT_V8` in `web/backend/safety_veto_serving.py`.
   Update the tests that pin v7 as the default:
   - `tests/test_web_safety_veto_variant.py` (`TestVariantParsing::test_released_default_is_v7`)
   - `tests/test_web_safety_veto_v7.py` (`TestSelection::test_v7_is_the_released_default`, `TestReleasedDefaultV7`)
   - `tests/test_web_safety_veto_v8.py` (`TestSelection::test_v8_is_opt_in_and_v7_stays_the_released_default`, `TestReleasedDefaultStillV7`)
   - The v8 serving-run S6 test already pins the pre-release v7 default with monkeypatch.
2. Restart the server and check that stderr shows
   `safety-veto-serving: active=True scope=watch_hero … strict_checkpoint_match=True wrapper_source_sha256=faf3695f… variant=v8 variant_requested=v8 wrapper_sources_match=True wrapper_method=free-space-veto/v8-space-and-head(lambda=8.0) reason=None`.
3. Rollback needs no code change. Restart with:
   - `SNAKE_SERVE_VETO_VARIANT=v7` to return to v7 (or `v5`/`v2`)
   - `SNAKE_SERVE_VETO_WATCH_HERO=0` to turn the veto off

## Non-claims

- STRICT_PASS + SERVING_PASS qualify v8 for release; they do not release it.
- At `lambda = 8` the head layer and the stronger space preference are confounded. The strict
  gate establishes v8(8) > v7(4) as a package; it does not attribute the gain.
- The served Watch hero respawns, while the strict gate measured a terminal hero. Play AI
  wrapping stays opt-in and v2-only. The SIMD engine path is not part of this
  qualification.

## Release (2026-10-03)

v8 (`free-space-veto/v8-space-and-head(lambda=8.0)`) is now the released Watch-hero default:
`VARIANT_RELEASED_DEFAULT = VARIANT_V8` in `web/backend/safety_veto_serving.py`. It was released by the research loop owner under the user's standing release authority.

- **Evidence:**
  - STRICT_PASS, sequential gate: `apex-veto-v8-strict-20261003/run-v1`, receipt `29b1f7f6…`.
  - SERVING_PASS: `apex-veto-v8-serving-20261003/run-v1`, receipt `5ac0836d…`, audit `bd7eec98…`.
- **Live check** on the released tree, with no environment set:
  - `variant=v8`, `active=True`, `scope=watch_hero`.
  - `strict_checkpoint_match=True`, `wrapper_sources_match=True`.
  - `wrapper_source_sha256=faf3695f…`, `reason=None`.
- **Tests:**
  - Web and serving tests: 587 passed, exit 0.
  - Full suite before the flip: 5159 passed. The 3 failures were:
    - the 2 known failures in `tests/test_eval_controlled_parity.py`;
    - `test_apex_veto_v8_strict.py::test_cli_prepare_resolves_the_real_spec_when_the_runner_is_main`, which is out of date. The finished run used up its final-namespace ledger entry, and the runner correctly refuses to reuse it.
- **Rollback:** no code change needed. Restart with `SNAKE_SERVE_VETO_VARIANT=v7` (or `v5`/`v2`), or with `SNAKE_SERVE_VETO_WATCH_HERO=0` to turn the veto off.
- **Unchanged:** Play AI wrapping stays opt-in and v2-only.
