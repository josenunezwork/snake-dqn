# Veto v8 (v7 space preference + v6 head avoidance), 2026-10-02/03

v8 (`src/evaluation/safety_veto_v8.py`, opt-in) = v7's space-preference decision at `lambda`,
then v6's opponent-head layer: if v7's choice is head-risky and a masked-legal, v5-eligible,
non-risky alternative exists, the choice is replaced by v7's own re-rank over those
alternatives (same speed mode first). With the head layer off, v8 is v7 exactly. Incumbent
throughout: **v7 at `lambda = 4`** (Tier-2 STRICT_PASS vs v5,
`apex-veto-v7-strict-20261002/run-v1`). Everything here is **Tier 1 (non-authoritative)**:
no default, champion, profile or served veto changed.

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

## Non-claims and next step

- ADVANCE is not promotion. v8(8) would need a Tier-2 strict gate against v7(4) on fresh
  worlds (its own namespace and source binding), then a serving qualification and a serving
  path (the SIMD engine supports v2/v5 only).
- At `lambda = 8` the head layer and the stronger space preference are confounded; the
  head-on drop (12 → 3) is consistent with the head layer, but this screen does not
  attribute the mass gain.
- Sizing note for a strict gate: per-mix SDs here 136 / 137 / 82 (frozen / scripted /
  mixed), with about 60% exact ties.
