# Boost-aware veto v5, 2026-10-01

## Tier-1 screen: **RECOMMEND_STRICT_GATE**

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-v5-screen-20261001/run-v1` |
| Receipt | sha256 `f3f5643e877e1bb1…` |
| Source | `c9ef706` |
| Namespace | `apex-veto-v5-screen-v1` |
| Episodes | 276; self-check PASS |
| Determinism | 24/24 C=A, 12/12 D=B |

**Motivation:** `docs/research/trap_horizon_2026-10-01.md` (branch). Of 42
self-collision deaths under v2, 24 were boosts into a tiny pocket. v2 scored
boosts only from their first cell.

**v5:** identical to v2 for normal-speed actions. A boost is eligible only if
its two-cell landing also has reachable space ≥ need. A failing boost is
replaced by the same direction at normal speed.

Arm A is champion + v2 (released); arm B is champion + v5.

| Mix | A (v2) | B (v5) | Δ (95% CI) | Holm-adj p | Worlds better / equal / worse |
|---|---:|---:|---|---:|---|
| frozen | 125.8 | 183.2 | +57.4 [29.4, 85.4] | 0.0002 | 21 / 19 / 0 |
| scripted | 93.7 | 144.8 | +51.1 [18.9, 83.3] | 0.0013 | 14 / 26 / 0 |
| mixed | 122.5 | 194.6 | +72.1 [37.9, 106.4] | 0.0002 | 19 / 20 / 1 |

v5 changed about 0.03% of decisions (frozen: 36 boost-landing vetoes, all
boost → normal speed in the same direction).

This is a Tier-1 screen: non-authoritative, and not a promotion. Next comes
the strict Tier-2 gate (`research/apex_veto_v5_strict_20261001`, branch
commit `5981251`) against the released v2.

## Strict Tier-2 gate: **STRICT_PASS** (2026-10-02)

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-v5-strict-20261001/run-v1` |
| Intent | `aa86ea18…` |
| Source | branch `5981251` |
| Authorization | standing user authority |
| Records | closeout `ff959fd0…`, receipt `cd843edf…`, decision `d891c407…` |
| Stages | calibration 48, final 1,614 (N = 269/mix), serving 50 episodes |
| Audit | independent audit PASS; decisions agree |

The incumbent is champion + released v2 veto; the candidate is champion + v5.
Fresh namespaces were used. N came from the v5 screen pilot at the
pre-declared MDE of 20 mass per mix.

| Mix | v2 mass | v5 mass | Δ (95% CI) | Holm-adj p | Survival v2 → v5 | Better / equal / worse |
|---|---:|---:|---|---:|---|---|
| frozen | 108.2 | 179.8 | +71.6 [58.6, 84.6] | 1e-22 | 0.451 → 0.600 | 140 / 121 / 8 |
| scripted | 99.6 | 163.0 | +63.3 [50.2, 76.5] | 1e-18 | 0.383 → 0.519 | 125 / 140 / 4 |
| mixed | 135.3 | 198.1 | +62.9 [48.9, 76.8] | 5e-17 | 0.513 → 0.640 | 129 / 127 / 13 |

- **Superiority:** Holm superiority in 3 of 3 mixes.
- **Scripted noninferiority:** lower bound +52.3 against a margin of −3.17.
- **Survival bands:** PASS.

Compared with the original no-veto champion from the v2 gate (49 / 48 / 62),
v5 is about 3–3.6× the mass.

This is a receipt only. The web serving path still has to be qualified for v5
before release.

## Web serving qualification: **SERVING_PASS** (2026-10-02)

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-v5-serving-20261002/run-v1` |
| Source | branch `6c013cd` |
| Design | 25 Watch + 25 Play episodes + 2 parity probes, through the real `GameSession` and `app._apply_control` |
| Result | 0 failures; S1–S6 all PASS; `serving_path_qualified=true` |

The split was accepted under standing authority. Only the Watch hero is
wrapped by the release, so 25 Watch episodes give more relevant evidence than
the 1 + 49 originally asked for.

## Release (2026-10-02): **DONE**, local `main`, not pushed

| Item | Value |
|---|---|
| Merge | `161d04c` |
| Release commit | `03abe8e` |
| Change | `VARIANT_RELEASED_DEFAULT = v5`; the Watch hero gets v5 when no environment variable is set |
| Rollback to v2 | `SNAKE_SERVE_VETO_VARIANT=v2` |
| Veto fully off | `SNAKE_SERVE_VETO_WATCH_HERO=0` |
| Play AI | stays opt-in |
| Fast suite on `main` | 3,832 passed; only the 2 known pre-existing `test_eval_controlled_parity` failures |

Test-isolation fix included: the app thread-pin test now runs in a fresh
interpreter, and conftest restores torch's intra-op thread count after each
test.

Live verification (`snake-web` from `.claude/launch.json`, no environment
variables set) logged:

```
safety-veto-serving: active=True scope=watch_hero mode=watch wrapped_ids=[0]
flags={'watch_hero': True, 'play_ai': False} checkpoint_sha256=43d4e2c5…
strict_checkpoint_match=True wrapper_source_sha256=d86d084e…
variant=v5 variant_requested=v5 wrapper_sources_match=True
wrapper_method=free-space-veto/v5-boost-aware reason=None
```

There were no server errors.

**Cumulative effect at H5000 for the Watch hero:**

| Configuration | Mass, frozen / scripted / mixed |
|---|---|
| Original champion | 49 / 48 / 62 (v2 gate run-v3) |
| Champion + v2 | 108 / 100 / 135 (v5 gate incumbent) |
| Champion + v5 | 180 / 163 / 198 |

Absolute levels vary between banks, but each step was gated on paired fresh
worlds.
