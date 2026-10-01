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
