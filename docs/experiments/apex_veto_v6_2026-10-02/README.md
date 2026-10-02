# Veto v6 (v5 + opponent-head avoidance + no-spacious fallback): Tier-1 screen, 2026-10-02

**Decision: NOT_ADVANCED.**

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-v6-screen-20261002/run-v1` |
| Receipt | sha256 `f191877b9d806d0e…` |
| Source | branch `7bf0644` |
| Namespace | `apex-veto-v6-screen-v1` |
| Episodes | 276; determinism and replay PASS |

Arm A is champion + v5 (released); arm B is champion + v6. Motivation: the
v5 death census (`docs/research/death_census_v5_2026-10-02.md`, branch).

| Mix | v5 | v6 | Δ (95% CI) | Holm-adj p | Better / equal / worse | Head-on deaths v5 → v6 |
|---|---:|---:|---|---:|---|---|
| frozen | 193.3 | 203.3 | +10.0 [−30.4, 50.3] | 0.38 | 11 / 22 / 7 | 3 → 1 |
| scripted | 147.8 | 177.8 | +30.0 [4.9, 55.1] | 0.031 | 13 / 22 / 5 | 6 → 1 |
| mixed | 191.9 | 205.3 | +13.4 [−17.0, 43.8] | 0.38 | 12 / 19 / 9 | 1 → 0 |

The pre-registered rule needed Holm superiority in at least 2 of 3 mixes; only
scripted passed.

**Interpretation (descriptive):**

- **Head avoidance:** it works on its target. Head-on deaths fell 10 → 2.
  Scripted, where head-ons concentrate, gained significantly.
- **No-spacious fallback:** it rarely acted (frozen: 1,313 fallback decisions,
  12 landing switches, 2 escape switches).

**Decision not to pursue v6 further:**

- Served Watch mode plays against champion copies, which is closest to the
  frozen mix, where v6's estimated gain is small and uncertain.
- The largest remaining loss is early enclosure. Most self deaths are sealed
  20–60 frames before death, beyond what veto-style search reaches.

v6 is retired. The released configuration stays v5.
