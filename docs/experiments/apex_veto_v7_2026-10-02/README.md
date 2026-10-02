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
