# Big-Lever Prefix Escape Dividend (`E6`), Frontier-Existence (`E7`), and Population Bridge (`E10`)

**Dates:** 2026-10-21
**Verdict:** Built, Mac-preflighted, and held pre-dispatch (`NOT_DISPATCHABLE` / `NOT_DISPATCHED`, `0` scientific Colab units spent) because upstream `E2` `R`/`F` candidate policies were blocked by the `E9` power gate and never produced.

## What was tried
- Registered `E6` (`REGISTRATION_E6.md`): substitute `64` candidate (`R` or `F`) decisions starting `100` decisions before late `b8-s3` deaths (`frame >= 2000`, `L >= 700`) before `b8-s3` resumes across `128` death worlds (`44` scripted, `42` frozen, `42` mixed) and `128` length-matched (`±100` cells) ordinary worlds.
- Registered `E7` (`REGISTRATION_E7.md`): roll out `8` stochastic `b8-s3` prefixes (`5%` non-greedy softmax temperature) from `100`, `150`, and `200` decisions before death across `128` death worlds and `2` independent confirmation RNG streams (`>= +20` mass on `>= 20%` of worlds -> `TRANSFER_ADMISSIBLE` vs `FRONTIER_SCARCE`).
- Registered `E10` (`REGISTRATION_E10.md`): two-phase population bridge `theta = mean_m(q_m * mu_m)` testing whether `UB90(U)` reaches `+40` (`FINITE_MENU_INSUFFICIENT` if `UB90(U) < 40`, minimum rescue fraction `r >= 0.163`, score ceiling `B = 2501.5`).
- Audited `11,224` high-range `+2**40` seeds, built an ordinal-label adapter around `strict_promotion.materialize_rosters`, and ran 2-worker full-H5000 Mac preflights (`production_preflight.py`).

## What happened
- **Seed audit & roster adapter (`PASS`):** Shifting derived seeds by `+2**40` (`bl-prefix-e6e7-development/20261021/v2`) passed with `0` collisions across `11,224` seeds (`3,998` unburned exploration seeds >= `3,072` needed after retiring `2` operational timing seeds), and `54` package tests passed with exact `300`-frame restoration parity.
- **Full-H5000 runtime sizing (`CAP_NOTE.md`):** Across 2 Mac workers, `E6` measured `0.0032107433 s/step` (`37.76M` max world-steps -> `520.156944 min` / `17.338565 u` expected on 12-worker L4 at `3×` slowdown, needing a `43.4 u` cap vs original `4 u`), and `E7` measured `0.0031417799 s/step` (`24.96M` max world-steps -> `341.745108 min` / `11.391504 u`, needing `28.5 u` vs original `20 u`).
- **Dispatch gate (`NOT_DISPATCHED`):** Held with `resource_readiness.ready = true` and `bl_prefix.ready = false` (`0` scientific units) because `E9` blocked `E2`, leaving no `R`/`F` endpoints or critic receipts.

## Why
Rolling prefix branches all the way to `H = 5,000` across `8` prefixes and `2` confirmation streams requires `24.96M–37.76M` world-steps (`5.7–8.7` L4 hours vs initial `4 u` / `20 u` estimates extrapolated from tiny fixtures), and `E6` scientifically depended on `E2` candidate checkpoints that `E9` prevented from running.

## What to reuse / what not to repeat
- Shift new development seed families into high 64-bit ranges (`+2**40`) with an ordinal-index roster wrapper around `strict_promotion.materialize_rosters` to avoid 32-bit historical seed collisions without editing frozen evaluation code.
- Multiply conditioned late-death rescue gains by natural root prevalence `q_m` and debit ordinary-world collateral `C` (`E10` two-phase bridge) rather than quoting conditioned root lifts as population H5000 gains.
- Size H5000 continuation caps from full-horizon `production_preflight.py` fixtures (`24.96M–37.76M` steps), not short-horizon smokes.

## Full record
Original: `git show 90b3db94:docs/research/history/bl-prefix-e6-e7-e10.md` | Primary sources: [bl_prefix/README.md](../../../research/bl_prefix/README.md), [REGISTRATION_E10.md](../../../research/bl_prefix/REGISTRATION_E10.md), [CAP_NOTE.md](../../../research/bl_prefix/CAP_NOTE.md)
