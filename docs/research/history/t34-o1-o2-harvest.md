# T34 Offense-vs-Defense Audit, O1 Offensive Harvest, and O2 Bank (prepared, never run)

**Dates:** 2026-10-20 to 2026-10-22
**Verdict:** The 36-world food-economy audit refuted the offense-pivot premise (`b8-s3` already took `39.71%` of gross mass from death drops under a `150`-pellet cap), `O1` passed `GO` locally (`35/48` worlds had a `>= +5` death-drop H500 branch), and closed-loop `O2` (`1,260`-world fallback vs `5,097` needed for `.80` power) closed `PARKED` / `UNUSED` without running a study world.

## What was tried
- Audited mechanics-v2 corpse rules and ran `36` instrumented H5000 terminal-hero `b8-s3` worlds (`12`/mix, `164,514` acted frames) tracking ambient, death-drop, and boost-trail intake to test whether offensive kill/corpse harvesting had larger upside than defensive survival search.
- Ran **O1** (`research/t34_o1_harvest`): a `48`-world (`16`/mix, `240` roots, `1,306` branches) L4 counterfactual census testing whether single-action H500 deviations capture `>= +5` death-drop mass (KILL bar `< 8/48`, frozen+mixed bar `>= 4`), after fixing timing-fixture roots (`10, 20`) when initial timing seeds died at `4479` and `2216` steps.
- Prepared **O2** (`research/t34_o2_harvest`): a closed-loop H5000 comparison of bare `b8-s3`, rescue-only `D`, and rescue+harvest `D+O` (`(D+O - D)` `LB90 > 0` and `>= +40` vs B8), resizing from `96` to `1,260` worlds (`420/mix`, `149.733333 u` cap) and repairing upload/CUDA-mask staging (`V3`).

## What happened
- **36-world `b8-s3` economy audit:** Bare `b8-s3` earned `332` kills (`18.083`/game scripted, `4.083` frozen, `5.500` mixed) with `0.125%` (`206`) boosted frames and **`39.71%` (`16,467`) of gross mass from death drops** (`50.53%` / `20,952` ambient, `4,049` trails). Mechanics v2 caps corpse+trail pellets at **`150`** (`1` pickup/tick; all `9` deaths at `L >= 800`, up to `1,454`, kept `150`).
- **`O1` harvest census (`GO`):** **`35/48` worlds** (`13` scripted, `9` frozen, `13` mixed; `22` frozen+mixed; `71/240` roots) had a surviving single-action H500 branch gaining `>= +5` death-drop mass (`parity_pass: true`).
- **`O2` closed-loop bank (`PARKED` / `UNUSED`):** Measured T23 SDs required **`5,097` worlds** for `.80` (`80%`) joint power at `+10` incremental mass; the `1,260`-world budget fallback had `.339888` capped power, hit upload (`1,195,740` B -> `16,655` B URL fix) and `CUDA_VISIBLE_DEVICES=""` bootstrap stops, refused chunk `00` on its L4 timing forecast, and closed unrun.

## Why
Keeping a long body alive dominates the total-horizon mass integral while the `150`-pellet FIFO cap and one-pickup-per-tick ceiling bound corpse-harvesting upside; detecting a `+10` incremental closed-loop mass lift over `D` against cross-world variance required `5,097` paired worlds (`O2`).

## What to reuse / what not to repeat
- Measure baseline food provenance (`39.71%` death drops) and simulator caps (`150` pellets) before assuming an unexploited gameplay economy, and apply `CUDA_VISIBLE_DEVICES=""` after GPU `bootstrap.sh` checks (`16,655` B URL catalog vs `1,214,804` B CLI upload).
- Ensure excluded timing-fixture roots (`10, 20`) lie well inside minimum trajectory survival rather than fixed late steps (`3000`) where a timing seed may already be dead (`2216`).

## Full record
Original: `git show 90b3db94:docs/research/history/t34-o1-o2-harvest.md` | Primary sources: [2026-10-21_offense_vs_defense/JUDGMENT.md](../advice/2026-10-21_offense_vs_defense/JUDGMENT.md), [PREREGISTRATION_O1.md](../../../research/t34_o1_harvest/PREREGISTRATION_O1.md), [t34_o2_harvest/README.md](../../../research/t34_o2_harvest/README.md)
