# Task-Aligned CZ `enemy_visible` Challenger & 6102 Continuation Series (2026-09-24 to 2026-09-26)

**Dates:** 2026-09-24 to 2026-09-26
**Verdict:** Retired (`2026-09-26`): task-aligned PQN continuation (`raster31v3`, `(9, 31, 31)` tactical grid) lifted `1/3` lineages (`2026096102`, `+37.9` to `+39.876` `H3000` mass over its parent), further continuation regressed solo mass (`-35.152`), and `6102` trailed incumbent `vector61` Apex by `-30.2` to `-66.4` mass at `H5000` while requiring `N = 8,676 > 120` pilot worlds.

## What was tried
- Continued all `3` CZ `enemy_visible` update-`4,608` PQN `raster31v3` parents (`2026096101`, `2026096102`, `2026096103`) for `31,005` mixed `[solo, s2, s2, s6, s6]` `H1024` rollouts (`23,776,196` hero transitions to update `35,613`, `Q scale 0.1`), evaluated on `32` paired `H3000` worlds × `3` profiles.
- Continued nominated challenger `2026096102` another `31,005` rollouts (`35,613 -> 66,618`, `7,933,551` transitions) against update `35,613`.
- Qualified the `6102` live serving bridge across `v1`–`v5` (`63`-hex literal fix, `(9, 31, 31)` tuple/JSON list fix, `monotonic` heartbeat fix, pytest `9.0.1` `subTest` count `82` vs `75` audit fix -> `AUDIT_ONLY_PASS`).
- Ran strict challenge pilot `pilot-v1` (`16` Apex dev + `16` paired `H5000` worlds/mix on `promotion-v2-watch-rect`, `720,000` frames) against `champion_a5_freespace_20260621.pth`.

## What happened
- **3-lineage `H3000` screen (`INCONCLUSIVE_SCREEN`, `6102` nominated):** Only `2026096102` cleared every gain and solo-retention bar (**`+37.946` (`+37.9`) solo, `+39.876` `food_pressure`, `+38.104` `mixed` mass**); `2026096101` (`-2.093` solo) and `2026096103` (`-17.199` solo, `+4.343` mixed) missed retention/gain bars.
- **Further `6102` continuation (`NEGATIVE_CONTINUATION_SCREEN`):** From `35,613 -> 66,618`, solo living mass fell **`-35.152`** (`78.750 -> 43.598`; `-5.067` `food_pressure`, `-7.299` `mixed`) even as food (`470.9 -> 660.1`), alive fraction (`0.806 -> 0.946`), and boost (`0.362 -> 0.646`) rose.
- **`H5000` strict pilot vs Apex (`PILOT_STOP_INFEASIBLE_AT_FIXED_MDE` / `INVALID_STOP_AUDIT_FAILED`, RETIRED):** Unaudited `6102` trailed Apex across all `3` mixes (**`-30.2` frozen, `-66.4` scripted, `-34.9` mixed** mass, winning `5/16` worlds each), and `10%`-MDE variance (`SD 55.8–118.3`) required **`N = 8,676 > 120`** worlds.

## Why
Task-aligned PQN on `raster31v3` remained far below `vector61` Apex at `H5000` (`-30.2` to `-66.4`), and `10%`-MDE gating on noisy `H5000` trajectories (`1–3` noise-to-mean ratio) needed `552–8,676` worlds/mix; for `66,618`'s `-35.152` mass drop alongside `0.555–0.646` boost, boost was not intervened on so **cause was not established**.

## What to reuse / what not to repeat
- Size `H5000` confirmation gates for thousands of fast vectorized episodes (`N = 552–8,676` at `10%` MDE) rather than `120` live-sim worlds.
- Normalize JSON round-trip shapes (`list` vs `tuple`), monotonic heartbeats, and pytest `subTest` counts before running strict auditors (`v1`–`v5` bridge stops).

## Full record
Original: `git show 90b3db94:docs/research/history/task-aligned-challenger.md` | Primary sources: [README.md](../../experiments/task_aligned_cz_integration_2026-09-24/README.md), [README.md](../../experiments/task_aligned_continuation_2026-09-24/README.md), [experiment_portfolio_review_2026-09-26.md](../experiment_portfolio_review_2026-09-26.md)
