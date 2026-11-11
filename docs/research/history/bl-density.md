# Big-Lever DENSITY-1 and DENSITY-2 Synthetic Arena Shrink vs Natural Doom-Root Yield (`bl-density`)

**Dates:** 2026-10-21 to 2026-10-22
**Verdict:** Closed as `TERMINAL / REJECTED`: across `492` DENSITY-2 episodes, all nine smaller-map and extra-opponent cells were rejected under the minimum-map realism rule (identity `12/12`), closing synthetic arena densification.

## What was tried
- Tested whether synthetic densification raises length `L >= 800` H300 doom-root yield over natural arm `A` (`145 × 83` map, five opponents): `B70m` (`121x69` map, five opponents), `B70p` (`121x69` map, seven opponents; `B50` dropped after Round 14), and `C-P1` (injected long serpentine bodies on `145 × 83`).
- Coarsened root sampling stride from `50` to `200` (`24` roots/episode) after Mac preflight measured `2.184720` CPU-s per `L800` H300 branch, and enforced Round-14 kill rules (`REJECTED` if yield `< 2*A`, avoidable fraction `< 0.5*A`, 35-feature logistic classifier AUC `>= 0.80`, or a minimum-map violation `A_min(L) = max(8*L_hero, U_total/rho_95, 8349)`).
- Ran the `90`-episode `DENSITY-1` pilot (`36` `A`, `18` `B70m`, `18` `B70p`, `18` `C-P1`) and scaled to `492`-episode `DENSITY-2` replication (`240` `A`, `72` `B70m`, `108` `B70p`, `72` `C-P1`) on ten single-thread L4 CPU workers with outcome-blind resume completion.

## What happened
- **`DENSITY-1` 90-episode pilot (`4,532` CPU-s, `INSUFFICIENT` natural denominator):** Natural `Arm A` yielded only `5/36` doom-bearing episodes (`1/12` frozen, `2/12` mixed, `2/12` scripted) and `0, 1, 1` (`2` total) witnessed-avoidable roots, vs `5/18` (`B70m`), `3/18` (`B70p`), and `11/18` (`C-P1`) doom-bearing episodes (`27/29` pilot doom roots were `DEATH_SELF`, median length `1186`).
- **`DENSITY-2` production & repins:** Initial L4 run verified `408/492` episodes (`4.90 u` of `16.50 u` series cap) before hitting a post-work telemetry deadline, leaving `84` primary + `12` Arm-A calibration units prepared across transport repins `02`–`04` (fixing the L4 family-canary gate to require CUDA observations only when CUDA is declared).
- **Terminal verdict (`REJECTED`):** All nine smaller-map/more-opponent cells across `DENSITY-2` violated the minimum-map rule and were `REJECTED` (local sparse-archive identity `12/12`), closing the density lever.

## Why
Shrinking the arena (`121x69`) or adding opponents (`7` vs `5`) increased raw doom-root encounters in pilot cells, but every densified cell violated the minimum-map area constraint `A_min(L) = max(8*L_hero, U_total/rho_95, 8349)` required to preserve natural `145 × 83` late-body coiling geometry.

## What to reuse / what not to repeat
- Exempt CPU-only L4 multi-worker manifests from requiring CUDA action observations in family-canary checks unless the job declares CUDA.
- Measure single-branch H300 CPU cost (`2.184720` CPU-s at `L800`) before sizing dense-stride root sweeps (`2,412,000` decisions exceeded a `90`-minute L4 session until stride moved `50 -> 200`).
- Do not shrink map dimensions or crowd extra snakes to harvest `L >= 800` trap states faster; both violate late-body geometry realism.

## Full record
Original: `git show 90b3db94:docs/research/history/bl-density.md` | Primary sources: [bl_density/README.md](../../../research/bl_density/README.md), [METRICS_AMENDMENT_04.md](../../../research/bl_density/METRICS_AMENDMENT_04.md), [COMPLETION_REPIN_04.md](../../../research/bl_density/COMPLETION_REPIN_04.md)
