# Solo-Food Safety Residual, Gate, and Free-Space Wrapper Series (Studies AT–BU)

**Dates:** 2026-09-19 to 2026-09-20
**Verdict:** Across `28` studies (`AT`–`BU`), a `2D` CNN safety head overfit (`21–32%` held fit), whereas a shared `3`-input per-action safety head (`1,500` updates, ranking hinge) generalized at `97.22%–98.57%` held fit and reached `96/96` (`100%`) solo survival across `3/3` seeds through `512`-step `H512` (`BO`), while food-parent fine-tunes (`BR`–`BU`) failed (`0/3`).

## What was tried
- Augmented frozen `FoodGeometryNetwork` parents (`AN`, `raster31v3`) with `2D` CNN body/wall/reachability safety residuals (`AT`–`BF r2`), testing KL preservation (`AW`), `16`-world coverage (`AYr1`), full-raster context (`AZ`), `3` flood-fill scalars (`BA r1`, `BB`), and disagreement decomposition (`BC`).
- Built a **parent-consistent safety oracle** `O` overriding the parent only when its normal choice was unroomy (`BD r1`, `BE`), then audited exact-view feasibility (`AX`, `BG`, `BI`).
- Replaced the CNN with a shared per-action `Linear(3,16)`–ReLU–`Linear(16,1)` head on `[deficit[i], threshold/cap, boost]`, replacing infeasible zero-KL/square penalties (`BH`, `BI`) with a capped-margin ranking hinge over `500` -> `1,500` updates (`BJ`–`BM`) and testing zero-shot `H256` (`BN`) and `H512` (`BO`).
- Audited remaining `>=128`-frame stalls (`BP`, `BQ`, `BS`) and tested `500`-update parent food fine-tunes (`1,536` own-policy rows `BR`, `16`-world breadth `BT`, CoordConv planes `BU`).

## What happened
- **CNN overfitting (`AT`–`BF r2`):** `2D` CNN safety residuals hit `100%` train fit but collapsed to **`21–32%` (`21.43%–31.82%`) held fit** (`BF r2`; `0, 0, 1` exact byte overlaps in `AX`), while `>50%` of raw `SpaceTeacher` diffs in `BC` were non-safety ties removed by oracle `O`.
- **Shared `3→16→1` head (`96/96` `H512` in `3/3` seeds):** On `138–146` distinct triples across `3,072` rows (`1,500` hinge updates), held fit hit **`97.22%, 97.78%, 98.57%`** (`99.93%–100%` ordinary keep in `BL`/`BM`) with **`96/96` (`100%`) survival in `3/3` seeds at `H128`, `H256`, and `H512`** (`BO`: `0` deaths vs `66, 75, 75`; food `77.01, 77.73, 77.93` vs `55.08, 52.75, 54.57`).
- **Food-stall fine-tunes (`BR`–`BU`, `FAIL 0/3`):** All `8` zero-food slice-lanes were inherited roomy parent trajectories (`0` safety overrides in `BP`/`BQ`), and `BR`, `BT`, and `BU` stalled at `86.65%–89.13% < 95%` held accuracy (`0/3` seeds).

## Why
Spatial CNN residuals memorized singleton body grids (`0, 0, 1` overlaps), and zero-KL/zero-square preservation formed negative Bellman–Ford cycles (`X` in `BI`) when identical `[deficit, threshold, boost]` triples appeared in both ordinary and veto states; the `3`-scalar shared head + ranking hinge removed both failure modes.

## What to reuse / what not to repeat
- Route invariant per-action safety quantities (`[deficit[i], threshold/cap, boost]`) through compact shared heads trained with action-ranking hinges rather than `2D` CNN residuals or zero-KL penalties.
- Strip non-safety teacher tie-break disagreements (`>50%` of `SpaceTeacher` diffs in `BC`) before supervising a safety override head.

## Full record
Original: `git show 90b3db94:docs/research/history/solo-food-safety-residual.md` | Primary sources: [experiment_portfolio_review_2026-09-26.md](../experiment_portfolio_review_2026-09-26.md)
