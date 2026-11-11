# Solo-Food PQN Warmstart, Imitation-to-RL Handoff, Horizon Extension (H64–H768), and One-Opponent Studies (AN–DA and Enemy-Dose)

**Dates:** 2026-09-14 to 2026-09-21
**Verdict:** Across `37` studies (`AN`–`DA` + `enemy-dose`), Q-output scaling (`0.1`), an annealed `CE=1` behavior anchor, own-body time-to-vacate (`TTL`) planes, unclipped `half_mse` loss, and explicit body-length scalars unlocked `3/3`-seed solo PQN competence through `768`-step `H768` (`CW`: `37/37` checks), while `1`-opponent `S2` paired learning passed in only `1/3` seeds (`CZ`/`enemy-dose`).

## What was tried
- Tested `2,598`-param PointNet vs `1,435,751`-param food-only raster BC (`AN`), `32` unanchored PQN updates (`AO`), `8`-update value-offset (`BV`), action-margin pressure (`BW`), Q-head scale `tau = 0.1` (`BX`), `CE=1` behavior anchors annealed over updates `9–32` (`AP`, `BY`, `BZ`, `CA`), food reward `1.0` vs `0.1` (`CB`), and `1,024`–`2,048` PQN updates (`CC`, `CG`).
- Extended solo horizons (`H128` -> `H256` -> `H384` -> `H512` -> `H768`) on `384`-lane (`32`-world) fresh banks by adding own-body `TTL` (`CK`), auditing `lambda` / Huber terminal clipping (`CL`–`CN`), switching to unclipped `half_mse` (`0.5 * MSE`, `CO`–`CU`), and exposing `length / 150` + `log1p(length) / log1p(150)` (`CV`, `CW`).
- Evaluated zero-shot `S2` (`1` `GreedyFood` opponent at `H256`, `CX`/`CY`) and trained `512` + `1,024` `S2` updates (`4096 -> 5632`) with tactical enemy channels `1–3` (`CZ`, `DA`, `enemy-dose`).

## What happened
- **BC-to-PQN handoff (`AN`–`CG`):** Unanchored `32`-update PQN collapsed `H64` food (`11.44–11.74 -> 0.88–7.74` in `AO`; `90.5–94.8` margin-eroding mass in `BW`), whereas `tau = 0.1` + annealed `CE=1` + `food01 = 0.1` (`1.0` fell to `5.29–5.81` in `CB`) recovered at mark `1024` (`11.73–12.15` in `CC`) and passed `17/17` `H128` checks in **`3/3` seeds** (`CG`: `357–370 / 384` alive).
- **`H256` -> `H384` (`CI`–`CR`):** Food-only PQN self-trapped at `H256` (`258–310 / 384`, `607/607` `DEATH_SELF` in `CI`/`CJ`); adding own-body `TTL` + unclipped `half_mse` passed **`3/3` seeds at `H256` (`CO`: `368 / 376 / 374`) and `H384` (`CR`: `348 / 366 / 381`)**.
- **`H512` -> `H768` (`CV`–`CW`, `3/3` PASS):** Adding explicit length scalars passed **`3/3` seeds at `H512` (`CV`: `380 / 380 / 384`) and `H768` (`CW`: `353 / 364 / 383` alive, `122.51 / 116.85 / 135.34` food vs teacher `94.52`, `37/37` checks)**.
- **`1`-opponent `S2` (`CX`–`DA`, `1/3` paired PASS):** `70/76` zero-shot `S2` deaths were head collisions (`CY`); enemy channels `1–3` passed all paired `S2` checks in `1/3` seeds (`6102`/`7202`: `373 -> 376 / 384`), with `19–32` deaths/run (`~0.02%` of `131k` rows) carrying `48.1–51.8%` of squared TD residual (`DA`).

## Why
Uncalibrated BC logits suffer greedy-action margin erosion under initial TD updates unless scaled (`0.1`) and anchored, Smooth-L1 (`Huber`) clipped `100%` of rare (`0.08–0.10%`) terminal self-trap residuals, and saturated grids needed own-body `TTL` and global length to time clearance (`DA` reports `S2` residual shares without claiming an optimizer mechanism).

## What to reuse / what not to repeat
- Calibrate Q scale (`0.1`) with a temporary annealed `CE` anchor and unclipped squared loss when warm-starting TD from supervised policies.
- Include per-cell own-body clearance time (`TTL`) and explicit body-length scalars in raster observations for long-horizon (`H256`–`H768`) survival.

## Full record
Original: `git show 90b3db94:docs/research/history/solo-food-pqn-handoff.md` | Primary sources: [SOLO_FOOD_SERIES.md](../../experiments/SOLO_FOOD_SERIES.md), [behavioral-horizons.png](../../experiments/solo_food_pqn_h768_length_confirmation_2026-09-21/behavioral-horizons.png), [margin-pressure.png](../../experiments/solo_food_pqn_margin_pressure_2026-09-20/margin-pressure.png)
