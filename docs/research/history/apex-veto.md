# Apex vector61 Serving-Time Free-Space Veto Progression (v1–v8) and Early October Diagnostics

**Dates:** 2026-09-26 to 2026-10-03
**Verdict:** Four sequential deterministic free-space wrappers (`v2 -> v5 -> v7 -> v8`) around frozen `champion_a5` each cleared `STRICT_PASS` and `SERVING_PASS`, lifting `5,000`-frame (`H5000`) mass `6–7×` (`~49–62` to `~337–369`) while overriding `0.20%–0.29%` of moves at `0.65 ms`/decision (`v3`, `v4`, `v6`, `v9` retired).

## What was tried
- Built speed-preserving one-step flood-fill veto `v2` (`need = min(length, 160)`) around `champion_a5_freespace_20260621.pth` (`43d4e2c5`, where `85` of `96` pilot deaths were self-collisions) and ran Tier-2 strict (`N = 235`/mix) and `50`-episode web serving gates.
- Tested tail-aware one-step `v3` and depth-`8` lookahead `v4`, then ran a depth-`40` (`50,000`-node) point-of-no-return (PNR) diagnostic (`trap_horizon_20261001`) that motivated two-step boost-landing veto `v5` (`N = 269`/mix strict gate).
- Audited `v5` deaths (`77.6%` self enclosures sealed `20–60` frames early, `7.8%` head-ons), screened opponent-head `v6`, and tested tail-aware space-preference re-ranking `v7` (`lambda = 4`, `N = 243`/mix) and space + opponent-head `v8` (`lambda = 8`, 4-look OBF gate, `N_max = 249`/mix).
- Ran a `180`-episode `v8` death census (`60` worlds/mix, `24/24` SIMD-live bit-exact) and benchmarked `40`-step rollout prototype `v9`.

## What happened
- **First two promoted upgrades (`v2`, `v5`):** `v2` gained `+76.2` / `+73.3` / `+82.6` (`49.2->125.4`, `47.7->121.0`, `61.8->144.5`); `v5` gained `+71.6` / `+63.3` / `+62.9` while changing `0.03%` of decisions (`STRICT_PASS` & `SERVING_PASS`).
- **Space-preference upgrades (`v7`, `v8`):** `v7(4)` gained `+130.3` / `+119.1` / `+123.0` (`0.58 ms`); `v8(8)` stopped `STRICT_PASS` at look `2` (`187/249` worlds/mix = `561` worlds) with `+30.1` / `+43.0` / `+36.5` (`337.2` frozen, `355.5` scripted, `368.6` mixed; survival `~0.83–0.92`, **`0.65 ms`/decision**).
- **Retired variants (`NOT_ADVANCED`):** `v3` changed `4/120` worlds; `v4` passed `0` mixes (`694` no-escape fallbacks); `v6` cut head-ons `10 -> 2` (`+30.0` scripted) but missed frozen/mixed (`+10.0`, `+13.4`); `v9` cost `17–37 ms` worst-case.
- **PNR & `v8` censuses:** Under `v2`, `74%` (`31/42`) of self deaths had PNR `<= 8` frames (`24/31` boost pockets, `0/10` after `v5`). Under `v8`, survival hit `61.7%` (`111/180`, `337.4` mass, `0.6%` head-on), with `75%` of remaining death headroom from multi-frame opponent encirclement (`+60` to `+91` mass ceiling).

## Why
`champion_a5` ignored its own `3` `160`-capped free-space inputs on rare trap entries (`0.20%–0.29%` of ticks) and two-cell boost landings; once `v8` fixed local traps and head-ons, static single-snake vetoes hit a ceiling against `20–60`-frame multi-snake encirclements.

## What to reuse / what not to repeat
- Keep `v8`, `v7`, `v5`, and `v2` as serving rollback wrappers (`SNAKE_SERVE_CHECKPOINT=champion`, `SNAKE_SERVE_VETO_VARIANT=v8|v7|v5|v2`), and keep the `30` `v8` safety scalars embedded in `ego2s-bv8` (`b8-s3`).
- Always rebind `AISnake._get_frame` closures when cloning `GameState` (`copy.deepcopy` bug caught in `v9`), and run Mac strict gates on AC with lid open (`run-v1`/`v2` battery sleep `INVALID_STOP`s).

## Full record
Original: `git show 90b3db94:docs/research/history/apex-veto.md` | Primary sources: [apex_veto_v8_2026-10-03/README.md](../../experiments/apex_veto_v8_2026-10-03/README.md), [trap_horizon_2026-10-01.md](../trap_horizon_2026-10-01.md), [serving_time_search_feasibility_2026-10-03.md](../serving_time_search_feasibility_2026-10-03.md)
