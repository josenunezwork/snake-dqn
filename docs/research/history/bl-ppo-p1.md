# Big-Lever E4/E2 PPO Pilot, WP-3 P1 100M PPO Endpoint, Y3 Critic Qualification, and A3 Learner Ceiling

**Dates:** 2026-10-20 to 2026-10-23
**Verdict:** Power gate `E9` blocked the two-arm `E2` pilot (`.27045`–`.30700` vs `.60` required), `E8` passed late-body critic qualification (`EV = 0.5867` via float64 Helmert least-squares, certifying value-head init only), `A3` kept `P1` on L4 (`0.98 s` learner vs `~9.3–9.6 s` host-CPU collection), and `Y3` stopped at `HOLD_MISSING_OPERATIONAL_PROOF`.

## What was tried
- Ran `E9` operating-characteristic simulations (`20,000` and `100,000` replicates using T23 paired H5000 mass SDs `364.683` scripted, `262.732` frozen, `212.360` mixed) to test whether two-arm `E2` (`GRU256` `R` vs feedforward `F`) met `>= .60` joint funding power at `+40` mass and `<= .10` null pass.
- Registered single-endpoint `WP-3 P1` (`100,000,000` hero transitions from `b8-s3`, `128` worlds × `128` steps, `12` L4 CPU collectors), gating actor unfreeze on `E8` held-out late-body (`L >= 800`) PopArt critic qualification (`EV >= .1` AND `MSE < baseline` using frozen B8 trunk + remaining-horizon clock).
- Qualified `Z-9` public-catalog HTTPS checkpoint staging (`2,409 B` catalog vs `30,720,004 B` direct uploads) and `A3` (`16,384`-transition 2-epoch PPO learner ceiling `<= 1.6384 s` on A100-40GB vs L4).

## What happened
- **`E9` power gate (`E2 BLOCKED`):** Full funding probability at `+40` was `.27045` (`96/32/32`) and `.30700` (`128/48/48`) vs `.60` required (null pass `.05050` / `.06160`); none of three `Z-4` replacements reached `.60` (best `.36216` at `192/64/64`).
- **`Z-9` upload & `A3` learner (`keep P1 on L4`):** Direct 6-checkpoint upload (`~690.6 s`) exceeded `300 s`, whereas the `2,409 B` catalog + HTTPS fetch took `65.41 s` (`242.11 s` worst case). Baseline `A3` missed `1.6384 s` (`1.87 s` A100 / `1.81 s` L4); optimized learner passed only on L4 (`0.98 s`; A100 `0.41 s` `REJECT_UNEXPLAINED_GRADIENT_CHANGE`) under `~9.3–9.6 s` CPU collection.
- **`E8` critic (`PASS`):** Online one-pass Adam (`source-02`) failed; offline `D1` converged to `EV = 0.5867 >= .10` and `MSE = 307,567.7 < 561,561.6` (`162` rows), and CPU float64 least-squares on a `256 × 255` Helmert basis (`rank 257`, `source-latebody-06`) passed critic qualification (`427,911,156 B` origin snapshot, certifying value-head init only).
- **`Y3` continuation (`HOLD_MISSING_OPERATIONAL_PROOF`):** Continuation stopped at `49,152` actor transitions (`3` updates) before the first `900 s` periodic capture, leaving no actor operational snapshot (repair `-05` `PREPARED_HELD`).

## Why
Cross-episode H5000 variance (`SD` `212.360`–`364.683`) limits 96–192-world-per-mix two-arm screens to `.27–.36` joint power at `+40`; in `E8`, frozen B8 features plus remaining clock linearly predict late returns (`EV 0.5867`), which one-pass Adam under-fit until replaced by exact float64 least squares; in `A3`, host-CPU rollout collection (`~9.3–9.6 s`) dwarfed GPU learning (`0.98 s`).

## What to reuse / what not to repeat
- Fit linear PopArt value warm-starts (frozen trunk + remaining-horizon clock) by exact float64 least squares (`EV 0.5867`) rather than a single under-converged Adam pass.
- Stage multi-checkpoint rosters via a tiny hash-pinned catalog (`2,409 B`) + HTTPS fetch (`bl_upload`) instead of uploading `30 MB` over CLI RPC (`~690.6 s` vs `300 s` cap).
- Keep CPU-collection-bound PPO runs (`~9.3–9.6 s` collection vs `0.98 s` learner) on L4 rather than A100, and check multi-arm gate power (`E9`) before dispatch.

## Full record
Original: `git show 90b3db94:docs/research/history/bl-ppo-p1.md` | Primary sources: [REGISTRATION_P1.md](../../../research/bl_ppo/REGISTRATION_P1.md), [E9_RULE.md](../../../research/bl_ppo/E9_RULE.md), [a3_p1_ceiling/README.md](../../../research/a3_p1_ceiling/README.md)
