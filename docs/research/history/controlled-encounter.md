# Controlled Encounter (2026-09-21 — 2026-09-22)

**Dates:** 2026-09-21 to 2026-09-22
**Verdict:** Closed: `256`-world empirical-teacher behavioral cloning reached `89.6–93.8%` held survival and `220–224` held food across `3/3` seeds (`3/3 RELATIVE PASS, 0/3 ABSOLUTE PACKAGE`), whereas native PQN $Q(\lambda)$ continuation failed its competence/retention gates (`LINE CLOSED (0/3 PASS)`).

## What was tried
- Built a `4`-frame (`H4`) `2`-snake tactical encounter benchmark (`576` cases across `6` families × `4` headings: `384` train on `16` worlds, `192` held on `8` worlds; teacher `576/576` alive, `228` held food) on `RasterDuelingNetwork` (`raster31v2`).
- Replaced off-crop-aliased clearance tie-breaks (`2` alias groups) with food-only `H4` sets, aggregated `1,040` learner-visited prefix states (`DAgger`), and weighted step-`1..4` food by **`8/4/2/1`** to break procrastination ties.
- Expanded training breadth from `16` -> `64` (`7,184` rows) -> `256` worlds (`25,616` rows reconciled under a finite-bank empirical teacher, `expand256` vs `repeat64` from mark `512 -> 1536`), auditing hidden food-respawn RNG (`180` cells) and L2/mark-`0` controls.
- Warm-started native PQN $Q(\lambda)$ from `expand256` (`lr=1e-4`, collector `cap=256 -> 16` in `SHORT64` -> `D256`), ran `1,152` `1`-action rescue rollouts, and tested crossing NLL rehearsal, direction-margin hinges, a `1,024`-update cycle, and `0.1` ambient bonuses.

## What happened
- **Procrastination tie-break (`8/4/2/1`):** Weighting earlier food (`8/4/2/1`) eliminated **`87/87` (`100%`)** procrastination loops and lifted teacher food from `383 -> 480` (train) and `221 -> 256` (held).
- **`64 -> 256`-world BC (`3/3 RELATIVE PASS, 0/3 ABSOLUTE PACKAGE`):** Expanding `64` (`repeat64`) to `256` worlds (`expand256`) lifted held food from **`152, 144, 160` to `220, 224, 220`** (`+0.3542, +0.4167, +0.3125`/case), fit to `93.23%, 95.83%, 93.75%`, and survival to **`89.6–93.8%` (`180/192, 176/192, 172/192`)** across `3/3` seeds (`0%` train coverage on all `48` fatal held games).
- **Collector horizon (`SHORT64`):** Cutting collector cap `256 -> 16` raised visited worlds from `51–63` to `206–207` (`1,069–1,106` `H16` food).
- **PQN continuation (`LINE CLOSED 0/3 PASS`):** Continuing to `D256` traded crossing for frontal survival (`28/28` crossing losses diverged at even steps `2`/`4`, `16/16` frontal rescues at odd steps `1`/`3`, `100%` reversed by `1`-step parent restore); rehearsal, hinges, `1,024`-cycle (`889–1,034` food), and ambient bonus all failed (`0/3`).

## Why
Unweighted `H4` ties allowed procrastination (`87/87` fixed by `8/4/2/1`), and `16`–`64` worlds left unseen states (`0%` train coverage on all `48` `expand256` deaths); for PQN's trade-off (`934` lost vs `40` gained rows, `3.00–3.63` terminal vs `0.25–0.27` interior TD residuals), `1`-action rescue proved single-action sufficiency in `44` selected cases, not an optimizer mechanism.

## What to reuse / what not to repeat
- Weight earlier rewards (`8/4/2/1`), reconcile hidden off-crop/RNG label aliases empirically, and scale unique training worlds (`256` vs repeating `64`) plus short collector caps (`cap=16` vs `256`).
- Do not expect unanchored or NLL/hinge-rehearsed PQN $Q(\lambda)$ continuation (`0/3` PASS, negative gradient cosines `-0.1746, -0.1073`) to preserve tactical crossing margins.

## Full record
Original: `git show 90b3db94:docs/research/history/controlled-encounter.md` | Primary sources: [README.md](../../experiments/controlled_encounter_common_path_2026-09-21/README.md), [README.md](../../experiments/controlled_encounter_hidden_rng_2026-09-21/README.md), [README.md](../../experiments/controlled_encounter_residual_policy_diagnostic_2026-09-21/README.md)
