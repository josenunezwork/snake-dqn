# Native Solo Behavioral Cloning, Spacious Teacher, and Learner-State Aggregation (`solo-native-imitation`, Series `X`–`AM`)

**Dates:** 2026-09-13 to 2026-09-14
**Verdict:** Food-geometry behavioral cloning (`AA`) confirmed `64`- and `128`-frame native solo gameplay (`90`–`96` of `96` alive) despite missing the `>= 95%` held-out fit gate (`88.623%` peak), whereas `256`-frame survival failed (`55`–`81` of `96` vs `>= 87/96` bar) across spacious-teacher BC (`AG`), two learner-state aggregation rounds (`AI`, `AJ`), and food-view auxiliaries (`AL`).

## What was tried
- Trained `500`-update `RasterDuelingNetwork` (`raster31v3`) behavioral cloning (`BC`) on `3,072` `64`-frame native `145 x 83` `GreedyFood` rows across `4` worlds (`X`), `16` worlds (`Y`), and food-geometry-only inputs (`AA`), plus an observation-only decoder probe on `12,288` (`X`/`Y`/`W`) and `10,775` (`AK`) saved rows.
- Evaluated `AA`'s `3` fixed policies zero-shot on fresh `64`-frame, `128`-frame, and `256`-frame (`AD`) `8`-world (`96`-lane) banks and replayed all `100` `256`-frame deaths.
- Calibrated a flood-fill `SpaceTeacher` (`96/96` `256`-frame alive, `46.375` food vs `GreedyFood` `67/96`, `41.729`) and ran `256`-frame BC (`AG`), teacher-veto miss replay (`AH`), two DAgger-style `1,536 + 1,536` learner-state aggregation rounds (`AI`, `AJ`), and food-view auxiliary/inference checks (`AL`, `AM`).

## What happened
- **`64`/`128`-frame success vs `256`-frame traps (`AA` -> `AD`):** `AA` missed `>= 95%` fit (`87.093%`–`88.623%`, decoder **`12,288 / 12,288`** and **`10,775 / 10,775`**), yet passed **`3/3` gates at `64` (`94–96 / 96` alive) and `128` frames (`90–91 / 96`)** before falling to **`55`, `79`, `66` of `96` at `256` frames** (`AD`, `100/100` fatal moves had `0` safe exits).
- **Spacious-teacher BC & aggregation (`AG`–`AM`, `FAIL`):** `AG` survived `71/96, 81/96, 56/96`, missing **`73.5%`–`86.5%`** of vetoes (`AH`: `97.3%`–`100%` veto error before death). Aggregation lifted survival (`AI`: `55→70, 66→73, 58→71`, `+12.15 pp`; `AJ`: `61→71, 65→72, 63→65`, `+6.60 pp`), but disagreement fit stayed `20.0%`–`51.1%` (`< 87/96` gate), and `AL` lost `-9.72 pp`.

## Why
All `100` `256`-frame deaths were sealed before the final frame (`0` advisory-safe fatal moves), and whole-raster BC overfit rare spacious-teacher veto states (`100%` train fit vs `20.0%`–`58.3%` held disagreement fit across `AG`, `AI`, `AJ`; the reports note this association without claiming counterfactual rescue proof).

## What to reuse / what not to repeat
- Separate closed-loop gameplay gates from exact teacher-action agreement (`AA` passed `64`/`128`-frame gameplay at `87.093%`–`88.623%` held fit).
- Do not rely on whole-raster `3,072`-row BC or `1,536`-row DAgger refreshes to generalize rare (`~4%`) free-space veto overrides (`20%–51%` held veto fit).

## Full record
Original: `git show 90b3db94:docs/research/history/solo-native-imitation.md` | Primary sources: [SOLO_FOOD_SERIES.md](../../experiments/SOLO_FOOD_SERIES.md), [README.md](../../experiments/README.md), [redesign-and-b8.md](../../notes/redesign-and-b8.md)
