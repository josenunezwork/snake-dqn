# FRP fine-tune family v2–v5 on vector61/vector67 (frp3-s12 + v8 Release)

**Dates:** 2026-10-03 to 2026-10-20
**Verdict:** Fine-tuning `vector61` Apex with in-loop `v8` shielding and scripted anchors produced `frp3-s12 + v8` (`STRICT_PASS`, `SERVING_PASS`, released `2026-10-06`), while diminishing follow-up rounds (`FRP-v4` `PARTIAL`, `FRP-v5-H` `KILL_H`, `FRP-v5-S2` `N_max = 5,551 > 600`) froze the `vector61` line (`2026-10-08`).

## What was tried
- Ran a `180`-episode H5000 (`5,000`-frame) death census of `champion_a5 + v8` (`43d4e2c5`, `lambda=8.0`) and profiled `GameState` (`perf_sim_20261007`, `2.10×` parity-verified evaluation speedup, `with_slots.py`).
- Trained `FRP-v2` (`5` seeds continuing `champion_a5` with in-loop `v8`, `0.25×` Q/reward scaling, mass-priced death arm `M` vs `C`) and `FRP-v3` (`5` fresh seeds `10–14`, arm `M3`, `60,000` updates adding `greedy_food` and `random_safe` anchors).
- Evaluated `frp3-s12 + v8` (`eec144bf`) on `LH-1`, the 4-look Tier-2 strict gate (`N_max = 275`/mix, survival band v2), and Mac web serving qualification (`125,000` Watch decisions + `25` Play episodes).
- Tested `FRP-v4` (`5` seeds vs `frp3-s12 + v8`), `FRP-v5-S` / `FRP-v5-S2` (`67`-D `vector67` sight inputs), and `FRP-v5-H` (`8` seeds at shorter discount `γ = 0.995` and doubled Q heads), then audited all `52` legacy `research/*` packages (`2026-10-20`).

## What happened
- **`FRP-v2` (`GO_L`) & `FRP-v3` (`GO_R`):** `FRP-v2` gained **`+41.3` H5000 mass** (`LB90 +26.3`, `5/5` seeds `+`, scripted survival `-0.057`), and `FRP-v3` gained **`+65.1` Phase R mass** (`HK LB90 +43.0`, `LH-1 CLEAR`, `-77%` boost burn).
- **`frp3-s12 + v8` release (`STRICT_PASS`, `SERVING_PASS`):** Stopped `STRICT_PASS` at look `2` (`n = 207`/mix: **`+91.0` frozen, `+59.8` mixed, `+61.7` scripted**; pooled survival `+0.016`, LB `-0.011 >= -0.05`, scripted LB `-0.060 >= -0.075`) and shipped (`0.61 ms` wrapper/decision, `18` deaths in `125,000` Watch frames).
- **Follow-up saturation & `vector61` freeze (`FROZEN`):** `FRP-v4` stalled (`+6.8`, `LB90 -15.6`, `PARTIAL`); `FRP-v5-S` missed fidelity (`0.1533 > 0.15`); `γ = 0.995` in `FRP-v5-H` failed (`KILL_H`, **`-41.5` mass** vs control, `UB90 -24.8`, `1/8` seeds `+`); `67`-D `FRP-v5-S2` (`GO_S_RECIPE`) gained **`+21.5` mass** (`+8.3` vs control) at **`N_max = 5,551 > 600`**, freezing `vector61`.
- **Legacy package audit:** Classified `7` packages `REUSABLE-LIVE`, `6` `REUSABLE-INFRA`, and `39` `DEAD` (kept for SHA-256/test pins).

## Why
Training with `v8` inside the loop, mass-priced deaths, and scripted anchors aligned the MLP with its runtime shield (`-77%` boost burn), whereas `92%` of remaining `frp3-s12` deaths were big-body self-enclosures (`235–881` length in `v8` census) invisible to `61`/`67`-D rays and worsened by `γ = 0.995`.

## What to reuse / what not to repeat
- Keep `frp3-s12 + v8` (`SNAKE_SERVE_CHECKPOINT=frp3-s12`) as rollback and mixed-Watch opponent, plus `perf_sim_20261007/with_slots.py` (`2.10×` live-sim speedup) and the `7` `REUSABLE-LIVE` / `6` `REUSABLE-INFRA` packages.
- Do not shorten discount to `γ = 0.995` (`-41.5` mass) or reopen `vector61` (`vector67` is default only if ever reopened).

## Full record
Original: `git show 90b3db94:docs/research/history/frp.md` | Primary sources: [models-and-training-history.md](../../notes/models-and-training-history.md), [frp3_strict_result_2026-10-06.md](../frp3_strict_result_2026-10-06.md), [frp3_serving_result_and_release_2026-10-06.md](../frp3_serving_result_and_release_2026-10-06.md)
