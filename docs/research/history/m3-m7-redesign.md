# October Spatial Redesign (GridBatchSim + ego2s-bv8 M1–M7) and T1–T11 / Track-4 Rollout & Return-Bank Series

**Dates:** 2026-10-07 to 2026-10-31
**Verdict:** Bare `B8` (`ego2s-bv8`, `~1.64M` params, no runtime veto) cleared T7 Phase R `STRICT_PASS` (`+138.67` H5000 mass over `frp3-s12 + v8`, `5/5` seeds `+`, Tier 2 `-0.016 >= -0.05`) and shipped `b8-s3`, while every post-B8 late-death fine-tune, feature probe, and privileged critic (`M6`, `T2–T6`, `T8–T11`, `t4_gate1`) closed `NULL`, `KILL`, or diagnostic-only.

## What was tried
- Built cell-exact `GridBatchSim` (`72/72` identical H5000 episodes, `27,923` scripted hero steps/s) and distilled `2.08M` `frp3-s12 + v8` decisions into `Ego2sNet`: `M2` (`12` scalars) vs `M2b` (`ego2s-b`, adding `enemy_next` + `18` scalars = `30` scalars).
- Trained multi-actor Double DQN in `M3-B` / `M3-C` (`gamma 0.99 -> 0.995`, `v2` kill bonus) and repaired the recipe in `M4-A` / `M4-B` (`ego2s-bv8`: `7×31×31` + `4×37×37` ego-rasters + `60` scalars incl. `30` `v8` safety scalars, `mass` reward `kappa=1.0`, `kill_scale=0`, `gamma -> 0.998`, `50/25/25` scripted/frozen/mixed, `v8margin` + on-policy `v8` labels, `5` seeds × `20M`).
- Evaluated `5`-seed `B8` on T7 fresh Phase R (`1,920` fresh worlds = `5,760` H10000 episodes) against `frp3-s12 + v8` and Tier 2 `champion + v8`.
- Tested post-B8 late-death (`L >= 800`) fixes: `M6` shaping (`TIGHT`, `DEATHW`), `T2/T2b` Huber deltas (`{10, 30, 100}`), `T3`/`T4a`/`T4c` `D=300` rollout-margin banks (`16,327` rows), `T4d–T4f` `D=300` return regression (`19,976` states), `T6` `RESTART` snapshots (`10,885` states), `T5/T8/T9/T10` input/horizon probes, `T11` `LOCAL30`, and `t4_gate1` privileged full-arena critic (`384` worlds).

## What happened
- **Distillation & `M4-B` (`B8`) -> T7 `STRICT_PASS`:** `M2` failed (`-29.5` mass), whereas `M2b` passed (`+52.4` vs bare `frp3-s12`, `LB90 +34.8`). `M3-C` collapsed to `1–11%` survival (`177.4`/`172.9`/`184.6` vs `434.9`), while `M4-B` (`B8`) passed T7 `STRICT_PASS` at **`+138.67` (`+138.666`) H5000 mass** (`LB90 +129.48`, `+93.75` H10000, Tier 2 `-0.0156`), shipping `b8-s3` (`+149.65`).
- **Margin & shaping fine-tunes (`NULL`/`KILL`):** `M6` was `NULL` (`-24` / `+10` mass); `T2/T2b` Huber sweeps gained `+0.072`..`+0.125` scripted survival but lost `14–202` mass; `T3` rollout margins (`55.86%` boost labels) lost `81–88` (`-88` / `-88` / `-81`) scripted mass; `T4a`/`T4c` (`8.58%` nonboost bank + `10,885` outside counters, `3.81%` drift) ended `KILL` (`-0.027` / `-0.014` survival).
- **Return & restart fine-tunes (`KILL`):** `T4e` `RETURN` burned **`34%` boost** (`+31.99 Q` boost target gap), `T4f`/`T11` tripped `2%`/`1%` boost guards, and `T6` `RESTART` gave `0.000` survival change.
- **Horizon probes & `t4_gate1` (`KILL_T4`):** `T1` found `46/49` (`93.9%`) late deaths escapable within `300` steps (`T3` dense `38/64 -> 54/64`, `608.1 -> 700.0` mass), yet `60`-scalar AUC fell `0.9121` (`H=30`) -> `0.7261` (`D=300`; `0.726–0.769` in `T5`/`T8`, `49–65`-step pocket lag in `T9`), and `t4_gate1` passed `0/8` bars (`+0.0266` EV, `0.5617` doom AUC).

## Why
Feeding `30` `v8` safety scalars + `enemy_next` into `Ego2sNet` with living-mass reward (`kappa=1.0`), zero kill bonus, `gamma=0.998`, and `50/25/25` mixes cured `M3-C`'s kill-lunge trap (`1–11%` survival), whereas `L >= 800` traps form over `50–100` multi-snake steps (`0.726–0.769` D300 AUC), making margin nudges spill caution and truncated `D=300` returns hide post-`300` boost burn (`+31.99 Q`).

## What to reuse / what not to repeat
- Reuse the `M4-B` `B8` training recipe: multi-scale ego-rasters + teacher safety cues (`ego2s-bv8`), pure living-mass reward (`kill_scale=0`), `gamma=0.998`, and `50/25/25` opponent mixes (`train_m3c.py`, `GridBatchSim`).
- Never combine explicit kill bonuses with short discount factors (`gamma=0.995`), and never regress truncated `D=300` rollout returns without pricing post-horizon segment loss (`34%` boost collapse).

## Full record
Original: `git show 90b3db94:docs/research/history/m3-m7-redesign.md` | Primary sources: [t7_phase_r_result_2026-10-16.md](../t7_phase_r_result_2026-10-16.md), [m4a_design_2026-10-12.md](../m4a_design_2026-10-12.md), [redesign_campaign_retrospective_2026-10-17.md](../redesign_campaign_retrospective_2026-10-17.md)
