# Served Agents

<a id="apex-veto"></a>
## Apex vector61 Serving-Time Free-Space Veto Progression (v1–v8) and Early October Diagnostics

**Dates:** 2026-09-26 to 2026-10-03
**Verdict:** Four sequential deterministic free-space wrappers (`v2 -> v5 -> v7 -> v8`) around frozen `champion_a5` each cleared `STRICT_PASS` and `SERVING_PASS`, lifting `5,000`-frame (`H5000`) mass `6–7×` (`~49–62` to `~337–369`) while overriding `0.20%–0.29%` of moves at `0.65 ms`/decision (`v3`, `v4`, `v6`, `v9` retired).

### What was tried
- Built speed-preserving one-step flood-fill veto `v2` (`need = min(length, 160)`) around `champion_a5_freespace_20260621.pth` (`43d4e2c5`, where `85` of `96` pilot deaths were self-collisions) and ran Tier-2 strict (`N = 235`/mix) and `50`-episode web serving gates.
- Tested tail-aware one-step `v3` and depth-`8` lookahead `v4`, then ran a depth-`40` (`50,000`-node) point-of-no-return (PNR) diagnostic (`trap_horizon_20261001`) that motivated two-step boost-landing veto `v5` (`N = 269`/mix strict gate).
- Audited `v5` deaths (`77.6%` self enclosures sealed `20–60` frames early, `7.8%` head-ons), screened opponent-head `v6`, and tested tail-aware space-preference re-ranking `v7` (`lambda = 4`, `N = 243`/mix) and space + opponent-head `v8` (`lambda = 8`, 4-look OBF gate, `N_max = 249`/mix).
- Ran a `180`-episode `v8` death census (`60` worlds/mix, `24/24` SIMD-live bit-exact) and benchmarked `40`-step rollout prototype `v9`.

### What happened
- **First two promoted upgrades (`v2`, `v5`):** `v2` gained `+76.2` / `+73.3` / `+82.6` (`49.2->125.4`, `47.7->121.0`, `61.8->144.5`); `v5` gained `+71.6` / `+63.3` / `+62.9` while changing `0.03%` of decisions (`STRICT_PASS` & `SERVING_PASS`).
- **Space-preference upgrades (`v7`, `v8`):** `v7(4)` gained `+130.3` / `+119.1` / `+123.0` (`0.58 ms`); `v8(8)` stopped `STRICT_PASS` at look `2` (`187/249` worlds/mix = `561` worlds) with `+30.1` / `+43.0` / `+36.5` (`337.2` frozen, `355.5` scripted, `368.6` mixed; survival `~0.83–0.92`, **`0.65 ms`/decision**).
- **Retired variants (`NOT_ADVANCED`):** `v3` changed `4/120` worlds; `v4` passed `0` mixes (`694` no-escape fallbacks); `v6` cut head-ons `10 -> 2` (`+30.0` scripted) but missed frozen/mixed (`+10.0`, `+13.4`); `v9` cost `17–37 ms` worst-case.
- **PNR & `v8` censuses:** Under `v2`, `74%` (`31/42`) of self deaths had PNR `<= 8` frames (`24/31` boost pockets, `0/10` after `v5`). Under `v8`, survival hit `61.7%` (`111/180`, `337.4` mass, `0.6%` head-on), with `75%` of remaining death headroom from multi-frame opponent encirclement (`+60` to `+91` mass ceiling).

### Why
`champion_a5` ignored its own `3` `160`-capped free-space inputs on rare trap entries (`0.20%–0.29%` of ticks) and two-cell boost landings; once `v8` fixed local traps and head-ons, static single-snake vetoes hit a ceiling against `20–60`-frame multi-snake encirclements.

### What to reuse / what not to repeat
- Keep `v8`, `v7`, `v5`, and `v2` as serving rollback wrappers (`SNAKE_SERVE_CHECKPOINT=champion`, `SNAKE_SERVE_VETO_VARIANT=v8|v7|v5|v2`), and keep the `30` `v8` safety scalars embedded in `ego2s-bv8` (`b8-s3`).
- Always rebind `AISnake._get_frame` closures when cloning `GameState` (`copy.deepcopy` bug caught in `v9`), and run Mac strict gates on AC with lid open (`run-v1`/`v2` battery sleep `INVALID_STOP`s).

### Full record
Original: `git show 90b3db94:docs/research/history/apex-veto.md` | Primary sources: [apex_veto_v8_2026-10-03/README.md](../../experiments/apex_veto_v8_2026-10-03/README.md), [trap_horizon_2026-10-01.md](../trap_horizon_2026-10-01.md), [serving_time_search_feasibility_2026-10-03.md](../serving_time_search_feasibility_2026-10-03.md)

<a id="frp"></a>
## FRP fine-tune family v2–v5 on vector61/vector67 (frp3-s12 + v8 Release)

**Dates:** 2026-10-03 to 2026-10-20
**Verdict:** Fine-tuning `vector61` Apex with in-loop `v8` shielding and scripted anchors produced `frp3-s12 + v8` (`STRICT_PASS`, `SERVING_PASS`, released `2026-10-06`), while diminishing follow-up rounds (`FRP-v4` `PARTIAL`, `FRP-v5-H` `KILL_H`, `FRP-v5-S2` `N_max = 5,551 > 600`) froze the `vector61` line (`2026-10-08`).

### What was tried
- Ran a `180`-episode H5000 (`5,000`-frame) death census of `champion_a5 + v8` (`43d4e2c5`, `lambda=8.0`) and profiled `GameState` (`perf_sim_20261007`, `2.10×` parity-verified evaluation speedup, `with_slots.py`).
- Trained `FRP-v2` (`5` seeds continuing `champion_a5` with in-loop `v8`, `0.25×` Q/reward scaling, mass-priced death arm `M` vs `C`) and `FRP-v3` (`5` fresh seeds `10–14`, arm `M3`, `60,000` updates adding `greedy_food` and `random_safe` anchors).
- Evaluated `frp3-s12 + v8` (`eec144bf`) on `LH-1`, the 4-look Tier-2 strict gate (`N_max = 275`/mix, survival band v2), and Mac web serving qualification (`125,000` Watch decisions + `25` Play episodes).
- Tested `FRP-v4` (`5` seeds vs `frp3-s12 + v8`), `FRP-v5-S` / `FRP-v5-S2` (`67`-D `vector67` sight inputs), and `FRP-v5-H` (`8` seeds at shorter discount `γ = 0.995` and doubled Q heads), then audited all `52` legacy `research/*` packages (`2026-10-20`).

### What happened
- **`FRP-v2` (`GO_L`) & `FRP-v3` (`GO_R`):** `FRP-v2` gained **`+41.3` H5000 mass** (`LB90 +26.3`, `5/5` seeds `+`, scripted survival `-0.057`), and `FRP-v3` gained **`+65.1` Phase R mass** (`HK LB90 +43.0`, `LH-1 CLEAR`, `-77%` boost burn).
- **`frp3-s12 + v8` release (`STRICT_PASS`, `SERVING_PASS`):** Stopped `STRICT_PASS` at look `2` (`n = 207`/mix: **`+91.0` frozen, `+59.8` mixed, `+61.7` scripted**; pooled survival `+0.016`, LB `-0.011 >= -0.05`, scripted LB `-0.060 >= -0.075`) and shipped (`0.61 ms` wrapper/decision, `18` deaths in `125,000` Watch frames).
- **Follow-up saturation & `vector61` freeze (`FROZEN`):** `FRP-v4` stalled (`+6.8`, `LB90 -15.6`, `PARTIAL`); `FRP-v5-S` missed fidelity (`0.1533 > 0.15`); `γ = 0.995` in `FRP-v5-H` failed (`KILL_H`, **`-41.5` mass** vs control, `UB90 -24.8`, `1/8` seeds `+`); `67`-D `FRP-v5-S2` (`GO_S_RECIPE`) gained **`+21.5` mass** (`+8.3` vs control) at **`N_max = 5,551 > 600`**, freezing `vector61`.
- **Legacy package audit:** Classified `7` packages `REUSABLE-LIVE`, `6` `REUSABLE-INFRA`, and `39` `DEAD` (kept for SHA-256/test pins).

### Why
Training with `v8` inside the loop, mass-priced deaths, and scripted anchors aligned the MLP with its runtime shield (`-77%` boost burn), whereas `92%` of remaining `frp3-s12` deaths were big-body self-enclosures (`235–881` length in `v8` census) invisible to `61`/`67`-D rays and worsened by `γ = 0.995`.

### What to reuse / what not to repeat
- Keep `frp3-s12 + v8` (`SNAKE_SERVE_CHECKPOINT=frp3-s12`) as rollback and mixed-Watch opponent, plus `perf_sim_20261007/with_slots.py` (`2.10×` live-sim speedup) and the `7` `REUSABLE-LIVE` / `6` `REUSABLE-INFRA` packages.
- Do not shorten discount to `γ = 0.995` (`-41.5` mass) or reopen `vector61` (`vector67` is default only if ever reopened).

### Full record
Original: `git show 90b3db94:docs/research/history/frp.md` | Primary sources: [models-and-training-history.md](../../notes/models-and-training-history.md), [frp3_strict_result_2026-10-06.md](../frp3_strict_result_2026-10-06.md), [frp3_serving_result_and_release_2026-10-06.md](../frp3_serving_result_and_release_2026-10-06.md)

<a id="b8-release-serving"></a>
## B8 Seed 3 Strict Gate (T7), Serving Qualification & Release (T12/T12b/T12c), Product Roster Audits (T13/T13b), and Mixed Watch Qualification & Default Switch (T15/T15b)

**Dates:** 2026-10-16 to 2026-10-17
**Verdict:** Bare `b8-s3` (`3e649783`, `veto_variant: none`) passed T7 `STRICT_PASS`, cleared Mac serving qualification (`T12b` `SERVING_PASS`) to ship as default live policy (`T12c`), and switched Watch's default roster to `mixed` (`b8-s3` hero vs `frp3-s12 + v8` opponents) after `T13`/`T13b` roster audits and `T15` `SERVING_PASS` (`T15b`).

### What was tried
- Ran one-look Mac CPU strict gate `T7` (`5,760` H10000 episodes + `45` H5000 prefix controls across `960` fresh worlds) comparing the `5`-seed bare `B8` family against `frp3-s12 + v8` (Tier 1) and `champion_a5 + v8` (Tier 2).
- Ran 8-check Mac serving qualification `T12` (`73` fresh seeds) and Check-4 GC/speed-ceiling repair `T12b` (`24 fps` for `<= 6` driven AI, `12 fps` for `7..12`) before flipping `CHECKPOINT_RELEASED_DEFAULT = "b8-s3"` (`T12c`).
- Audited continuous-respawn H5000 Watch rosters (`25` paired fresh worlds/cell) at `12` snakes (`T13`) and at the `6`-snake product default + `8`-snake `all_b8` (`T13b`) across `all_b8`, `all_frp3` (hero-only `+v8`), `all_champion` (hero-only `+v8`), and `mixed` (`b8-s3` hero slot `0` vs `5` or `11` `frp3-s12 + v8` opponents).
- Qualified `SNAKE_SERVE_WATCH_ROSTER=mixed` across 6 Mac serving checks (`T15`, `33` fresh seeds) and switched Watch's default roster to `mixed` (`T15b`).

### What happened
- **`T7` (`STRICT_PASS`) & `T12b`/`T12c` (`RELEASED`):** Pooled `B8` beat `frp3-s12 + v8` by `+138.666306` H5000 mass (`LB90 +129.477718`, `5/5` seeds `+`; Tier 2 survival `-0.015625 >= -0.05`), selecting `b8-s3` (`+149.650051`). Clamping `7..12`-AI speed to `12 fps` fixed `T12`'s `1/120` `Watch12` GC overrun (`0 / 120` overruns; `Watch6` `5.967 / 8.048` ms, `Play6` `4.589 / 5.790` ms).
- **Red-Queen arena effect (`T13` & `T13b`):** Legacy `all_frp3` wrapped only hero slot `0` in `v8`, so `5` or `11` unwrapped foes churned at `2.109` arena deaths/1k (`12` snakes; `1.159`/1k at `6` snakes; `300` hero kills in `T13`).
- **Homogeneous `all_b8` vs `all_frp3`:** `all_b8` cut arena deaths `65–66%` (`0.712` and `0.404`/1k) and raised arena mass `+53%` (`1646.70` vs `1072.01`) to `+61%` (`1324.33` vs `824.57`), while making hero scores harder (`6`-snake hero p95 length `596` vs `723`, `-127` outside `100` -> `PLAN_MIXED_PENDING_ROSTER_QUALIFICATION`).
- **Mixed Watch (`T13b` -> `T15` `SERVING_PASS` -> `T15b` `RELEASED`):** At `6` snakes (`T13b`), `mixed` cut bare `b8-s3` hero deaths/1k from **`0.392` to `0.296`** (`+0.096`/1k gap `<= 0.10`; **`0.272`/1k** in `T15`, p50/p95 length `248`/`841`) and lifted hero H5000 mass from **`236.91` to `316.76`** (`Watch6` update p50/p95 `6.027 / 7.792` ms < `11.688 ms` trigger, `0 / 120` overruns).

### Why
Legacy Watch inflated hero stats because `5` or `11` opponents ran unwrapped without `v8`, whereas `6` `b8-s3` clones created much larger, longer-lived opponents (Red-Queen effect); `mixed` restored balanced exhibition opponents (`frp3-s12 + v8`) while keeping hero slot `0` completely bare, and `Watch12` `24 fps` overruns were `27.8–42.9 ms` Python GC pauses rather than `5.14 ms` neural forwards.

### What to reuse / what not to repeat
- Never compare single-hero Watch scores across changing opponent populations (Red-Queen effect); evaluate against fixed opponent mixes (`T7` / `mixed` roster).
- Enforce server-side fps ceilings (`24 fps` for `<= 6` AI, `12 fps` for `7..12` AI) so `27–43 ms` Python GC tails during `12`-snake snapshot serialization never cause paced tick overruns.

### Full record
Original: `git show 90b3db94:docs/research/history/b8-release-serving.md` | Primary sources: [t12_b8_serving_qualification_2026-10-17.md](../t12_b8_serving_qualification_2026-10-17.md), [b8_s3_release_2026-10-17.md](../b8_s3_release_2026-10-17.md), [t15_mixed_roster_qualification_2026-10-17.md](../t15_mixed_roster_qualification_2026-10-17.md)

<a id="watch6-confirmation"></a>
## Watch6 Live `GameState` Search Confirmation (`v1–v3`), Replay Bank Release, and All-Length (`L >= 1`) Extension

**Dates:** 2026-10-30 to 2026-11-01
**Verdict:** Confirmed that 300-step exact-stop lookahead (`exact-stop-l2-prime`) cuts hero deaths and boosts mass in live Watch6 (`GO_EFFICACY_ONLY`), shipped the 132-session precomputed Watch replay bank, and launched confirmation for all-length search (`Watch6-L1-V1` `IN FLIGHT`).

### What was tried
- Tested `exact-stop-l2-prime` (depth `H = 300` exact survival search every 5th frame when body length `L >= 600`) against bare `b8-s3` (the served convolutional snake network with no wrapper) across `N = 96` fresh paired `H = 14,400`-frame (`10 min` at `24 fps`) Watch6 sessions (`1` hero vs `5 × frp3-s12 + v8(lambda = 8)` opponents), mirrored cell-for-cell against `GridBatchSim`.
- Built precomputed `.watch.gz` deterministic replay banks (`research/watch_replay/`) so browser Watch serves confirmed search-assisted play at `24 fps` with live bare `b8-s3` fallback during cold startup verification.
- Tested two shortcuts for remaining deaths: `R6` (a 64-world / 384-root census and classifier trigger for safe mass harvesting below `L < 600`) and `R7` (depth-2, `K = 15`-tick deterministic spatial macro-options on 1-step-all-die states).
- Tested `exact-stop-l1` (removing the `L >= 600` gate to run `H = 300` search at all lengths `L >= 1` every 5th frame) on `48` paired `H = 6,000` Watch6-v2 sessions, then launched the 3-arm `Watch6-L1-V1` confirmation (`N = 96`, `H = 14,400`).

### What happened
- **Watch6 `V2`:** Ended `INVALID_OR_INCOMPLETE` (`88/96` sessions exported after shard 5 lost 3 attempts to pre-start staging/RPC failures; all `88` pairs stayed sealed with `0` outcomes read).
- **Watch6 `V3` (`GO_EFFICACY_ONLY`, `ELIGIBLE_FOR_REPLAY_QUALIFICATION`):** Passed across `96/96` pairs (`2,764,800/2,764,800` mirror frames, `0` mismatches): `deaths / 1k frames` fell `0.3248 → 0.1997` (`449 → 276`, `-0.1251` / `-0.125`, `-38.5%`, `CI95 [-0.1535, -0.0968] < 0`), and pooled mass rose `399.92 → 654.57` (`+63.68%` / `+63.7%`, Fieller `CI95 [+49.79%, +78.50%] >= -5.0%`).
- **Replay serving (`SERVED IN WATCH`):** Qualified `132/132` `H = 6,000` recordings (`watch-replay-bank-live-132`) on Mac at `24 fps` with cold readiness `0.349–0.435 s` (`< 0.45 s`), live-fallback `update p95 = 8.78 ms <= 11.688 ms`, and `0/120` paced overruns (`step p99 = 11.34 ms <= 41.67 ms`).
- **Shortcuts (`R6`, `R7`) & `exact-stop-l1`:** `R6` (`KILL`: `7.69%` lethal flips, OOF ROC-AUC `0.474–0.521`) and `R7` (`KILL` on `3/5` bars: `+3.85 pp < +10.0 pp` lift, `2 > 0` ordinary lethal flips) failed.
- **All-length search (`ADOPT`):** `exact-stop-l1` cut `L < 600` deaths `34 → 1` (`deaths/1k` `0.1597 → 0.0521`, `-67.4%`; mass `461.16 → 559.43`, `+98.27` / `+21.31%`) at `2.821×` branches (`1,268.5` vs `449.6`), launching `Watch6-L1-V1` (`IN FLIGHT`).

### Why
Most deaths remaining under `exact-stop-l2-prime` (`34` of `46` in the 48-session screen) occurred before length `600`, where local scalar classifiers failed (`0.474–0.521` ROC-AUC), whereas exact `H = 300` rollouts at `L >= 1` caught early self-traps directly; worst-case live search batches (`4.018 s` on L4) required `.watch.gz` replays to keep `24 fps` smooth.

### What to reuse / what not to repeat
- Reuse `.watch.gz` verified replay banks (`research/watch_replay/`) with cooperative GIL-yielding verification (`<= 20%` duty cycle) to serve slow lookahead controllers at `24 fps` with automatic live `b8-s3` fallback.
- Size Colab shard wall caps from empirical p95 runtimes (`7,557.99 s/pair`) rather than theoretical physical-maximum branch counts, which forecast `49.8 h` against a `22 h` ceiling and blocked all `18` candidate layouts.
- Keep incomplete confirmation banks (`88/96` in `V2`) sealed and unread rather than salvaging partial shards; do not repeat static scalar triggers (`R6`) or short 2-step spatial macro-options (`R7`) for `L < 600` or all-die escapes.

### Full record
Original: `git show 90b3db94:docs/research/history/watch6-confirmation.md` | Primary sources: [REGISTRATION_V3.md](../../../research/watch6_search_efficacy_v3/REGISTRATION_V3.md), [watch_replay/README.md](../../../research/watch_replay/README.md), [REGISTRATION_L1_V1.md](../../../research/watch6_l1_confirm/REGISTRATION_L1_V1.md)
