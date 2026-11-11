# Teaching Search

<a id="solo-native-imitation"></a>
## Native Solo Behavioral Cloning, Spacious Teacher, and Learner-State Aggregation (`solo-native-imitation`, Series `X`–`AM`)

**Dates:** 2026-09-13 to 2026-09-14
**Verdict:** Food-geometry behavioral cloning (`AA`) confirmed `64`- and `128`-frame native solo gameplay (`90`–`96` of `96` alive) despite missing the `>= 95%` held-out fit gate (`88.623%` peak), whereas `256`-frame survival failed (`55`–`81` of `96` vs `>= 87/96` bar) across spacious-teacher BC (`AG`), two learner-state aggregation rounds (`AI`, `AJ`), and food-view auxiliaries (`AL`).

### What was tried
- Trained `500`-update `RasterDuelingNetwork` (`raster31v3`) behavioral cloning (`BC`) on `3,072` `64`-frame native `145 x 83` `GreedyFood` rows across `4` worlds (`X`), `16` worlds (`Y`), and food-geometry-only inputs (`AA`), plus an observation-only decoder probe on `12,288` (`X`/`Y`/`W`) and `10,775` (`AK`) saved rows.
- Evaluated `AA`'s `3` fixed policies zero-shot on fresh `64`-frame, `128`-frame, and `256`-frame (`AD`) `8`-world (`96`-lane) banks and replayed all `100` `256`-frame deaths.
- Calibrated a flood-fill `SpaceTeacher` (`96/96` `256`-frame alive, `46.375` food vs `GreedyFood` `67/96`, `41.729`) and ran `256`-frame BC (`AG`), teacher-veto miss replay (`AH`), two DAgger-style `1,536 + 1,536` learner-state aggregation rounds (`AI`, `AJ`), and food-view auxiliary/inference checks (`AL`, `AM`).

### What happened
- **`64`/`128`-frame success vs `256`-frame traps (`AA` -> `AD`):** `AA` missed `>= 95%` fit (`87.093%`–`88.623%`, decoder **`12,288 / 12,288`** and **`10,775 / 10,775`**), yet passed **`3/3` gates at `64` (`94–96 / 96` alive) and `128` frames (`90–91 / 96`)** before falling to **`55`, `79`, `66` of `96` at `256` frames** (`AD`, `100/100` fatal moves had `0` safe exits).
- **Spacious-teacher BC & aggregation (`AG`–`AM`, `FAIL`):** `AG` survived `71/96, 81/96, 56/96`, missing **`73.5%`–`86.5%`** of vetoes (`AH`: `97.3%`–`100%` veto error before death). Aggregation lifted survival (`AI`: `55→70, 66→73, 58→71`, `+12.15 pp`; `AJ`: `61→71, 65→72, 63→65`, `+6.60 pp`), but disagreement fit stayed `20.0%`–`51.1%` (`< 87/96` gate), and `AL` lost `-9.72 pp`.

### Why
All `100` `256`-frame deaths were sealed before the final frame (`0` advisory-safe fatal moves), and whole-raster BC overfit rare spacious-teacher veto states (`100%` train fit vs `20.0%`–`58.3%` held disagreement fit across `AG`, `AI`, `AJ`; the reports note this association without claiming counterfactual rescue proof).

### What to reuse / what not to repeat
- Separate closed-loop gameplay gates from exact teacher-action agreement (`AA` passed `64`/`128`-frame gameplay at `87.093%`–`88.623%` held fit).
- Do not rely on whole-raster `3,072`-row BC or `1,536`-row DAgger refreshes to generalize rare (`~4%`) free-space veto overrides (`20%–51%` held veto fit).

### Full record
Original: `git show 90b3db94:docs/research/history/solo-native-imitation.md` | Primary sources: [SOLO_FOOD_SERIES.md](../../experiments/SOLO_FOOD_SERIES.md), [README.md](../../experiments/README.md), [redesign-and-b8.md](../../notes/redesign-and-b8.md)

<a id="solo-food-safety-residual"></a>
## Solo-Food Safety Residual, Gate, and Free-Space Wrapper Series (Studies AT–BU)

**Dates:** 2026-09-19 to 2026-09-20
**Verdict:** Across `28` studies (`AT`–`BU`), a `2D` CNN safety head overfit (`21–32%` held fit), whereas a shared `3`-input per-action safety head (`1,500` updates, ranking hinge) generalized at `97.22%–98.57%` held fit and reached `96/96` (`100%`) solo survival across `3/3` seeds through `512`-step `H512` (`BO`), while food-parent fine-tunes (`BR`–`BU`) failed (`0/3`).

### What was tried
- Augmented frozen `FoodGeometryNetwork` parents (`AN`, `raster31v3`) with `2D` CNN body/wall/reachability safety residuals (`AT`–`BF r2`), testing KL preservation (`AW`), `16`-world coverage (`AYr1`), full-raster context (`AZ`), `3` flood-fill scalars (`BA r1`, `BB`), and disagreement decomposition (`BC`).
- Built a **parent-consistent safety oracle** `O` overriding the parent only when its normal choice was unroomy (`BD r1`, `BE`), then audited exact-view feasibility (`AX`, `BG`, `BI`).
- Replaced the CNN with a shared per-action `Linear(3,16)`–ReLU–`Linear(16,1)` head on `[deficit[i], threshold/cap, boost]`, replacing infeasible zero-KL/square penalties (`BH`, `BI`) with a capped-margin ranking hinge over `500` -> `1,500` updates (`BJ`–`BM`) and testing zero-shot `H256` (`BN`) and `H512` (`BO`).
- Audited remaining `>=128`-frame stalls (`BP`, `BQ`, `BS`) and tested `500`-update parent food fine-tunes (`1,536` own-policy rows `BR`, `16`-world breadth `BT`, CoordConv planes `BU`).

### What happened
- **CNN overfitting (`AT`–`BF r2`):** `2D` CNN safety residuals hit `100%` train fit but collapsed to **`21–32%` (`21.43%–31.82%`) held fit** (`BF r2`; `0, 0, 1` exact byte overlaps in `AX`), while `>50%` of raw `SpaceTeacher` diffs in `BC` were non-safety ties removed by oracle `O`.
- **Shared `3→16→1` head (`96/96` `H512` in `3/3` seeds):** On `138–146` distinct triples across `3,072` rows (`1,500` hinge updates), held fit hit **`97.22%, 97.78%, 98.57%`** (`99.93%–100%` ordinary keep in `BL`/`BM`) with **`96/96` (`100%`) survival in `3/3` seeds at `H128`, `H256`, and `H512`** (`BO`: `0` deaths vs `66, 75, 75`; food `77.01, 77.73, 77.93` vs `55.08, 52.75, 54.57`).
- **Food-stall fine-tunes (`BR`–`BU`, `FAIL 0/3`):** All `8` zero-food slice-lanes were inherited roomy parent trajectories (`0` safety overrides in `BP`/`BQ`), and `BR`, `BT`, and `BU` stalled at `86.65%–89.13% < 95%` held accuracy (`0/3` seeds).

### Why
Spatial CNN residuals memorized singleton body grids (`0, 0, 1` overlaps), and zero-KL/zero-square preservation formed negative Bellman–Ford cycles (`X` in `BI`) when identical `[deficit, threshold, boost]` triples appeared in both ordinary and veto states; the `3`-scalar shared head + ranking hinge removed both failure modes.

### What to reuse / what not to repeat
- Route invariant per-action safety quantities (`[deficit[i], threshold/cap, boost]`) through compact shared heads trained with action-ranking hinges rather than `2D` CNN residuals or zero-KL penalties.
- Strip non-safety teacher tie-break disagreements (`>50%` of `SpaceTeacher` diffs in `BC`) before supervising a safety override head.

### Full record
Original: `git show 90b3db94:docs/research/history/solo-food-safety-residual.md` | Primary sources: [experiment_portfolio_review_2026-09-26.md](../experiment_portfolio_review_2026-09-26.md)

<a id="controlled-encounter"></a>
## Controlled Encounter (2026-09-21 — 2026-09-22)

**Dates:** 2026-09-21 to 2026-09-22
**Verdict:** Closed: `256`-world empirical-teacher behavioral cloning reached `89.6–93.8%` held survival and `220–224` held food across `3/3` seeds (`3/3 RELATIVE PASS, 0/3 ABSOLUTE PACKAGE`), whereas native PQN $Q(\lambda)$ continuation failed its competence/retention gates (`LINE CLOSED (0/3 PASS)`).

### What was tried
- Built a `4`-frame (`H4`) `2`-snake tactical encounter benchmark (`576` cases across `6` families × `4` headings: `384` train on `16` worlds, `192` held on `8` worlds; teacher `576/576` alive, `228` held food) on `RasterDuelingNetwork` (`raster31v2`).
- Replaced off-crop-aliased clearance tie-breaks (`2` alias groups) with food-only `H4` sets, aggregated `1,040` learner-visited prefix states (`DAgger`), and weighted step-`1..4` food by **`8/4/2/1`** to break procrastination ties.
- Expanded training breadth from `16` -> `64` (`7,184` rows) -> `256` worlds (`25,616` rows reconciled under a finite-bank empirical teacher, `expand256` vs `repeat64` from mark `512 -> 1536`), auditing hidden food-respawn RNG (`180` cells) and L2/mark-`0` controls.
- Warm-started native PQN $Q(\lambda)$ from `expand256` (`lr=1e-4`, collector `cap=256 -> 16` in `SHORT64` -> `D256`), ran `1,152` `1`-action rescue rollouts, and tested crossing NLL rehearsal, direction-margin hinges, a `1,024`-update cycle, and `0.1` ambient bonuses.

### What happened
- **Procrastination tie-break (`8/4/2/1`):** Weighting earlier food (`8/4/2/1`) eliminated **`87/87` (`100%`)** procrastination loops and lifted teacher food from `383 -> 480` (train) and `221 -> 256` (held).
- **`64 -> 256`-world BC (`3/3 RELATIVE PASS, 0/3 ABSOLUTE PACKAGE`):** Expanding `64` (`repeat64`) to `256` worlds (`expand256`) lifted held food from **`152, 144, 160` to `220, 224, 220`** (`+0.3542, +0.4167, +0.3125`/case), fit to `93.23%, 95.83%, 93.75%`, and survival to **`89.6–93.8%` (`180/192, 176/192, 172/192`)** across `3/3` seeds (`0%` train coverage on all `48` fatal held games).
- **Collector horizon (`SHORT64`):** Cutting collector cap `256 -> 16` raised visited worlds from `51–63` to `206–207` (`1,069–1,106` `H16` food).
- **PQN continuation (`LINE CLOSED 0/3 PASS`):** Continuing to `D256` traded crossing for frontal survival (`28/28` crossing losses diverged at even steps `2`/`4`, `16/16` frontal rescues at odd steps `1`/`3`, `100%` reversed by `1`-step parent restore); rehearsal, hinges, `1,024`-cycle (`889–1,034` food), and ambient bonus all failed (`0/3`).

### Why
Unweighted `H4` ties allowed procrastination (`87/87` fixed by `8/4/2/1`), and `16`–`64` worlds left unseen states (`0%` train coverage on all `48` `expand256` deaths); for PQN's trade-off (`934` lost vs `40` gained rows, `3.00–3.63` terminal vs `0.25–0.27` interior TD residuals), `1`-action rescue proved single-action sufficiency in `44` selected cases, not an optimizer mechanism.

### What to reuse / what not to repeat
- Weight earlier rewards (`8/4/2/1`), reconcile hidden off-crop/RNG label aliases empirically, and scale unique training worlds (`256` vs repeating `64`) plus short collector caps (`cap=16` vs `256`).
- Do not expect unanchored or NLL/hinge-rehearsed PQN $Q(\lambda)$ continuation (`0/3` PASS, negative gradient cosines `-0.1746, -0.1073`) to preserve tactical crossing margins.

### Full record
Original: `git show 90b3db94:docs/research/history/controlled-encounter.md` | Primary sources: [README.md](../../experiments/controlled_encounter_common_path_2026-09-21/README.md), [README.md](../../experiments/controlled_encounter_hidden_rng_2026-09-21/README.md), [README.md](../../experiments/controlled_encounter_residual_policy_diagnostic_2026-09-21/README.md)

<a id="t34-claim-a"></a>
## T34 Claim A Bare-Student Distillation and Evaluation Pipeline

**Dates:** 2026-10-19 to 2026-10-21
**Verdict:** Closed `KILL` at the registered L4 `D0` diagnostic (`d2454013`) — shared-weight distillation flipped only `0.4%` of rescues, while split-stream objectives flipped `100%` of free Qs (`225/225`) yet broke held-out `b8-s3` anchor guards within `< 512` neural updates, leaving `R1` (`t34_claim_a_r1`) `UNUSED / CLOSED_WITHOUT_CONSUMER`.

### What was tried
- Built `t34_exit_b8_fit` (`S1b` bare `Ego2sNet` and `S1a` adding a zero-init 4-plane `152x88` full-arena conv tower with `max|dQ| == 0`), a once-per-round lossless `uint8` mmap cache (`12,777` B/row `S1b`, `66,281` B/row `S1a`), and parallel `32 x H5000` `DevProbe` (`4.0 ms`/decision/worker).
- Tested the v1 shared-weight objective (`KL` at `tau = 0.05` + `margin = 0.1` hinge + `0.01*adv_MSE`, `1:1` searched:unsearched quota) on `39` excluded A5 v2 worlds (`190,779` rows: `190,483` `shadow_confirm`, `225` decisive `horizon_rescue` in `21/39` worlds, `71` `all_die_delay`).
- Ratified and ran the `3`-fold × `9`-cell `D0` diagnostic (`t34_claim_a_loss`) testing split-stream losses (`L_A = KL + 0.01*adv_MSE` on `896` anchors; `L_D = tau*CE + hinge` with no self-KL on `128` decisive rows).
- Swept `trainable` B8 and frozen-B8 `residual` heads over `U* in {32, 128, 512, 1024, 2048, 4096}` updates against `>= 70%` decisive flips and anchor guards (`<= 5%` length-decile disagreement, `<= 1.5x` boost ratio, `<= 90%` single-action share).

### What happened
- **v1 shared-weight algebraic lock:** Optimizing unconstrained free per-row Q-values for `5,000` Adam steps (`lr = 1e-2`) flipped only **`0.4%`** of decisive rows to `a*` (**`97.8%`** stayed bare `b8-s3`), and the `1:1` sampler ran only `~1` update/epoch on `39` worlds.
- **Split-stream `D0` rerun (`KILL`):** Removing self-KL from decisive rows achieved **`100%` (`225/225`) free-Q flips** (`1.0` vs `0.0044` frozen, passing `>= 99%` eligibility), but **every neural cell (`trainable` and `residual`) failed the held-out anchor guards at `32..4096` updates** (none held `>= 512` updates), leaving `800+200` `R1` unused.

### Why
Applying `KL(softmax(Q_prev/0.05) || softmax(Q/0.05))` (`1/tau = 20`) on decisive rows cancelled the `0.1`-margin hinge; once split streams fixed free-Q flips, rare `horizon_rescue` states (`225 / 190,779`) overlapped ordinary states in single-frame space so gradient updates broke normal `b8-s3` anchor behavior before `512` steps.

### What to reuse / what not to repeat
- Never apply a parent-policy self-KL term on decisive teacher-override rows; separate anchor (`shadow_confirm`) and decisive (`horizon_rescue`) loss streams and drop zero-weight `all_die_delay` rows.
- Reuse once-per-round `uint8` mmap observation caches (`12,777` B/row, `1.8 s` fit) instead of per-epoch replay regeneration, and test free-Q + anchor spillover (`D0`) before funding `800+200`-world label banks (`R1`).

### Full record
Original: `git show 90b3db94:docs/research/history/t34-claim-a.md` | Primary sources: [2026-10-21_t34_claim_a_amendment.md](../specs/2026-10-21_t34_claim_a_amendment.md), [t34_claim_a/README.md](../../../research/t34_claim_a/README.md), [t34_claim_a_r1/README.md](../../../research/t34_claim_a_r1/README.md)

<a id="bl-labels"></a>
## Big-Lever F0 Hindsight Label Factory, F1 Placement Ruler, and D256/D1024/D4096 Production

**Dates:** 2026-10-23 to 2026-10-25
**Verdict:** Closed with Mac tranches `D256` (`1,566` units across `864` parents) and `D1024` (`3,865` units across `2,160` rows) `COMPLETE`, while `D4096` was sealed `NOT_RUN_BY_RULE` after `F4` Look `R-A` `STOP_FUNDING`, and both the L4 CPU tail lane and GPU label premise (`Step -1`, `CLOSE_NO_BOUND`) closed.

### What was tried
- Built the `F0` hindsight label factory (`research/bl_labels`) for bare `b8-s3` across S-gate (`H=5,000`) and `Watch6` (`H=14,400`), branching counterfactual `H=300` actions from 20-step ring snapshots at up to `90` doom + `8` ordinary roots/life with `10%` bare-reroll and `10%` `H=1,440` continuation audits.
- Ran Colab CPU Canary A (`36` session parents), prepared the `F1` 64-root placement ruler (Colab CPU vs 12-vCPU L4), and calibrated Mac `W12` vs `W6` throughput (`>= 1.20` bar) plus rolling-median slowdown and `>= 30 s` foreign-process persistence guards.
- Produced nested Mac label tranches `D256` (`864` parents: `192` fit + `64` reserve lives), `D1024` (`2,160` fold `1–4` parents), and early-started sealed `D4096` (`7,020` fold `1–4` rows), while testing incremental ring trimming (`equiv-02`) and the `Step -1` GPU vs Mac `W12` speed ruler (`>= 4x` bar).

### What happened
- **Canary A (`INCOMPLETE_BUT_INFORMATIVE`) & `W12` calibration (`PASS`):** Canary A verified `44/44` receipt-listed payloads (`31/36` parents in `6,465.88 s`, `1,012` restores, `82` bare rerolls, `0` mismatches; `1.2%` restore overhead). Paired Mac `W12` vs `W6` achieved `1.36x` wall throughput (`>= 1.20` bar) with `0` thermal trips.
- **`D256` Mac production (`COMPLETE`):** Completed all `864` parents (`1,566` immutable units) across `8` attempts in `16,333.70 s` (`4.537 h`, `~85 parents/h` at `W12`, `0` thermal trips).
- **`D1024` Mac production (`COMPLETE`):** After `equiv-02` matched all life payloads byte-for-byte (`FAIL` only on session metadata `beq-gate-scripted-00`), `D1024` ran OLD `61900af2` at `W6` and completed all `2,160` rows (`3,865` immutable units, `0` pending) across `2` attempts (resuming from `362` units after a `7,584.94 s` preemption).
- **L4/GPU closures & `D4096` (`NOT_RUN_BY_RULE`):** L4 v2 failed `C6` (`12` sidecar-download timeouts -> `13` attempts vs limit `3`), and `Step -1` measured `3.251x` optimistic / `1.751x` pessimistic vs Mac `W12` (`W12_BELOW_4`, `CLOSE_NO_BOUND`). `D4096` (`60` parents completed in `5,400 s` of `468,000 s` cap) was sealed `NOT_RUN_BY_RULE` after `D1024` `R-A` returned `STOP_FUNDING`.

### Why
20-step ring snapshots plus Q-ordered `exact_stop` kept restore/replay overhead to `1.2%` (`47.2%` natural play, `33.5%` doom branches), letting Mac `W12` run fast enough that an A100 GPU missed `>= 4x` (`3.251x`); single-life `1.30x` slowdown checks had tripped falsely (`~10%`/life) on Apple Silicon E-core scheduling until replaced by `4 * width` rolling medians.

### What to reuse / what not to repeat
- Reuse atomic per-life/session `.sha256` + `.complete` exports with verified union ledgers so preempted multi-attempt runs (`8` in `D256`, `2` in `D1024`) skip completed lives with zero recomputation.
- Guard Apple Silicon `W12` worker pools by aggregate wall throughput (`1.36x` vs `W6`), `4 * width` rolling medians, and `>= 30 s` (`7`-tick) `(PID, birth_tick)` persistence rather than single-life per-worker latencies.

### Full record
Original: `git show 90b3db94:docs/research/history/bl-labels.md` | Primary sources: [label-factory.md](../../notes/label-factory.md), [bl_labels/README.md](../../../research/bl_labels/README.md), [INTENT.md](../../../research/bl_labels/INTENT.md)

<a id="bl-round30"></a>
## Round-30 Post-F4 Optimizer Certificate Diagnosis (`R30-1`), Rescue-Mass Ranking (`R30-2`), and `G5` Calibration Hold

**Dates:** 2026-10-23 to 2026-10-25
**Verdict:** Companion `R32` passed (`RETAIN_FIXED_DESIGN`, synthetic only) and `R33` closed (`CANCEL_CPU_PILOT_NO_REMAINING_DEMAND`), while `R30-1` main and `R30-2` closed on `HOLD` at Gate `G5` (operationally unresolved, not a scientific `KILL`) after `Decimal50` gradient enclosure hit its `600 s` cap and needed a `40`-min probe exceeding the qualified `30`-min ceiling.

### What was tried
- Ordered `ADOPT_DIAGNOSIS_FIRST` (`bl_round30`) after F4's `1,542`-parameter residual failed zero-harm at R-A: `R30-0` corpus admission, `R30-1` optimizer certificate (`2,000` seed-`3403` Adam batches vs float64 L-BFGS `<= 500` closures + `Decimal50` enclosure capped at `600 s`), and conditional `R30-2` mass ranking (`>= 10%` regret cut, `>= 0.70` retention, `H = 0`).
- Admitted the post-F4 closed training corpus (`960` `D1024` fold-1–4 lives: `730` train lives / `291` parents / `65,825` rows and `230` validation lives / `81` parents / `20,698` rows) and ran the exact-input native L4 canary (`G4`) plus three L4 timing calibration probes (`probe-01` to `probe-03`) and `probe-04` dry-run.
- Ran companion `R32` (`research/r32_rehearsal`, `28`-unit synthetic sequential-consumer rehearsal on CPU) and `R33` (`research/r33_ops`, deterministic task-journal and 3-shard Mac preflight closeout).

### What happened
- **`R32` & `R33`:** `R33` passed `64` tests (`1.06 s`), `15` Mac preflight checks, and `15` CI nodes (`CANCEL_CPU_PILOT_NO_REMAINING_DEMAND`, `0 u`). `R32` (`r32-cpu-main-4f11670a-01`, `28/28` units, `0.0351 u`) passed (`RETAIN_FIXED_DESIGN`, synthetic only; consumer readiness `BLOCKED`) after fixing missing `df` in `standard_t` (`heavy_tail_df=5`).
- **`R30` `G4` & `G5` hold:** Native L4 canary `G4` passed (`1,100.253 s` wall, `715.99 s` worker), but `G5` held because unseparated screening surrogate `7,962 min` (`2 × 7,962 > 106` min main ceiling) lacked per-phase timings.
- **Timing probes (`probe-01`, `probe-02`):** `probe-01` refused pre-allocation (`UPLOAD_CAP` from a `512`-KiB export cap also bounding uploads); `probe-02` (`988.648 s`) failed `INCOMPLETE` with cause `UNKNOWN` under a blanket handler.
- **Timing probe `03` (`INCOMPLETE / DIAGNOSED`):** `probe-03` (`1,121.888 s`) logged cache `102.6 s`, Adam `~0.05 s / 10 updates` (`~10.5 s` for `2,000`), closures `~0.5 s` (`~257 s` for `500`), and `CERTIFICATE_DEADLINE` at `600.016 s`.
- **Route closure (`CLOSED ON HOLD` at `G5`):** `probe-04` (`1,200 s` certificate, `2,400 s` / `40`-min session) was refused by `canary.py:213`'s `30`-min probe ceiling, closing the calibration route while `F5` proceeded independently.

### Why
Optimization itself projected to a few minutes (`~10.5 s` Adam, `~257 s` L-BFGS closures), whereas computing the `Decimal50` directed gradient enclosure exceeded `600 s` (`600.016 s`), and doubling that sub-deadline required a `40`-minute probe session prohibited by the qualified supervisor's `30`-minute probe cap.

### What to reuse / what not to repeat
- Separate upload staging bounds (`64` MiB) from small timing-export byte caps (`524,288` bytes), and use public checkpoint URLs plus single-archive metadata bundling (`10,240` bytes) to avoid `UPLOAD_CAP`.
- Emit durable per-phase telemetry records rather than blanket exception handlers so probe timeouts pinpoint the exact phase (`CERTIFICATE_DEADLINE` vs `UNKNOWN`).
- Run synthetic consumer rehearsals (`R32`) before real-data looks to catch statistical code bugs (`standard_t` missing `heavy_tail_df=5`) at near-zero cost (`0.0351 u`).

### Full record
Original: `git show 90b3db94:docs/research/history/bl-round30.md` | Primary sources: [r30-r32-f5-successors.md](../../notes/r30-r32-f5-successors.md), [RULING.md](../round30_postf4_20261025/RULING.md), [INTENT.md](../round30_postf4_20261025/INTENT.md)

<a id="f4-f5"></a>
## Frozen-B8 Residual Dose Study & Harm-Weighted Successor (`F3` / `F4` / `F5`)

**Dates:** 2026-10-23 to 2026-10-26
**Verdict:** `F4` ended in terminal `STOP_FUNDING` at Look `R-A` (`D1024` lifted rescue retention `+6.62 pp` but failed zero-harm with `H = 19` confirmed lethal flips, sealing `D4096` `NOT_RUN_BY_RULE`), and exploratory harm-weighted successor `F5` closed before any fit (`F5_CLOSED_NO_FIT`) after `732` Sandbox B configs proved a `~4.5%` retention ceiling at `<= 1%` ordinary flips.

### What was tried
- Built `F3` and pre-registered `F4` (`research/bl_residual`): fitted a zero-initialized `Linear(256, 6)` residual on cached 256-D `b8-s3` trunk features (`2,000` Adam updates, batch `256` = `128` doom + `128` ordinary rows) across nested rescue-life doses `D256` (`192` fit lives), `D1024` (`960` fit lives), and `D4096` (`4,032` fit lives), plus a `64`-life fold-0 reserve.
- Ran `F2` (comparing `100%` vs `25%` `A2` `c0` frozen-linear `D300` models on `1,481` `T5` rows across `269` worlds) and evaluated `F4` `D256` (`fit-0000.pth`) and `D1024` (`fit-0001.pth`) at single Look `R-A` against retention (`>= +0.05` gain, `LB90 > 0`) and natural-archive zero-harm (`H = 0` on `560` natural rows / `29` parents) gates.
- Authorized a once-only `F4` harm post-mortem (`development-parents.json`) and launched exploratory `F5`, selecting arm `W` (`x4` ordinary survival-pair weight where `B8` survives and an alternative dies, vs control `C`) on the reconciled `960` `D1024` lives (`86,523` rows: `79,156` doom, `7,367` ordinary) with `13`-unit idempotent L4 timing recovery (`672–683` + `C/W` calibration).

### What happened
- **`F2` & `F4` Look `R-A` (`STOP_FUNDING`):** `F2` struck volume (`Δ = -0.029034`, `LB90 = -0.042134 <= 0`). At `R-A`, `D1024 - D256` life-macro rescue retention passed (`+0.06621223984499343` / `+6.62 pp >= +0.05`, `LB90 = +0.048454798628559156` / `+4.85 pp > 0`), but `harm_screen_d1024_pass = false` (`H >= 1` vs `H = 0`), halting `D4096` after `18,267.14 s` as `NOT_RUN_BY_RULE`.
- **`F4` post-mortem:** Across `560` natural rows (`29` parents), `D256` caused `H = 20` confirmed exact-death flips (`493` changed actions) and `D1024` caused `H = 19` (`541` changed actions, `U = 0`), triggering `F5` arm `W`.
- **`F5` timing hold:** All `13/13` L4 timing units finished (`4,090.25 s`), but Mac reduction returned `HOLD_REGISTERED_COVERAGE_FAILED` (`0/3` Watch fixtures `672–674` reached `L >= 700` or `5,000` decisions, and charging full `21.377 s` `rollout_all` bundles per scalar continuation inflated conservative stage-wall to `88.969 h` / `88.97 h` vs the `8 h` / `16 h` guard).
- **`F5` final closure (`F5_CLOSED_NO_FIT`):** After `732` Sandbox `B`/`B-2`/`B-3` configs (`0/432` in `B-3`, `k`-NN ceiling `~0.15–0.17`, `~4.5%` retention at `<= 1%` flips), the owner closed `F5` before any fit.

### Why
Scaling rescue dose (`D256 -> D1024`) improved rescue agreement (`+6.62 pp`) while still rewriting most ordinary decisions (`541/560` rows, `H = 19` deaths) because frozen 256-D `B8` features cannot separate rare self-traps from normal play; `F5` hit its `88.97 h` hold because the cost equation multiplied bundle time (`21.38 s`) across `88,320` scalar continuations.

### What to reuse / what not to repeat
- Bundle small staging files (`18 -> 5 -> 3` in `D256`) and publish per-unit immutable exports (`f5_timing_resume`) so `RPC_EXEC_POLL_TIMEOUT` interruptions never discard completed Colab work.
- Price cost equations per actual simulator bundle call rather than per scalar action row (`88.97 h` overcount), and do not fit post-hoc heads on frozen `phi_B8` expecting zero natural-archive harm.

### Full record
Original: `git show 90b3db94:docs/research/history/f4-f5.md` | Primary sources: [f4-study.md](../../notes/f4-study.md), [JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md](../../../research/bl_residual/ledger/F4_R_A_20261025/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md), [F5_CLOSED_NO_FIT.md](../../../research/bl_residual/ledger/F5_CLOSURE_20261026/F5_CLOSED_NO_FIT.md)

<a id="sandbox-b"></a>
## Standing A100 Development Sandbox (`B1`/`B2`/`B3` Probes) and Prospective Child Collector v2

**Dates:** 2026-10-26 to 2026-10-27
**Verdict:** No candidate passed across `60` B-1 configs, `240` B-2 config-threshold rows, and `432` B-3 points (`0/432` met `>= 0.70` rescue retention at `<= 0.01` ordinary flips and `<= 1` harm, peaking around `~4.5%` retention at `<= 1%` flips), prompting the owner to close `F5` with no fit while keeping `tools/sandbox/` active.

### What was tried
- Admitted the filtered `D1024` counterfactual rescue corpus (`730` lives, `291` parents, `65,825` rows split into `54,771` train / `11,054` validation rows with `6,201` decisive and `936` ordinary validation rows after excluding `248` F4/census/R30 parents) under `tools/sandbox/{admission,preflight}.py`.
- Ran `B-1` (`60` configs, `15,000` updates on A100): swept linear, length/margin-gated 2-layer MLP, and cosine-normalized residual heads on frozen 256-D B8 features (`phi`) across `5` ordinary KL weights `{1,2,4,8,16}` and `4` survival-pair weights `{1,2,4,8}`.
- Ran `B-2` (`48` configs × `5` switch margins = `240` rows, `48,000` updates): swept `normed_linear`, `cosine_fastgain`, and `supervised_gated_normed` heads across `kl_pairs` vs `hinge_top`, ordinary:survival ratios `{0.5,2,8,32}`, and row-uniform vs life-balanced sampling.
- Ran `B-3` (`432` points, `6,000` updates): tested cosine `k`-NN retrieval (`k in {5,15,50}`) plus `12` direct survival-veto configs adding `[phi, q_base]` context and `12`-D local crop geometry (`7` crop values + `5` v8 scalars, fixing neck-TTL blocking), alongside `tools/sandbox/supervisor_v2.py` (`wait4` child CPU collector).

### What happened
- **`B-1` (`No candidate`, `68.18 s`):** Rescue retention reached `0.552`, but every configuration caused `>= 7` ordinary-harm-proxy cases on validation.
- **`B-2` (`0/240` promising, `296.88 s`, `0.33 u`):** Best rescue retention `0.587` required a `0.925` ordinary flip rate; at ordinary flip rate `<= 0.01`, best retention was `0.038` of `6,201` decisive rows.
- **`B-3` (`0/432` promising, `83.07 s`, `0.29 u`):** Best retention `0.571` came at `0.997` ordinary flips; at `<= 0.01` (`<= 1%`) ordinary flips, best retention was `0.045` (`~4.5%`), missing `>= 0.70` (`>= 70%`) retention with `<= 1` harm and forming the basis to close `F5` with no fit.
- **Child Collector v2 (`ACCEPT` offline):** Passed `621` compatibility and `263` focused tests (`12,680,000 ns` CPU captured in local smoke), remaining `NOT_ADMISSIBLE` for live served panels pending host integration qualification.

### Why
Single-frame 256-D B8 features (`phi`), `[phi, q_base]`, and `31 × 31` local crop geometry overlap heavily between future self-traps and safe coiling states because full-arena multi-snake enclosures and off-crop tail release happen outside the `31 × 31` view, forcing every head onto one retention-vs-false-override curve.

### What to reuse / what not to repeat
- Reuse `tools/sandbox/{admission,preflight,supervisor_v2}.py` (`wait4` integer-nanosecond child CPU accounting without touching the qualified `research/colab_cli` tree).
- Do not repeat post-hoc linear, MLP, retrieval, or local-crop veto heads on static frozen `ego2s-bv8` features expecting selective rescue retention (`>= 70%` at `<= 1%` ordinary flips).

### Full record
Original: `git show 90b3db94:docs/research/history/sandbox-b.md` | Primary sources: [sandbox_b_heads/README.md](../../../research/sandbox_b_heads/README.md), [B2.md](../../../research/sandbox_b_heads/B2.md), [B3.md](../../../research/sandbox_b_heads/B3.md)

<a id="t34-ct-1028"></a>
## T34 Counterfactual Learnability + Dynamic-Topology Probe (`t34-ct-1028` / `t34-cf-1028`)

**Dates:** 2026-10-28 to 2026-10-29
**Verdict:** Prepared and canary-qualified on A100 with `allocations=0` on the main run — counterfactual learnability and dynamic-topology results were not yet collected, while compact `v2` snapshot compression (`28.3–35.5x`) and startup/watchdog liveness fixes passed live GPU verification.

### What was tried
- Built `CTDeviceSim` (GPU counterfactual simulator with `7` startup forced-state CPU/device parity checks), `LookbackRing`, and `H = 300` `4 + 4` Common-Random-Number (`CRN`) branching across `8` pre-death lookbacks `{15, 35, 65, 100, 140, 185, 235, 285}` on `1,024` worlds (`820` train / `204` held-out assessment, `2,623,488` disjoint 63-bit seeds).
- Designed a 3-arm distillation probe (`frozen_phi` frozen B8 trunk, `full_bv8` trainable B8, and `full_bv8_plus_topology` adding `(4, 83, 145) + 6` full-board dynamic-topology channels) against pre-set bars (`>= 25%` rescuable deaths, `>= 30%` `full_bv8` retention and `>= +15 pp` over `frozen_phi` at `<= 1.0%` anchor flips, plus `>= +15 pp` topology lift).
- Built lossless `v2` snapshot/topology compression (`ct_snapshots.py`, `ct_topology.py`) to fit `50,000` anchors + `2,560` roots inside stage export caps (`256 MiB` stage / `384 MiB` total).
- Ran live A100 CUDA canaries (`ct-cuda-smoke-20261028-08` and `-09`) and 50-iteration watchdog qualification tests to harden cold-startup heartbeats, `SIGUSR1` preemption saves, and native timeout exits for H100/A100 dispatch.

### What happened
- **Main probe (`Prepared`, `allocations=0`):** Dual-variant H100-first (`sm_90`) and A100-fallback (`sm_80`) packets (`sandbox-ct-cf-20261028-14-r4t1`, `720 min` base / `1,440 min` effective cap, `193.0 u` A100) passed `2x` `dispatch.sh --dry-run` under transport tree `3a7b2188`, with no scientific run executed in this record.
- **Lossless `v2` compression:** Deriving wall occupancy from `own_ttl`/`enemy_ttl`, delta encoding, byte-shuffling, and chunked `zlib(level=6)` packing (`~45` tensors instead of `84,480`) shrank projected `stage-0001.pt` by `28.3–35.5x` (`1,449.4 MiB -> 51.27 MiB` at `L=700`, `3,110.3 MiB -> 87.72 MiB` at `L=2,400`).
- **A100 startup & `SIGUSR1` (`PASS`):** Canary `08` exposed a `355.51 s` silent gap during cold Numba/CUDA startup; starting `StatusHeartbeat` (`15 s` timer before heavy imports) in canary `09` kept max poll gap to `17.3 s <= 30 s` and saved a `267.4 MB` `SIGUSR1` snapshot in `2.190–2.195 s` (`3.097–3.169 s` total exit `< 15 s` SLA).
- **Watchdog fix:** Offline qualification caught a `1/10` (`1` in `10`) CPython `3.12` (`3.12.3`) deadlock in `faulthandler.dump_traceback_later(exit=True)`, fixed by external helper `ct_watchdog_guard.py` (`0.25 s` post-dump `SIGKILL` grace).

### Why
Cold CUDA/Numba compilation and startup parity checks ran for `355.51 s` before the main script emitted any stage status, and CPython `3.12` `faulthandler` deadlocked in `_Py_DumpTracebackThreads -> dump_frame` when walking active heartbeat threads during a timeout dump.

### What to reuse / what not to repeat
- Start a lightweight stdlib `StatusHeartbeat` before importing PyTorch/Numba/CUDA so `355.51 s` cold-compile gaps never look like hung workers to the Colab supervisor.
- Pair `faulthandler` timeouts on Python `3.12` with an independent process (`ct_watchdog_guard.py`, `SIG_IGN` on `SIGUSR1`/`SIGINT`) that sends `SIGKILL` if traceback dumping stalls.
- Reuse domain-specific lossless packing (deriving redundant occupancy planes + byte-shuffle + `zlib`) instead of saving tens of thousands of raw PyTorch leaf tensors (`28.3–35.5x` smaller).

### Full record
Original: `git show 90b3db94:docs/research/history/t34-ct-1028.md` | Primary sources: [CT_1028.md](../../../research/sandbox_a_resident/CT_1028.md), [CT_1028_14.md](../../../research/sandbox_a_resident/CT_1028_14.md), [CT_1028_14_RESEAL_04_TRANSPORT01.md](../../../research/sandbox_a_resident/CT_1028_14_RESEAL_04_TRANSPORT01.md)
