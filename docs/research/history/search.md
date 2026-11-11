# Search

<a id="b8-late-death-t1-t11"></a>
## B8 Late-Game Death Census, Two-Expert Panel and T1–T11 Probe Series (2026-10-14 to 2026-10-17)

**Dates:** 2026-10-14 to 2026-10-17
**Verdict:** Exact search proved `46/49` (`93.9%`) of B8's late (`L >= 800`) self-enclosure deaths were escapable within `300` decisions (`T3` dense shadow `54/64` vs `38/64` survival), every bare-network fine-tune and static feature probe (`T2–T6`, `T8–T11`) failed (`0.73–0.77` D300 AUC), and unmodified `b8-s3` passed restated `T7` `STRICT_PASS` for release.

### What was tried
- Ran a scripted death census (`b8_death_census_20261014`, `5` seeds × `32` worlds) after M4 Phase R look 2 showed `B8` beating `frp3-s12 + v8` by `+128.7` mass (`LB +101.5`) while failing single-seed guard `c6` vs `champion_a5 + v8` (`-0.156` vs `>= -0.05`, single-seed `SE` `~0.075`).
- Replayed all `75` census deaths (`49` at `L >= 800`, `26` at `L < 800`) backward `300` steps across all legal actions (`T1a`), probed B8's `256`-D post-ReLU fuse vector (`T1c`, `T5b`), and ran the `T3` `D=300` dense rollout ceiling (`L >= 600`, stride `5`).
- Screened `5`-seed × `8M` fine-tunes from B8 across Huber/anchor regimes (`T2`), `T3` rollout labels, `T4a` two-sided counters, `T4e/T4f` `H300` return regression, `T6` snapshot restarts, and `T11` `LOCAL30`.
- Probed `24` per-action geometry features (`T5`), arrival-race + raw-raster D300 CNN (`T8`), `46`-pair post-LAD divergence (`T9`), and tail-certificate / horizon AUC decay (`T10`).

### What happened
- **Death census:** B8's death hazard per `1,000` decisions was `0.068` at `L < 800` vs `0.177` at `L >= 800` (`49/75` deaths, `85%` enclosures, median trap `31` steps; champion spent only `5.6%` of decisions at `L >= 800` vs B8's `42%` of `659,595` decisions).
- **Search headroom (`T1a`, `T3` dense `GO`):** `T1a` found `46/49` (`93.9%`) late deaths escapable within `300` steps (median LAD Q-gap `0.45`; `17/46` boost-only), and `T3` dense search lifted scripted survival `38/64` -> **`54/64`** and mass `608.1` -> **`700.0`** with just `75` changed actions in `301,589` decisions.
- **Static features & post-LAD lag (`T1c`, `T5`, `T8–T10`):** H100 fuse AUC was `0.837`, but D300 AUC stayed `0.73–0.77` across `60` scalars (`0.912 → 0.726` D30->D300), `24` geometry features (`0.748`), fuse GBMs (`0.7630–0.7693`), arrival-race (`0.749`), and raw-raster CNN (`0.742`); `75%` pocket separation appeared at offset **`49`** (`65–83` sustained, `T9`).
- **Fine-tunes (`KILL` / `No GO`) & `T7` (`STRICT_PASS`):** `T2` (`HUBER_ANCHOR0` `KILL`, `ANCHOR0` `+0.003`), `T3` labels (`-88.0 / -87.9 / -81.2` mass), `T4a` (`-0.027` survival, `-24.1` mass), `T4e` (`34%` boost at `0.8M`), `T6` (`0.0000` survival), and `T11` (`1.073% > 1%` boost at `300,160` steps) failed; unmodified `B8` passed `T7` (`+138.67`, Tier 2 `-0.0156`).

### Why
B8 dies more often at `L >= 800` primarily because it grows big enough to spend `42%` of its decisions there (vs `5.6%` for champion), where fatal enclosures look nearly identical to safe play at the LAD (median Q-gap `0.45`) and separate geometrically only `49–65` steps later, causing static single-frame student updates to over-perturb normal play or collapse into boost.

### What to reuse / what not to repeat
- Stratify survival hazard by body length (`L < 800` vs `L >= 800`) and avoid single-seed `32`-world survival gates (`SE ~0.075`, `MDE ~0.19`).
- Use online multi-step rollout search (`75` interventions in `301,589` decisions lifted survival `38/64 -> 54/64`) rather than static D300 feature engineering (`0.73–0.77` AUC) to rescue sealed-neck enclosures.

### Full record
Original: `git show 90b3db94:docs/research/history/b8-late-death-t1-t11.md` | Primary sources: [PANEL_DECISION_MEMO.md](../panel_2026-10-15/PANEL_DECISION_MEMO.md), [t1_commitment_study_2026-10-15.md](../t1_commitment_study_2026-10-15.md), [t3_ceiling_test_2026-10-15.md](../t3_ceiling_test_2026-10-15.md)

<a id="t34-search-teacher"></a>
## T34 ExIt-B8 Search Teacher, Shadow Design, and Claim A Distillation (`t34-search-teacher`)

**Dates:** 2026-10-18 to 2026-10-21
**Verdict:** Claim A (bare student distillation) closed `KILL` at `D0`, Round 0 closed `KILL / INTEGRITY_FAILURE`, and `O2` closed `DEFERRED & CLOSED`, while deterministic rolling-shadow (`D=300`/`500`) bisection search, `K=12` L4 CPU fan-out, and `S2` compiled `v8` (`1.90×` speedup) superseded stochastic `N8/H128` search and remained in Claim B.

### What was tried
- Tested stochastic `SearchTeacher` (`N=8` rollouts × `6` actions, `H=128`, `eps=0.02`, `L >= 600 OR margin < 0`) in Phase 0 (`160` paired H5000 worlds) and lockstep `T33` (`W×6×N` batched continuation).
- Measured `M1–M4` (`research/t34_shadow`): `15,000`-frame live/`GridBatchSim`/`D=300` shadow parity (`M1`), binary bisection + radius-`3` search on `64` T3 worlds (`75` historical changes, `M2`), serial Watch Last-Avoidable-Decision (`LAD`) latency (`M3`), and CPU batch-1 ONNX vs PyTorch runtime (`M4`).
- Scaled the `teacher_v2` engine on Colab L4 (`A5 v2`, `>= 50 worlds/h` bar) and ran `Round 0` (`240` paired H5000 worlds) with `Round 0-D` (`<= 120 u`) prepared as a fallback.
- Tested **Claim A** (`D0`): distilling `horizon_rescue` labels from `39` A5 v2 worlds (`190,779` rows, `225` decisive) into a bare `Ego2sNet` (`>= 70%` held-out decisive flip + `>= 98.5%` unsearched / `>= 99.5%` high-margin anchor guards).
- Ran `S0–S2` CPU speedups (`native_v8`) and `O1/O2` corpse-harvest checks (`48`-world `O1` H500 census and `O2` closed-loop sizing).

### What happened
- **Stochastic `T33` vs Shadow `M1–M4`:** Two `N8/H128` (`eps=0.02`) draws agreed on only `98/300` (`32.7%`) `L >= 600` decisions (`23.31–24.01 dec/s` on MPS at `N4/H32`).
- **Shadow parity & bisection (`M1` GO, `M2` `23.08×`):** `M1` had `0` diffs over `15,000` frames; `M2` bisection cut rollout work `23.08×` (`17,334.25` vs `400,000` row-steps/world), recovering `39/42` (`92.9%`) true `H=300` rescues while `33/75` T3 changes were `1`-, `2`-, or `4`-frame death delays (`M4` ONNX `2.49×` faster: `0.138 ms` vs `0.344 ms`).
- **Engine `A5 v2` (`GO`) & `S2` (`1.90×`):** Replacing Python `full_digest` (`1.71×`) and running `K=12` 1-thread `W=1` CPU processes reached `57.2` and `54.4 worlds/h/lane` on L4 (`~0.036 u/world`). Compiling `v8` (`56.8%` of `S0`'s `103.1 s` 6-world round) sped up the whole workload `1.90×` (`98.5 -> 51.9 s`, `144/144` byte-identical).
- **Round 0 & Claim A `D0` (`KILL`):** Round 0 hit `KILL / INTEGRITY_FAILURE` (`analyze` called `10 s` before pushing the replay ledger; `240` pairs sealed unread). Claim A's v1 loss flipped `0.4%` (`1/225`) of rescues, and amended `D0` hit `KILL` (no objective kept anchor guards `>= 512` updates).
- **Offense `O1/O2`:** `O1` passed `GO` (`35/48` worlds `>= +5` mass), while `O2` closed (`5,097` pairs / `~150` runtime-h needed for `80%` power at `+10`).

### Why
Small-sample `eps=0.02` rollouts were too noisy (`32.7%` self-agreement), `33/75` historical T3 overrides were doomed `1`–`4`-frame stalls, `horizon_rescue` rows (`225/190,779`) collided with ordinary states under single-frame features (`D0` `KILL`), and batch-1 CUDA (`4.35 ms`) lost to 1-thread CPU (`0.93 ms` PyTorch, `0.138 ms` ONNX).

### What to reuse / what not to repeat
- Use deterministic `eps=0` rolling shadows + bisection (`23.08×` less work) and filter out doomed `all_die_delay` stalls so only true `horizon_rescue` branches are kept.
- Run sequential rollouts on `K=12` 1-thread CPU workers per L4 with contiguous numpy digests (`1.71×`) and `S2` Numba `v8` (`1.90×`), not batch-1 CUDA.
- Do not repeat small-`N` `eps > 0` action ranking (`32.7%` agreement) or single-frame behavioral cloning of rare (`225/190,779`) multi-step rescues.

### Full record
Original: `git show 90b3db94:docs/research/history/t34-search-teacher.md` | Primary sources: [SUCCESS_CONTRACT.md](../final_run_20261019/SUCCESS_CONTRACT.md), [DESIGN_V2_SHADOW.md](../final_run_20261019/DESIGN_V2_SHADOW.md), [t34_shadow/RESULTS.md](../../../research/t34_shadow/RESULTS.md)

<a id="t34-pathb"></a>
## T34 Path B Paced 24 fps Background Search, Retention Repairs, and Unpaced Colab Arm

**Dates:** 2026-10-19 to 2026-10-21
**Verdict:** Closed `STOP_OPERATOR02_DO_NOT_RUN_PACED_STUDY` — the 80-doom real-time `24 fps` deliverability smoke (`t34_pathb_smoke`) delivered `0 / 34` (`0.0%`) horizon-surviving rescues at both `N=1` and `N=6` CPU threads, stopping the ratified `~18–19 h` `operator-02` Mac study before launch.

### What was tried
- Built Path B v1 (`H=40`, stride `5` at `L >= 600`, retired pre-data) and Path B v2 (`D=500` continuous exact shadow + earliest-first forward search in batches of `<= 8` branch points) for visibly labelled 24 fps online assistance (**Claim B**, `mixed-Watch6` update `p95 <= 11.688 ms`, `<= 5%` cadence loss).
- Sized and ratified `t34-retention-v2-operator-02` (`48` paired H5000 terminal worlds/mix = `144` total + `32` Watch6 sessions; `20,000`-rep OC power `0.80360`/`0.82905` at `+92` mass / `+0.25` survival and `20%` death cut, Type-I `0.0262 <= 0.06`).
- Amended `operator-02` pre-data to unpace the Mac `bare` arm (`ARM_FPS = {"bare": 0, "paced": 24}`, saving `~8.7–10 h`), offload the descriptive unpaced arm to Colab (`42.533128` u cap), and use a relative boost floor (`0.5 * s_bare <= s_candidate <= 0.10` vs `b8-s3`'s `0.12%` rate).
- Ran the pre-committed `57.2 min` deliverability smoke (`STOPPING_RULE.md`, `>= 50%` delivery bar) across `75` historical T3 changes (`34` D300 rescues + `41` delays) and `5` M3 Watch6 doom intervals at real `24 fps` deadlines under `N=1` and `N=6` worker threads.

### What happened
- **`N = 1` worker thread (`STOP`):** Delivered `7 / 75 = 9.33%` changes (all `7` delay-only, below `>= 50%` bar) and **`0 / 34 = 0.0%` rescues** (`0 / 5` M3). First improving candidates took `10.79–17.58 s` on full-lead (`~401–433` frame) T3 dooms, with `475` improving evaluations arriving already late (`293` tick overruns; `23.989–24.024 fps` M3 rate).
- **`N = 6` worker wrapper (`STOP`):** Delivered **`0 / 75 = 0.0%` changes** and **`0 / 34` rescues** (`0 / 5` M3); first full improvements slowed to `12.82–29.99 s` (`75.134 s` on M3) while M3 frame rate collapsed to `4.969–15.917 fps` (`21077` tick overruns, `166` late improvements).
- **Study disposition:** `t34-retention-v2-operator-02` (`~18–19 h` Mac + `42.533128 u` Colab) remained unstarted and `SNAKE_SERVE_PATHB` stayed default `OFF`.

### Why
Earliest-first CPU search over `D=500` needed `10.79–17.58 s` (`362.498 s` unpaced per H5000 world), missing real `24 fps` frame deadlines (`475` late improvements at `N=1`); for `N=6`'s fps collapse (`4.969–15.917 fps`), `RESULT.md` did not isolate the cause under the wrapper and recorded load (the `S4` design memo separately noted in-process threads share the GIL).

### What to reuse / what not to repeat
- Run a cheap (`57.2 min`) deadline-deliverability smoke on historical rescue windows before launching multi-hour (`~18–19 h`) paced retention studies.
- Run clock-independent `bare` baselines unpaced (`fps=0`, byte-identical across `1,300` ticks, `<= 3.0 h` vs `11.7 h` at `24 fps`), and isolate concurrent Python sessions in separate processes rather than multi-threading inside one GIL.

### Full record
Original: `git show 90b3db94:docs/research/history/t34-pathb.md` | Primary sources: [t34_pathb_smoke/RESULT.md](../../../research/t34_pathb_smoke/RESULT.md), [PROTOCOL_V2.md](../../../research/t34_pathb/PROTOCOL_V2.md), [PREREGISTRATION_RETENTION_V2.md](../../../research/t34_pathb/PREREGISTRATION_RETENTION_V2.md)

<a id="t35-probes"></a>
## T35 Auxiliary Outcome-Head Probe and Two-Deviation Rescue Probe

**Dates:** 2026-10-20
**Verdict:** Both Mac-only probes (`0` Colab units) closed `KILL` — the frozen-trunk `Linear(256, 12)` H300 outcome head missed both bars (`0.747` survival AUC vs `>= 0.80` GO / `<= 0.77` KILL; `46.2 %` teacher agreement vs `< 70 %` KILL, data-starved at `~235` rows/fold), and the `k = 8` two-deviation rescue beam rescued only `2/7` worlds (`0` novel escapes across `33` all-die events).

### What was tried
- Fit `t35_outcome_head` (`Linear(256, 12)` via full-batch L-BFGS with inner-CV L2, world-level 5-fold CV) on `b8-s3`'s frozen `256`-D fuse vector across the `39` A5 v2 worlds (`296` outcome-tested rows, `225` decisive rows / `1,350` `(row, action)` pairs in `21` worlds) against a B8 own-continuation control arm.
- Ran `t35_two_deviation` (`GridBatchSim`, layer-1 expansion -> beam `k = 8` -> layer-2 expansion, confirmed alive at `H=300` and `min(t2 + 600, 5000)`) on the `33` historical T3 `D=300` all-die events across `7` distinct worlds (`7` primary earliest-per-world events + `26` `EXTEND` events; `W >= 3/7` `REAL`, `W <= 1/7` `KILL`, `W = 2/7` `EXTEND`).
- Used both probes as zero-Colab-unit pre-screens to decide whether auxiliary H300 outcome supervision or deeper GPU two-deviation search deserved campaign funding.

### What happened
- **Outcome-head probe (`KILL`):** Teacher-arm per-action survival AUC was **`0.747`** (`95 %` CI `0.705 – 0.792`, firing `<= 0.77` KILL; control arm `0.428`, margin `+0.320`; raw B8 Q AUC `0.639`, 5-epoch Adam `0.550`), and outcome-argmax matched `a_star` on **`46.2 %`** (`39.1 – 52.2 %`, firing `< 70 %` KILL; chosen action survived on `60.9 %` of rows).
- **Two-deviation beam (`EXTEND` `2/7` -> pooled `KILL` `W_pool = 2/7`):** Primary run (`2,048,954` row-steps, `1,024,477`/rescue) rescued `2/7` worlds (`b8_s3_19@4693` censored at `35` frames; `b8_s3_9@4188` confirmed `600` frames), triggering `EXTEND`, where **`26/26` events were `NOT_RESCUED`** (`W_novel = 0`, `W_uncensored = 1`), firing pooled `KILL` and withholding GPU search funding.
- **Explicit hedges:** Outcome-head `KILL` (`~235` training rows/fold) holds for this frozen representation and data size, not as proof the horizon is unrecoverable from state; two-deviation `KILL` applies to the registered `k = 8` beam heuristic, not as proof that no two-deviation rescue exists.

### Why
Frozen `b8-s3` features (`256`-D single-frame representation on `225` decisive rows) plateaued at the `0.747` reactive AUC ceiling (`46.2 %` action match), while the `k = 8` beam recovered only the `2` worlds where T3's own history already contained a multi-deviation escape and found `0` layer-2 `t2 + 300` survivors in the other `5` worlds.

### What to reuse / what not to repeat
- Gate expensive GPU search ports and auxiliary student heads behind `< 1 h`, `0`-unit Mac falsifiers on existing rollout banks (`39` A5 worlds / `33` T3 all-die events).
- Do not expect linear readouts of `b8-s3`'s frozen `256`-D trunk (`0.747` AUC, `46.2 %` agreement) or narrow `k = 8` two-deviation beams (`W_novel = 0`) to replace online rollout planning.

### Full record
Original: `git show 90b3db94:docs/research/history/t35-probes.md` | Primary sources: [t35_outcome_head/RESULT.md](../../../research/t35_outcome_head/RESULT.md), [RESULT_PRIMARY.md](../../../research/t35_two_deviation/RESULT_PRIMARY.md), [RESULT_EXTEND.md](../../../research/t35_two_deviation/RESULT_EXTEND.md)

<a id="t34-round0"></a>
## T34 Round 0 Sparse-Teacher (`teacher_v2`) & Round 0-D Dense-Teacher Premise Tests (`t34-round0`)

**Dates:** 2026-10-20 to 2026-10-21
**Verdict:** Closed `KILL / INTEGRITY_FAILURE` pre-look — all `240` paired H5000 worlds completed and passed `10%` (`24`-world) Mac replay, but `analyze` was invoked before publishing the replay ledger append, sealing all `240` pair outcomes unread and leaving fallback `t34_round0d` (`DRAFT`) unactivated.

### What was tried
- Pre-registered `t34_round0`: a `240`-pair H5000 test (`96` scripted, `72` frozen, `72` mixed in `100/100/40` L4 chunks, `32.64 u` initial / `65.28 u` max liability) of deterministic `teacher_v2` (`H = 300`, shadow `D = 300`, bisection + radius `3`) vs bare `b8-s3`.
- Set GO bars requiring scripted survival `>= 0.66` (`>= +0.10` gain), equal-mix mass gain `>= +40` (`LB90 > 0`), survival guard A (`LB95 > -0.05` pooled, `> -0.075` per mix, `>= 0.30` absolute), and a pre-look amended relative boost floor (`0.5 * bare_share <= teacher_share <= 0.10`).
- Ran `20,000`-rep operating-characteristic (`OC`) simulations across `5` historical/normal models, verified 3-way physical worker parity (`09d00b7a304d861f880d52e14c8ad9d22567770667ae6991574586110fd12917`), and attested mechanics v2 (`ATTESTED_MECHANICS_V2`).
- Prepared contingent `t34_round0d` (`DenseTrigger(600, 5)`, `272` disjoint seeds, `120 u` ceiling, `89/89` tests passing) in case Round 0 returned a scientific `STOP`/`INCONCLUSIVE`.

### What happened
- **Operating characteristics & sizing:** Protected-regression joint Type-I error stayed `0.00750–0.03670 <= 0.06` across all `5` models; joint mass+survival pass at true `+40` was `0.23945` / `0.26175` (mass-only `0.93195` / `0.92905` at `+60`), and `A5` L4 ran at `57.173` and `54.372` worlds/h/lane (`>= 50` bar).
- **Pre-look boost amendment (`30b2739a`):** Replaced the absolute `[0.005, 0.10]` (`0.5%`) boost floor after measuring bare `b8-s3`'s natural boost share at `0.12%` over `190,779` decisions.
- **Terminal settlement (`KILL / INTEGRITY_FAILURE`):** All `240` pairs sealed `COMPLETE` and `24` Mac replay worlds passed, but `analyze` ran before the replay's ledger append was published to remote history; `require_published_history` raised `HistoryError` before reading any pair file, sealing all `240` outcomes unread and leaving `t34_round0d` unactivated.

### Why
`analyze` enforced a strict remote-git publication check (`require_published_history`) as its first step and mapped `HistoryError` directly to terminal `KILL` when the operator ran `analyze` before pushing the replay ledger commit.

### What to reuse / what not to repeat
- Calibrate behavioral guard floors against the parent checkpoint's measured rate (`b8-s3` boosts on `0.12%` of decisions, below a naive `0.5%` absolute floor).
- Automate ledger publication inside the phase script rather than relying on manual operator `git push` ordering between replay and `analyze`, and never map a pre-look git check to an unrecoverable study `KILL`.

### Full record
Original: `git show 90b3db94:docs/research/history/t34-round0.md` | Primary sources: [PREREGISTRATION_ROUND0.md](../../../research/t34_round0/PREREGISTRATION_ROUND0.md), [t34_round0/README.md](../../../research/t34_round0/README.md), [VERIFICATION_ROUND0D.md](../../../research/t34_round0d/VERIFICATION_ROUND0D.md)

<a id="t34-o1-o2-harvest"></a>
## T34 Offense-vs-Defense Audit, O1 Offensive Harvest, and O2 Bank (prepared, never run)

**Dates:** 2026-10-20 to 2026-10-22
**Verdict:** The 36-world food-economy audit refuted the offense-pivot premise (`b8-s3` already took `39.71%` of gross mass from death drops under a `150`-pellet cap), `O1` passed `GO` locally (`35/48` worlds had a `>= +5` death-drop H500 branch), and closed-loop `O2` (`1,260`-world fallback vs `5,097` needed for `.80` power) closed `PARKED` / `UNUSED` without running a study world.

### What was tried
- Audited mechanics-v2 corpse rules and ran `36` instrumented H5000 terminal-hero `b8-s3` worlds (`12`/mix, `164,514` acted frames) tracking ambient, death-drop, and boost-trail intake to test whether offensive kill/corpse harvesting had larger upside than defensive survival search.
- Ran **O1** (`research/t34_o1_harvest`): a `48`-world (`16`/mix, `240` roots, `1,306` branches) L4 counterfactual census testing whether single-action H500 deviations capture `>= +5` death-drop mass (KILL bar `< 8/48`, frozen+mixed bar `>= 4`), after fixing timing-fixture roots (`10, 20`) when initial timing seeds died at `4479` and `2216` steps.
- Prepared **O2** (`research/t34_o2_harvest`): a closed-loop H5000 comparison of bare `b8-s3`, rescue-only `D`, and rescue+harvest `D+O` (`(D+O - D)` `LB90 > 0` and `>= +40` vs B8), resizing from `96` to `1,260` worlds (`420/mix`, `149.733333 u` cap) and repairing upload/CUDA-mask staging (`V3`).

### What happened
- **36-world `b8-s3` economy audit:** Bare `b8-s3` earned `332` kills (`18.083`/game scripted, `4.083` frozen, `5.500` mixed) with `0.125%` (`206`) boosted frames and **`39.71%` (`16,467`) of gross mass from death drops** (`50.53%` / `20,952` ambient, `4,049` trails). Mechanics v2 caps corpse+trail pellets at **`150`** (`1` pickup/tick; all `9` deaths at `L >= 800`, up to `1,454`, kept `150`).
- **`O1` harvest census (`GO`):** **`35/48` worlds** (`13` scripted, `9` frozen, `13` mixed; `22` frozen+mixed; `71/240` roots) had a surviving single-action H500 branch gaining `>= +5` death-drop mass (`parity_pass: true`).
- **`O2` closed-loop bank (`PARKED` / `UNUSED`):** Measured T23 SDs required **`5,097` worlds** for `.80` (`80%`) joint power at `+10` incremental mass; the `1,260`-world budget fallback had `.339888` capped power, hit upload (`1,195,740` B -> `16,655` B URL fix) and `CUDA_VISIBLE_DEVICES=""` bootstrap stops, refused chunk `00` on its L4 timing forecast, and closed unrun.

### Why
Keeping a long body alive dominates the total-horizon mass integral while the `150`-pellet FIFO cap and one-pickup-per-tick ceiling bound corpse-harvesting upside; detecting a `+10` incremental closed-loop mass lift over `D` against cross-world variance required `5,097` paired worlds (`O2`).

### What to reuse / what not to repeat
- Measure baseline food provenance (`39.71%` death drops) and simulator caps (`150` pellets) before assuming an unexploited gameplay economy, and apply `CUDA_VISIBLE_DEVICES=""` after GPU `bootstrap.sh` checks (`16,655` B URL catalog vs `1,214,804` B CLI upload).
- Ensure excluded timing-fixture roots (`10, 20`) lie well inside minimum trajectory survival rather than fixed late steps (`3000`) where a timing seed may already be dead (`2216`).

### Full record
Original: `git show 90b3db94:docs/research/history/t34-o1-o2-harvest.md` | Primary sources: [2026-10-21_offense_vs_defense/JUDGMENT.md](../advice/2026-10-21_offense_vs_defense/JUDGMENT.md), [PREREGISTRATION_O1.md](../../../research/t34_o1_harvest/PREREGISTRATION_O1.md), [t34_o2_harvest/README.md](../../../research/t34_o2_harvest/README.md)

<a id="t34-big-lever"></a>
## T34 Big-Lever Advisor Program (Rounds 01–37), Plans v1–v3, V0 Latency Ladder, Step −1 Ruler, and M3 Evidence Boundary

**Dates:** 2026-10-20 to 2026-10-25
**Verdict:** Offline periodic lookahead (`L2'`) proved a real survival-driven gain (`FUND_SYSTEMS`), and the compiled opponent veto kernel (`K-1`) sped up rollouts `20.13×`, but co-resident live Mac search (`M1`, `GC A1`) failed frame-pacing bars so live serving stayed on bare `b8-s3` (`KEEP_B8`).

### What was tried
- Chartered a multi-round advisor and judge program (`CHARTER.md`, Rounds `01`–`37`, `PLAN.md` through `PLAN_v3.md`) to find one large lever using Colab GPUs that produces a Mac-served snake beating bare `b8-s3` by at least `+40` five-thousand-frame (`H5000`) mass without hurting survival.
- Evaluated the `L2'` offline search ceiling (unchanged T3 dense search: depth `300` every fifth decision at length `>= 600`) against bare `b8-s3` on `160` fresh paired H5000 worlds (`64` frozen, `64` mixed, `32` scripted).
- Built and tested live Mac serving enablers: `K-1` (`research/bl_k1`, native Numba kernel for `opponent_v8` safety checks inside rollouts), `M0` (rescue lead-time census), `M1` (co-resident `24 fps` frame-pacing test), `GC A1` (`gc.freeze()` Python garbage-collection mitigation), `V0` (Mac latency ladder), and `M3` (`bl_evidence`, clock-seam and strict-pin validator).
- Ran Wave-1 falsifiers across alternative levers: `O1`/`O2` (500-step corpse-harvest search), `E1b`/`A1` (early GPU simulator speed), `E3` (full-board geometry and history probes), `DENSITY-2` (shrunk maps or extra opponents), `E9` (power gate), and `P1`/`E8` (long-horizon PPO critic gate).

### What happened
- **Offline search (`FUND_SYSTEMS`) & `K-1` speedup:** `L2'` beat bare `b8-s3` by `+83.0` pooled mass (`LB90 +64.1`; `+56.1` frozen, `+27.6` mixed, `+165.3` scripted) with `28` rescues and `0/160` negative worlds while changing `0.07–0.52` actions per `1,000`. `K-1` passed (`294/294` identity, `20.13×` speedup vs `>= 5×` bar, cutting frozen batch p50 `19.5–21.4 s → 2.3–2.4 s`).
- **Live Mac serving (`KEEP_B8`) & `M3` boundary:** `M0` found lead window `L* = 72` (`F = 0.944`), but `M1` printed `FINAL FAIL` (`5–26` overruns/cell; bare game alone overran `43×` at `13.2 ms` update p95), `GC A1` failed (`13.8–23.3 ms` p95), `V0` smoke was invalidated, and `M3` (`bl_evidence`, `123` tests, merged `65722b9d`) kept qualification `NOT_GRANTED` / `NOT_READY`.
- **Wave-1 gates:** Prior `Path B v2` (`STOP`, `0/34` rescues), `D0` (`KILL`), and `Round 0` (`KILL` / `INTEGRITY_FAILURE` pre-look) stood. `O1` passed (`35/48` worlds) while `O2` needed `>= 5,097` worlds (`NOT DISPATCHED`); `E1b` hit `0.217` steps/s vs `>= 20,000` (`NO_CONTINUE`) and `A1` `~160` vs `2,000` (`STOP_ENGINE`).
- **Additional falsifiers:** `E3` failed (`LB90 −5.41`, history `−18.2%`, `KILL`); `DENSITY-2` rejected all `9` cells (`TERMINAL`); `E9` blocked two-arm PPO (joint funding `.27`/`.31` vs `.60`); `P1` critic passed (`EV 0.587`, `MSE 296k` vs `528k`) with continuation held.

### Why
`L2'` gains came from rare self-trap rescues (`85%` `DEATH_SELF`) rather than frequent tactical edits, while live co-resident Mac serving failed because CPython generation-2 garbage-collection pauses (`>= 20 ms` on `86/89` overrun ticks) stalled the `24 fps` loop and `gc.freeze()` triggered `4×` more collections instead of removing them.

### What to reuse / what not to repeat
- Reuse `K-1` native Numba `opponent_v8` (`20.13×`), `bl_evidence` clock/pin checks, and the standing `JUDGMENT_RESUME_POLICY.md` requirement for idempotent completed units with verified-ledger resume-skip.
- Quote `frozen`/`mixed` gains (`+28` to `+56` times delivered fraction) rather than the scripted-inflated `+83` pooled headline when sizing product impact.
- Do not repeat map shrinking (`DENSITY-2`), uncalibrated multi-arm screens (`E9`), or co-resident CPU search inside the live Mac `24 fps` server process.

### Full record
Original: `git show 90b3db94:docs/research/history/t34-big-lever.md` | Primary sources: [CHARTER.md](../advice/2026-10-21_big_lever/CHARTER.md), [PLAN_v3.md](../advice/2026-10-21_big_lever/PLAN_v3.md), [big-lever-experiments.md](../../notes/big-lever-experiments.md)

<a id="bl-served-search-l2"></a>
## Big-Lever Served Search Ceiling (`L0′` / `L2′`) and PLAN_v3 Follow-Up Packages (`M0` / `B2` / `C1`), plus `X5` Bare Mac Timing (`bl-served-search-l2`)

**Dates:** 2026-10-21 to 2026-10-22
**Verdict:** Offline exact-stop 300-step search (`L2′`) beat bare `b8-s3` by `+83.00` pooled H5000 mass on `160` paired development worlds (`FUND_SYSTEMS`), while L4 CPU placement was `NOT_DISPATCHABLE`, live co-resident serving failed `M1` (`KEEP_B8`), `B2`/`C1` stayed prepared without running, and `X5` ended `INCOMPLETE` / `NOT_READY`.

### What was tried
- Tested `L2′` (development-only offline ceiling for the unchanged T3 dense-search rule: check every fifth decision at length `>= 600`, evaluate legal actions to depth `300` with `exact_stop` early exit on the first survivor, vs bare B8 s3) across `160` paired H5000 worlds (`64` frozen, `64` mixed, `32` scripted).
- Ran `L0′` saved-smoke deadline bookkeeping on `80` last-avoidable-decision (`LAD`) records (`34` rescue-capable, `41` delay-only) and measured L4 vs Mac native+exact-stop batch throughput against the `R >= 2.18` batches/s admission bar.
- Ran `M0` (rescue lead-time census on the `160` `L2′` worlds) and prepared `B2` (`32` Watch6 `14,400`-decision trajectories for `L >= 800` natural escape roots), `C1` (`24` paired `14,400`-frame Watch6 sessions), and `X5` (`research/bl_x5`, bare-B8 Mac timing adapter).

### What happened
- **Throughput rulers:** Stock-hook L4 search measured `0.166` batches/s vs the `2.18` admission bar (`NOT_DISPATCHABLE` on L4), whereas the Mac native+exact-stop ruler passed with zero-tolerance chosen-action parity, admitting Mac `L2′` execution after correcting the opponent-veto premise (`veto_slots=()`).
- **`L2′` offline ceiling (`FUND_SYSTEMS`, development only):** Cleared the `>= 40` pooled point and `LB90 > 0` frozen/mixed bars: pooled-equivalent `+83.00` (`+83.0`, `LB90 +64.12` / `+64.1`, `UB90 +101.87`), frozen `+56.12` (`LB90 +34.81`), mixed `+27.60` (`LB90 +12.11`), scripted `+165.27` (`LB90 +114.77`).
- **`L2′` survival & edits:** Pooled no-death delta was `+0.198` (`LB90 +0.153`) and alive-frame fraction `+0.075` (`LB90 +0.058`), changing `54` / `21` / `79` actions (`0.18` / `0.07` / `0.52` per `1,000` decisions) for `13` / `5` / `10` (`28` total) productive rescues and `0` newly caused deaths.
- **Follow-ups:** `L0′` logged `0/62` own-deadline hits (`29/34` D300 reachable at `1` s). `M0` recorded `L* = 72`, `F = 0.944`, before co-resident serving failed `M1` (`FINAL FAIL` / `KEEP_B8`). `B2` (`52` mocked tests) and `C1` were `PREPARED, NOT RUN`; `X5` stopped after `55.12 s` for overlapping the K-1 audit (`INCOMPLETE` / `NOT_READY`).

### Why
Evaluating bare B8 first and stopping at the first H300 survivor (`exact_stop`) preserved exact chosen actions while cutting Mac runtime enough to measure `160` worlds; the `+83.00` gain came from sparse (`0.07–0.52` per `1,000` moves) self-trap rescues with `0` newly induced deaths, while stock-hook L4 CPU rollouts (`0.166` batches/s) missed the `2.18` threshold.

### What to reuse / what not to repeat
- Reuse `exact_stop` ordering (test bare B8 action first, then descending Q, stopping on the first H300 survivor) for zero-tolerance speedup of survival-ranked rollout teachers.
- Ensure snapshot-based branch workers serialize the immutable `AppConfig` alongside simulator state (`WatchDriver.snapshot()` erratum in `B2`); otherwise fresh spawned workers revert to mechanics v1 defaults and fail rule checks.
- Quote `frozen`/`mixed` gains (`+28` to `+56` times delivered fraction) rather than the `+83` pooled headline, and do not place uncompiled Python search loops on Colab L4 vCPUs.

### Full record
Original: `git show 90b3db94:docs/research/history/bl-served-search-l2.md` | Primary sources: [L2prime_mac_analysis_20261022.json](../results/L2prime_mac_analysis_20261022.json), [PREREGISTRATION_L2.md](../../../research/bl_served_search/PREREGISTRATION_L2.md), [bl_served_search/README.md](../../../research/bl_served_search/README.md)

<a id="bl-served-tick"></a>
## Big-Lever Served-Tick Timing Diagnostics, Mac Y1/L1-prime Latency Rulers, and Paced Identity (`bl-served-tick`)

**Dates:** 2026-10-21 to 2026-10-28
**Verdict:** Built timing rulers and host-load admission guards with no efficacy or serving qualification result: `L1′` Mac latency returned `NOT_READY(INSTRUMENT_MISSING)`, three-oracle paced identity stayed `BUILD_VERIFIED_NOT_QUALIFIED`, and served-tick panel attempts `07` and `08` ended `INVALID`.

### What was tried
- Built `bl_paced_identity` (`X7` and Round-09 `Z-2`), a three-oracle harness testing paced search interleaving with integer-nanosecond request cutoffs (`publish_cutoff_ns` = target tick − `2` ticks, `consume_cutoff_ns` = target tick + `1` tick) across four clock-seam cases.
- Ran `bl_mac_latency` (`Y1` dense H300 batch ruler across `9` length `800`/`1200`/`1500` fixtures at CPU `W=1`, plus `10` CPU/MPS `L1′` co-resident stress cells) against `24 fps` Watch6 latency ceilings (`update p95 <= 11.688 ms`, zero overruns, `S p99 <= 0.830–1.458 s`).
- Built the `12`-cell bare Watch6 `served_tick_probe.py` timing panel (`3` seeds × `4` arms, `200` warm-up + `2,400` measured ticks at `24 fps`), host-load classification rules (`classified-v1`, `classified-desktop-v1`, `classified-supervised-v2`), and an opt-in inspector/WebSocket JSON process split (`dto_split_w0_k0`).

### What happened
- **`bl_paced_identity` (`BUILD_VERIFIED_NOT_QUALIFIED`):** Mac preflight (`26` synthetic fixtures, `43` tests, plus `72` `Z-2` clock-seam tests) passed, but the `18+96` production root bank and paced oracle remained `NOT_RUN` / `INCOMPLETE`.
- **`Y1` batch ruler:** Contended CPU `W=1` H300 p50/p95 batch latencies were `10.557 / 10.607 s` to `21.221 / 25.814 s` on `frozen` (`22.749 / 23.311 s` at `frozen-1200`), `7.603 / 7.642 s` to `10.823 / 10.925 s` on `mixed`, and `0.595 / 0.626 s` to `0.649 / 0.681 s` on `scripted`.
- **`L1′` stress cells (`NOT_READY(INSTRUMENT_MISSING)`, nominated cell: none):** Hit `S p99` of `22.580–27.244 s` on CPU (`update p95` `39.744–48.271 ms`, `280–390` overruns) and `34.600–41.202 s` on MPS (`291–364` overruns); quiet-host retries refused or aborted on `mediaanalysisd`/`Spotlight` and `analyticsd` (`63.7%` core).
- **Served-tick panel & DTO split:** Attempts `07` (`24.38 s`, `ps` at `28.0%`) and `08` (`259.08 s`, Outlook at `26.2%` after cells `00–02` of `12`) ended `INVALID` on foreign-load guards; live remote supervision remained `IMPLEMENTATION INCOMPLETE — LIVE ADMISSION UNAVAILABLE`, and the `dto_split_w0_k0` panel (`501` tests passed) was not run.

### Why
`L1′` ran before the native `S2`/`K-1` opponent-veto kernel was enabled — making stock-Python `frozen` and `mixed` H300 batches roughly `12×–38×` slower than `scripted` — and lacked a production reachable-root provider to measure consumption latency `C`; panel attempts `07` and `08` tripped strict `>= 20%` single-core foreign-process guards before desktop apps were classified separately.

### What to reuse / what not to repeat
- Reuse explicit host-load classification (`>= 20%` single-core abort for unknown/foreign workloads, shared `> 150%` for `> 60 s` background ceiling for macOS services and enrolled desktop apps) instead of killing or renicing user processes.
- Reuse the two-cutoff integer-nanosecond request contract (`publish_cutoff_ns`, `consume_cutoff_ns`) whenever testing asynchronous helper processes against a fixed-rate game loop.
- Do not salvage or splice partial timing cells from `INVALID` host-contended runs, and do not benchmark MPS for co-resident `L1′` search (`34.600–41.202 s` `S p99` vs `22.580–27.244 s` CPU).

### Full record
Original: `git show 90b3db94:docs/research/history/bl-served-tick.md` | Primary sources: [RESULT_L1.md](../../../research/bl_mac_latency/RESULT_L1.md), [SERVED_TICK_README.md](../../../research/bl_served_search/SERVED_TICK_README.md), [bl_paced_identity/README.md](../../../research/bl_paced_identity/README.md)
