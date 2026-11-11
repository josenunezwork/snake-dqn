# Inputs and Networks

<a id="apex-small-task"></a>
## Apex Small-Task Mechanism & Terminal-Mask Diagnostic Series (2026-09-22 to 2026-09-24)

**Dates:** 2026-09-22 to 2026-09-24
**Verdict:** Fresh `vector61` `ApexLearner` (`BC0`) solved `H=8` and `H=16` (`96/96` in `3/3` seeds) where `BC512` warm-starts regressed, `2x2` width/breadth scaling (`512` vs `1,024` units × `96` vs `384` roots, `240,000` updates) gave `0/4` paired gains (`NO_RELIABLE_SCALING_BENEFIT`), and `14/15` `H=256` self-collisions picked space `0.03125` over `1.0` one move before trapping.

### What was tried
- Transferred the controlled food microbenchmark to `vector61` `ApexNetwork(61, 512, 6)` and canonical `ApexLearner` (PER `alpha=0.6` + `3`-step TD + target net), comparing `512`-update BC (`BC512`) warm-starts vs from-scratch `BC0` across `5,000` TD updates (`H=8`) and zero-shot `H=16` / `H=64` transfer (`3` seeds).
- Tested `5,000`-update `H=64` continuations (`eps=0.4` vs `0.1`, `H16/H64` replay mix, uniform `alpha=0.0` replay) plus saved-record loop, coverage, teacher-geometry, and `gamma=0.99` finite-return (`G64`) audits.
- Ran a `2x2` fresh-start factorial (`w512` = `429,831` params vs `w1024` = `1,646,087` params × `n96` vs `n384` roots × `3` seeds × `20,000` updates = `240,000` updates) evaluated on `H=64` and `96`-case `H=256`.
- Continued `w512_n96` to mark `40,000` (`H=64` vs `H=256` collection) and audited terminal masks and `t_death - 1` choices across all `576` `H=256` games (`146,609` frames).

### What happened
- **From-scratch TD vs naive BC warm-start (`H=8`, `H=16`):** Both `BC512` and from-scratch `BC0` (`5,000` TD updates) hit **`96/96` held food & survival in `3/3` seeds** (with `BC0` transferring `96/96` to `H=16`), whereas continuing TD from `BC512` regressed to `26/9, 17/15, 64/62` train/held food (`max Q` `11.66`–`14.67` vs `0.247`–`0.426`).
- **`H=64` ablations & `2x2` scaling (`NO_RELIABLE_SCALING_BENEFIT`):** `eps=0.1`, `H16/H64` mix, and `alpha=0.0` all failed joint retention. Across `240,000` scaling updates, `11/12` cells passed `H=64` HOLD (`289`–`303` vs teacher `304`), but **`0/4` paired width/breadth contrasts** passed across seeds and **`0/12` cells passed `H=256`** (`830`–`1,007` food vs `1,046`; `92/96`–`96/96` survival).
- **Pre-trap localization (`14/15` at `t_death - 1`):** All **`15/15` `H=256` deaths** occurred on the exact first empty-safe-mask frame, and on `t_death - 1`, **`14/15` snakes chose a move with recorded free-space `0.03125` while a safe normal neighbor had `1.0`** (`20/20` rescued in `2`-frame provisional rollouts).

### Why
`ApexLearner` with PER and target networks learns short-horizon foraging reliably, whereas `3`-step (`1`-step mask) TD rewards fail to penalize entering `0.03125`-space pockets until the trap closes — a horizon-weighting bottleneck that widening the MLP (`512 -> 1,024`) cannot fix (the `BC512` `max Q` gap is reported as descriptive, not causal proof).

### What to reuse / what not to repeat
- Enforce free-space preferences directly via serving-time vetoes (`v2`–`v8`) or multi-step/teacher safety labels rather than widening `vector61` MLPs (`0/4` gains across `240,000` updates).
- Audit held-bank input disjointness before running continuations (`11`, `1`, and `48` overlaps caught in qualification stops).

### Full record
Original: `git show 90b3db94:docs/research/history/apex-small-task.md` | Primary sources: [TASK_ALIGNED_AND_APEX_SERIES.md](../../experiments/TASK_ALIGNED_AND_APEX_SERIES.md), [table.md](../../experiments/apex_terminal_mask_2026-09-23/table.md), [provisional-table.md](../../experiments/apex_native_pretrap_counterfactual_2026-09-23/provisional-table.md)

<a id="bl-info-value"></a>
## Big-Lever E3 Full-Board/History Information-Value Probe and Round-31 Bounded Future-Noise Audit (`bl-info-value`)

**Dates:** 2026-10-21 to 2026-10-24
**Verdict:** `E3` (`bl_info_value`) ended `SMOKE_COMPLETE / PREPARED, NOT DISPATCHED` after three Colab attempts were repaired with `1,954,150 B` Mac action tapes (no production run or scientific GO/KILL in this record), and Round-31 (`bl_future_noise`) closed cleanly at `STOP_AT_A0` (`BLOCKED_MISSING_CLEARANCE`, `0` admitted parents, `0` Colab units).

### What was tried
- Built `E3` (`research/bl_info_value`) on `256` unsealed T3 worlds (`1,536` roots: `768` decisive, `768` counterweight; `192/32/32` split), parameter-matched at `270,522` parameters across three students.
- Tested whether full `145 × 83` 6-plane occupancy + unclipped tail release (`Student B`) or a 128-decision 64-unit GRU (`Student C`) cuts D300 mass regret `>= 20%` at `<= 1%` counterweight switches vs `b8-s3` inputs (`Student A`).
- Repaired three successive `E3` Colab L4 failures across Recoveries `02`–`04`: `57,831,386 B` upload timeout, Colab exec RPC timeouts, and a Mac-vs-x86 float32 Q-tie action divergence at decision `2916`.
- Built and ran the Round-31 Future-Noise Audit (`research/bl_future_noise`), testing whether frozen rescue actions still beat bare actions across four independent CRN future-RNG suffixes (`A2`), gated by a pre-data `>= 64`-parent / `>= 16`-pair cleared-archive census (`A0`).

### What happened
- **`E3` Recoveries 02–03:** Attempt `01` (`334.09 s`, `0.2 u`) timed out uploading `57,831,386 B` (`57.8 MB`), fixed in Recovery `02` via a `360,881 B` catalog (`160x` smaller) + asset `bl-e3-inputs-20261021` (`51,787,546 B`). Attempt `02` (`1,445.43 s`, `33/256` worlds, `1.1 u` cumulative) hit RPC timeouts, fixed in Recovery `03` (`2` reserved vCPUs, `nice >= 10`, scratch outside `exports/`).
- **`E3` Mac action tapes (`SMOKE_COMPLETE / PREPARED, NOT DISPATCHED`):** Attempt `03` (`2,318.74 s`, `2.4 u` cumulative) stopped `INCOMPLETE` on an x86 root mismatch at decision `2916`. Recovery `04` recorded Mac action tapes (`1,954,150 B` v3 archive in `588.08 s`), reproducing `2916` in `18.80 s` and cutting smoke `60.73 s -> 25.74 s` (`1.31 h / 2.62 u` L4 forecast; not dispatched here).
- **Round 31 (`STOP_AT_A0`):** Passed `16/16` boundary tests (`0.21 s`) and stopped at `A0` (`0.048 s`, `0` admitted parents, `0 u`) because no separately cleared, unsealed historical archive with `>= 64` parents and `>= 16` same-mix pairs existed.

### Why
Re-running neural policies closed-loop on Colab x86 to reconstruct Mac-selected trajectory roots failed at decision `2916` because arm64 vs x86 float32 Q-value ties flip greedy actions; earlier attempts blew the upload cap (`57.8 MB`) or starved the Colab Jupyter RPC loop under unreserved vCPU saturation and `exports/` scratch churn.

### What to reuse / what not to repeat
- Record compact all-seat Mac action tapes (`trace_bank.py`, `1,954,150 B` / `< 2 MB` for `256` worlds) whenever x86 workers must reconstruct exact Mac-selected states without float32 Q-tie drift.
- Reserve `2` vCPUs (`resources.py`), keep per-world scratch files outside the watched `exports/` directory, and fetch large static checkpoint bundles via HTTPS rather than `57.8 MB` CLI uploads.
- Enforce fail-closed pre-data archive clearance checks (`A0`) before spending Colab compute on historical root banks.

### Full record
Original: `git show 90b3db94:docs/research/history/bl-info-value.md` | Primary sources: [bl_info_value/README.md](../../../research/bl_info_value/README.md), [PROTOCOL.md](../../../research/bl_info_value/PROTOCOL.md), [bl_future_noise/README.md](../../../research/bl_future_noise/README.md)

<a id="a100-a2"></a>
## Big-Lever A2 Historical-Label Model-Scale and Hazard-Target Falsifier on A100

**Dates:** 2026-10-22
**Verdict:** Completed all `480` fold-fit units on A100 with `scale = NOT_CLOSED_EXPLORATORY` and `target = KEEP_SINGLE_HORIZON`: frozen-B8 linear features (`0.767` D300 AUC) beat fresh `1.64M` (`0.706`) and `6.68M` (`0.711`) encoders while parent-world volume added `+0.02` to `+0.05` AUC, and the 5-interval hazard head missed the `>= 10%` Brier improvement bar.

### What was tried
- Pre-registered a `480`-unit factorial (`2` datasets × `4` encoders × `2` heads × `3` parent-world doses `{25%, 50%, 100%}` × `2` seeds × `5` folds, `20` epochs, AdamW `lr=0.001`) on historical `T5`/`T8` (`19,976` states / `103,264` branches / `342` worlds) and excluded-`A5` (`296` rows / `225` decisive in `21` of `39` worlds).
- Compared `4` encoders — frozen-B8 256-D linear probe (`c0`), B8-initialized `1.64M` (`s_b8`), random-init `1.64M` (`s_random`), and wider/deeper random-init `6.68M` (`m_random`, `6,680,134` params) — against the scale-closure rule (`m_random - s_random < 0.02` AND world-bootstrap LB95 `<= 0` AND `50% -> 100%` slope `< 0.01`).
- Compared a 5-interval censored discrete-hazard head (`{30, 60, 100, 200, 300}`, normalized via `alive_steps + 1`) against a single-horizon D300 binary head (`>= 10%` relative D300 Brier reduction bar).
- Repaired three staging/runtime failures across Packs `02`–`06` (`pydantic` import miss, `352.60 s` `UPLOAD_CAP`, and concurrent endpoint accounting race).

### What happened
- **Encoder scale (`scale = NOT_CLOSED_EXPLORATORY`):** At `100%` `T5` parent-world dose, D300 death-prediction AUC was `0.767` for frozen-B8 linear probe `c0` (best overall), `0.719` for `s_b8` (`1.64M`), `0.711` for `m_random` (`6.68M`), and `0.706` for `s_random` (`1.64M`).
- **Volume vs parameters:** Scaling parameters (`1.64M -> 6.68M`, `0.706 -> 0.711`) gave no meaningful lift, whereas increasing parent-world dose (`25% -> 50% -> 100%`) added `+0.02` to `+0.05` AUC.
- **Hazard target (`target = KEEP_SINGLE_HORIZON`):** The `{30, 60, 100, 200, 300}` discrete-hazard head missed the `>= 10%` relative D300 Brier score improvement over single-horizon D300 supervision.
- **Import & transfer repairs (`COMPLETE` on Pack-06):** Added `tools.ops.import_preflight` (`python -I` checking `541`–`545` imports across `179`–`180` modules after Pack-02 failed at `792.49 s` / `2.2014 u` on missing `pydantic`). Deduplicated `103,264` `T4e` rows to `19,976` observations with XZ-9 (`91.49 MB -> 34.18 MB`, `12/12` array parity) after Pack-03 hit `UPLOAD_CAP` at `352.60 s`.

### Why
Training `1.64M`–`6.68M` spatial encoders from scratch on `19,976` states (`342` worlds) overfit compared to pretrained `b8-s3` features (`0.767`), while single-frame local observations capped all tested models at `<= 0.767` D300 AUC regardless of `6.68M` capacity or multi-horizon hazard intervals.

### What to reuse / what not to repeat
- Run `tools.ops.import_preflight` (`python -I` in a clean venv) before Colab dispatch to catch lazy transitive imports (`src.core.config_loader -> pydantic`) at zero GPU cost.
- Deduplicate shared root-state observations before XZ-9 archiving (`91.49 MB -> 34.18 MB`) to stay well below the `300`-second Colab upload cap.
- Prefer broader parent-world diversity (`+0.02` to `+0.05` AUC) over wider single-frame encoders (`6.68M` vs `1.64M`) or multi-interval hazard heads when fitting D300 survival probes.

### Full record
Original: `git show 90b3db94:docs/research/history/a100-a2.md` | Primary sources: [a100_a2/README.md](../../../research/a100_a2/README.md), [PREREGISTRATION.md](../../../research/a100_a2/PREREGISTRATION.md), [HANDOFF_TRANSFER_20261022.md](../../../research/a100_a2/HANDOFF_TRANSFER_20261022.md)
