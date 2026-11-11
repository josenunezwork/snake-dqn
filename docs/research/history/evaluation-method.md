# Evaluation Method

<a id="sequential-gate"></a>
## Sequential Strict Gate, Phase R Protocol, and Two-Tier Gate Calibration

**Dates:** 2026-09-26 to 2026-10-16
**Verdict:** Adopted and active: the `4`-look Lan–DeMets O'Brien–Fleming sequential strict gate (`strict-sequential-obf-bonferroni-v1`) and paired survival band `v2` (`pooled_ni_continue`, ratified `2026-10-06`) cut expected evaluation episodes by `27–75%` while controlling familywise error `<= 0.0492` (`sequential_phase_r` remains opt-in simulation/code).

### What was tried
- Calibrated `H5000` mass decomposition, cross-policy pairing correlation, and Lan–DeMets OBF futility spending over `144` saved episodes (`48` dev Apex + `48` candidate `6102` + `48` paired Apex; `85/96` Apex self-deaths: `40/48` dev, `45/48` pilot) in `gate_calibration_20260926`.
- Validated the `4`-look (`61, 122, 183, 243` worlds/mix) sequential strict gate (`src/evaluation/sequential_gate.py`) over `200k` null / `50k` alternative Monte Carlo reps, `300k` skewness reps, and `1,250` exact cross-check reps (`sequential_gate_validation_20261002`).
- Validated paired survival noninferiority (`paired_ni_at_stop`, `M = 0.05`) vs the old `16`-world reference band (`paired_band_validation_20261003`), then calibrated pooled continuation band `v2` (`pooled_ni_continue`, `rci_obf`, `survival_band_v2_20261006`) and opt-in `sequential_phase_r` (`10,000` reps).

### What happened
- **Pairing & sample-size calibration:** Divergent trajectories drove per-world mass correlation near zero (**`r = -0.22` to `+0.11`**: `-0.01` frozen, `+0.11` mixed, `-0.22` scripted; death timing `R^2 = 0.85–0.90`), and a `16`-world draw inflated `10%`-MDE scripted `N` from `3,047` to `8,676` (`212/141/762` at `20%`). OBF futility held false stops at `0.096 <= 0.10` (vs `~27%` naive).
- **`4`-look OBF strict gate (`ADOPTED`):** Matched `sequential_gate.py` on `1,250/1,250` reps (`0` mismatches), held global-null `P(PASS) = 0.00058` and FWER `0.0492`, and cut expected worlds (vs `729` fixed) by **`27–75%`** (`532` = `-27.0%` at `delta = 0`, `369` = `-49.4%` at `+60`, `183` = `-74.8%` at `+130`; `597` at `+30`, `0.9434` power), adding a skew guard for `-2.83` skewness.
- **Paired survival bands (`v2` `RATIFIED`):** Un-paired `16`-world bands falsely rejected non-regressing models **`77–91%`** (`9.2–23.1%` pass) while letting `-0.05` drops pass `0.07–0.32` (`6.6–32.2%`); paired NI cut false rejects to `~7–9%` (`91.4%–94.5%` pass) and `-0.05` error to `0.020–0.053`, with `v2` (`pooled_ni_continue`, `0.075` floor) adding `CONTINUE_BANDS`.

### Why
Multi-snake chaos destroys per-world mass correlation once two policies take a single different move (`r = -0.22` to `+0.11`), requiring near-unpaired `N`, whereas pairing shares exact world seeds for deterministic incumbent caching and eliminates `16`-world reference-draw noise (`SD ~0.06` vs `0.02` margin).

### What to reuse / what not to repeat
- Reuse `src/evaluation/sequential_gate.py` and `research/sequential_strict_template/` (`/v2-pooled` with `rci_obf` and pre-final skew check; `sequential_phase_r` is opt-in) for confirmation gates.
- Never size gates assuming positive paired correlation between divergent policies, and never repeat unadjusted interim futility checks (`~27%` false stops) or un-paired `16`-world survival bands (`77–91%` false rejects).

### Full record
Original: `git show 90b3db94:docs/research/history/sequential-gate.md` | Primary sources: [gate_calibration_20260926/README.md](../../../research/gate_calibration_20260926/README.md), [paired_band_validation_20261003/README.md](../../../research/paired_band_validation_20261003/README.md), [sequential_strict_template/README.md](../../../research/sequential_strict_template/README.md)

<a id="bl-density"></a>
## Big-Lever DENSITY-1 and DENSITY-2 Synthetic Arena Shrink vs Natural Doom-Root Yield (`bl-density`)

**Dates:** 2026-10-21 to 2026-10-22
**Verdict:** Closed as `TERMINAL / REJECTED`: across `492` DENSITY-2 episodes, all nine smaller-map and extra-opponent cells were rejected under the minimum-map realism rule (identity `12/12`), closing synthetic arena densification.

### What was tried
- Tested whether synthetic densification raises length `L >= 800` H300 doom-root yield over natural arm `A` (`145 × 83` map, five opponents): `B70m` (`121x69` map, five opponents), `B70p` (`121x69` map, seven opponents; `B50` dropped after Round 14), and `C-P1` (injected long serpentine bodies on `145 × 83`).
- Coarsened root sampling stride from `50` to `200` (`24` roots/episode) after Mac preflight measured `2.184720` CPU-s per `L800` H300 branch, and enforced Round-14 kill rules (`REJECTED` if yield `< 2*A`, avoidable fraction `< 0.5*A`, 35-feature logistic classifier AUC `>= 0.80`, or a minimum-map violation `A_min(L) = max(8*L_hero, U_total/rho_95, 8349)`).
- Ran the `90`-episode `DENSITY-1` pilot (`36` `A`, `18` `B70m`, `18` `B70p`, `18` `C-P1`) and scaled to `492`-episode `DENSITY-2` replication (`240` `A`, `72` `B70m`, `108` `B70p`, `72` `C-P1`) on ten single-thread L4 CPU workers with outcome-blind resume completion.

### What happened
- **`DENSITY-1` 90-episode pilot (`4,532` CPU-s, `INSUFFICIENT` natural denominator):** Natural `Arm A` yielded only `5/36` doom-bearing episodes (`1/12` frozen, `2/12` mixed, `2/12` scripted) and `0, 1, 1` (`2` total) witnessed-avoidable roots, vs `5/18` (`B70m`), `3/18` (`B70p`), and `11/18` (`C-P1`) doom-bearing episodes (`27/29` pilot doom roots were `DEATH_SELF`, median length `1186`).
- **`DENSITY-2` production & repins:** Initial L4 run verified `408/492` episodes (`4.90 u` of `16.50 u` series cap) before hitting a post-work telemetry deadline, leaving `84` primary + `12` Arm-A calibration units prepared across transport repins `02`–`04` (fixing the L4 family-canary gate to require CUDA observations only when CUDA is declared).
- **Terminal verdict (`REJECTED`):** All nine smaller-map/more-opponent cells across `DENSITY-2` violated the minimum-map rule and were `REJECTED` (local sparse-archive identity `12/12`), closing the density lever.

### Why
Shrinking the arena (`121x69`) or adding opponents (`7` vs `5`) increased raw doom-root encounters in pilot cells, but every densified cell violated the minimum-map area constraint `A_min(L) = max(8*L_hero, U_total/rho_95, 8349)` required to preserve natural `145 × 83` late-body coiling geometry.

### What to reuse / what not to repeat
- Exempt CPU-only L4 multi-worker manifests from requiring CUDA action observations in family-canary checks unless the job declares CUDA.
- Measure single-branch H300 CPU cost (`2.184720` CPU-s at `L800`) before sizing dense-stride root sweeps (`2,412,000` decisions exceeded a `90`-minute L4 session until stride moved `50 -> 200`).
- Do not shrink map dimensions or crowd extra snakes to harvest `L >= 800` trap states faster; both violate late-body geometry realism.

### Full record
Original: `git show 90b3db94:docs/research/history/bl-density.md` | Primary sources: [bl_density/README.md](../../../research/bl_density/README.md), [METRICS_AMENDMENT_04.md](../../../research/bl_density/METRICS_AMENDMENT_04.md), [COMPLETION_REPIN_04.md](../../../research/bl_density/COMPLETION_REPIN_04.md)

<a id="r32-r33-ops"></a>
## Round 32 End-to-End Rehearsal and Round 33 Operations (`r32-r33-ops`)

**Dates:** 2026-10-23 to 2026-10-24
**Verdict:** `R32` synthetic consumer trace, ten-label R9 oracle, and 15-path lifecycle matrix passed (`PASS` for synthetic subcontracts, `RETAIN_FIXED_DESIGN` for statistics, while strict validation, registry admission, and release readiness stayed `BLOCKED` and main held for a same-pin canary); `R33` operations verified (`64` passed) and canceled its CPU pilot on zero demand.

### What was tried
- Built `research/r32_rehearsal`: `consumer-02` exported-byte trace (factory payload → label ledger → three-update fit → fused exports → fresh-process serving selectors), `ORACLE.json` (hand-calculated cases for all ten ordered R9 labels plus boundary ties), and `matrix.py` (`3` positive resume/stop paths and `12` fault injections).
- Built `STATISTICS.md` / `statistics.py` (`R32-5`): prospective comparison of the equal `128`-world/mix design against cost-aware fixed allocation and a single futility look at `64`/mix (`28` units, shared `288 min / 0.384 u` ceiling).
- Delivered `research/r33_ops` and `research/r33_verification`: `tools/ops/task_journal.py` (`100` deterministic SQLite crash/restart schedules), `package_spec.py`, `proof_dependencies.py`, and a `15`-node 3-shard Linux contract suite.

### What happened
- **`R32` synthetic subcontracts (`PASS` synthetic / `BLOCKED` consumer readiness):** Combined light verification passed (`24` passed, `76` deselected in `25.35 s`); `consumer-02`, the ten-label `ORACLE.json`, and all `3 + 12` lifecycle paths passed while production `F4` admission stayed `BLOCKED` pending an `ego2s`/T34 strict-request and registry adapter.
- **`R32` canary fix (`c1ab2750`):** First Colab CPU canary failed `INCOMPLETE` (`136.26 s` / `0.003028 u`) on `FileExistsError` because the supervisor pre-creates the empty `output` directory.
- **`R32` statistics fix (`RETAIN_FIXED_DESIGN`, main `HOLD`):** Passing `heavy_tail_df` (`5`) to `Generator.standard_t` (`4f11670a`) passed `28/28` local units (`9.10 s`), holding Colab main for a same-pin `4f11670a` canary (`16.84 min / 0.02245 u` prior series debit).
- **`R33` operations (`Implementation verified; CPU pilot canceled`):** Passed `64` tests (`1.06 s`), `15` Mac preflight checks (`3` shards, `3.58 s`), `15` hosted Linux CI nodes (`40057728587`), and `100` crash/restart schedules; handoff-time SLA targets stayed `INSUFFICIENT_LT10` and `A1-PROF` stayed `OWNER_HANDOFF_UNACKNOWLEDGED / NATIVE_PENDING`.

### Why
The Colab CLI supervisor pre-creates the attempt `output` directory before launching the remote worker (breaking guards that demand a non-existent path), `R32-5` kept `RETAIN_FIXED_DESIGN` because published data could not calibrate joint variance or costs, and `c1ab2750` canary proof could not admit `4f11670a` under the same-pin rule.

### What to reuse / what not to repeat
- Reuse `tools/ops/task_journal.py` and the `R32` fault-injection matrix (`12` sidecar, truncation, namespace, and snapshot faults) to rehearse consumer lifecycles offline before real data exist.
- Make output-directory guards accept an existing empty directory while still refusing files, symlinks, or non-empty trees, and never treat a canary from an earlier commit pin as valid admission for a modified commit.

### Full record
Original: `git show 90b3db94:docs/research/history/r32-r33-ops.md` | Primary sources: [r32_rehearsal/README.md](../../../research/r32_rehearsal/README.md), [STATISTICS.md](../../../research/r32_rehearsal/STATISTICS.md), [r33_ops/HANDOFF.md](../../../research/r33_ops/HANDOFF.md)
