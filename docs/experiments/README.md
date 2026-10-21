# docs/experiments — map

**Historical.** This folder records the September 6 – October 3, 2026 research
era: PQN/raster probes, the solo-food and controlled-encounter lines, the
task-aligned challenger and the Apex safety-veto releases. None of it is the
current campaign. The active work is T34 (search teacher + DAgger into the B8
CNN), documented in [`../research/final_run_20261019/`](../research/final_run_20261019/)
(start with its `HANDOFF_2026-10-20.md` and `CAMPAIGN_LOG.md`), with authority
in [`../STATUS.md`](../STATUS.md). Read the studies here as evidence about past
decisions, not as instructions.

## How the folder is organized

- **One directory per study**, named `<series>_<topic>_<YYYY-MM-DD>` (the date
  is the study date). Each normally has a `README.md` closeout with the
  pre-registered rule, the result and the decision, plus figures (`*.png`) and
  sometimes JSON receipts or manifests. Raw runs, checkpoints and full receipts
  live outside the repo under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`;
  READMEs cite them by path and SHA-256.
- **Five loose `.md` files** are cross-study syntheses or execution reports
  (listed below).
- One undated directory, [`pqn_per_env_autoreset_v1/`](pqn_per_env_autoreset_v1/),
  holds only a frozen opt-in contract.
- Each study fixes its own parents, worlds, horizon and decision rule. A pass in
  one study does not pool with or transfer to another.

At the time of writing: 204 entries (199 study directories + 5 `.md` files).

## Series

Per-series catalogs (written in parallel; each lists every study in its series):

| Catalog | Covers |
|---|---|
| [SOLO_FOOD_SERIES.md](SOLO_FOOD_SERIES.md) | `solo_food_*`, `solo_native_*`, other `solo_*` |
| [TASK_ALIGNED_AND_APEX_SERIES.md](TASK_ALIGNED_AND_APEX_SERIES.md) | `task_aligned_*`, `apex_*` |
| [OTHER_SERIES.md](OTHER_SERIES.md) | `controlled_encounter_*`, `pqn_*`, and the one-off studies |
| [state_system_audit_2026-09-06/README.md](state_system_audit_2026-09-06/README.md) | the system audit evidence bundle |

### State/system audit (2026-09-06, 1 directory)

[`state_system_audit_2026-09-06/`](state_system_audit_2026-09-06/README.md) is
the evidence bundle for the expanded state-and-system review. It pins 100 audited
source/config files by hash and retains observations by area (`vector`, `raster`,
`pqn`, `apex`, `dynamics`, `serving`, `evaluation`, ...). It ran no training and
changed no product source. It sits beside the same-day
[`pqn_sampler_2026-09-06/`](pqn_sampler_2026-09-06/) campaign.

### Early RL repair and PQN probes (2026-09-06 to 09-13, plus 09-21)

These are the `pqn_*` directories, the
[RL research portfolio](rl_research_portfolio_2026-09-12/README.md) (its
`EXPERIMENT_LEDGER.md` separates completed runs from prepared or unrun work),
the [four-hour work window](four_hour_2026-09-13/README.md), and single studies
of 09-12/13 (`observation_value`, `learning_rate`, `exploration_floor`,
`direction_coverage`, `mass_objective`, `canonical_anchor`, `common_world_retention`,
`compute_probe`, `scripted_advice`). The four-hour window concluded: keep Apex as
incumbent and stop extending the tested PQN recipe unchanged (150k arms fell
below their 50k checkpoints; policies collapsed into right-turn loops).
[`pqn_mac_capacity_2026-09-21/`](pqn_mac_capacity_2026-09-21/README.md) is a
later capacity study. Catalogued in [OTHER_SERIES.md](OTHER_SERIES.md).

### Solo food (2026-09-13 to 09-21, ~100 directories)

The largest series: `solo_food_*` (78), `solo_native_*` (14) and eight other
`solo_*` studies. It asks whether a raster/PQN learner can collect food and
survive alone, then with one opponent. It moved from an exploration-floor screen
that failed its all-seed rule, through supervised vs native-PQN contrasts, to
native PQN warm-starts that passed solo H512/H768 food-and-survival screens on
three fixed seeds. It failed paired retention with one GreedyFood opponent
(`solo_food_pqn_one_opponent_2026-09-21`). Read the two syntheses first:
[solo_learning_status_2026-09-20.md](solo_learning_status_2026-09-20.md) and
[solo_food_learning_progress_2026-09-21.md](solo_food_learning_progress_2026-09-21.md).
Catalogued in [SOLO_FOOD_SERIES.md](SOLO_FOOD_SERIES.md).

### Controlled encounter (2026-09-21 to 09-22, 42 directories)

A scripted, finite encounter bank (frontal/cross threat vs parallel/receding
benign families) for measuring opponent handling, followed by teacher,
imitation, native-TD and label/coverage diagnostics on that bank. The
[calibration](controlled_encounter_calibration_2026-09-21/README.md) admitted
576 cases with zero model inference. The
[native full-cycle continuation](controlled_encounter_native_full_cycle_2026-09-22/README.md)
failed every final competence and retention conjunction, ending the
unchanged-native-dose line. Catalogued in [OTHER_SERIES.md](OTHER_SERIES.md).

### Task-aligned challenger (2026-09-24 to 09-26, 9 directories)

The `task_aligned_*` studies continued the PQN lineage 2026096102 as a challenger
to Apex: CZ integration and continuation, bridge qualification/replacement and
recovery, and a strict pilot. The
[strict pilot](task_aligned_strict_pilot_2026-09-26/README.md) ended
`INVALID_STOP_AUDIT_FAILED`, and the line was **retired** under its
pre-registered kill criterion; Apex/vector61 stayed incumbent. Catalogued in
[TASK_ALIGNED_AND_APEX_SERIES.md](TASK_ALIGNED_AND_APEX_SERIES.md).

### Apex (2026-09-22 to 10-03, 32 directories)

Two parts. First (09-22 to 09-24), studies that applied the research lessons to
the incumbent feedforward Apex network: growth/second-food censuses, replay and
exploration continuations, teacher/imitation, horizon and width scaling (e.g.
[research transfer](apex_research_transfer_2026-09-22/README.md)). Second
(09-27 to 10-03), the decision-time safety veto: the screen and strict gate, web
serving qualification, and versions v3, v5, v6, v7 and v8
([v8](apex_veto_v8_2026-10-03/README.md) is the last). Catalogued in
[TASK_ALIGNED_AND_APEX_SERIES.md](TASK_ALIGNED_AND_APEX_SERIES.md).

## Loose syntheses and reports

| File | What it is |
|---|---|
| [rl_repair_execution_2026-09-08.md](rl_repair_execution_2026-09-08.md) | RL repair packages and G0 qualification; X0 screen inconclusive |
| [pqn_followup_2026-09-12.md](pqn_followup_2026-09-12.md) | Per-env PQN autoreset; B5 ended `INCONCLUSIVE_STOP` |
| [solo_learning_status_2026-09-20.md](solo_learning_status_2026-09-20.md) | Solo-learning evidence map (updated through the 09-21 studies) |
| [solo_food_learning_progress_2026-09-21.md](solo_food_learning_progress_2026-09-21.md) | Readable synthesis of the solo-food line |
| [research_stopping_point_2026-09-21.md](research_stopping_point_2026-09-21.md) | Stopping point; it was resumed later the same day |

## What came after

From October onward, results are recorded elsewhere, including the FRP-v2..v5
and M3 lines, the B8 release, and T34. See [`../STATUS.md`](../STATUS.md)
and [`../research/`](../research/).
