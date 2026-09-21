# Research stopping point — September 21, 2026

**Historical stopping point; explicitly resumed later on September 21.** The user
subsequently authorized up to 85% of this Mac's RAM/compute and said “sounds great
get started.” The [capacity study](pqn_mac_capacity_2026-09-21/README.md), longer-dose
study, and controlled-encounter calibration have since completed. Earlier completed
study criteria and results remain unchanged. The original results below describe
the earlier stopping point; the resumption section records subsequent work.

Completed work is preserved in local commits, frozen checkpoints, raw gameplay,
training diagnostics, resource receipts, and audited closeouts. Apex is unchanged;
no tournament promotion or remote push occurred.

- The earlier solo exploration-floor campaign was reconciled from completed
  receipts. All seeds remain reported under their original criteria.
- The original food microbenchmark showed imitation succeeding in all three
  seeds while the original PQN recipe succeeded in one. Subsequent native PQN
  work established food collection and survival across all three fixed seeds in
  768-frame solo games. See the [progress synthesis](solo_food_learning_progress_2026-09-21.md).
- The latest [CZ comparison](solo_food_pqn_enemy_access_2026-09-21/README.md)
  completed six training runs and 28 gameplay evaluations. All final policies
  passed absolute food/survival checks. Enemy-aware learned progress passed only
  seed 2026096102; 6101 lost food performance and 6103 lacked a positive endpoint
  confidence bound. Blind learned progress passed no seeds. Both all-three
  conclusions remain false. Scientific jobs were not repeated.
- [DA saved-data diagnosis](solo_food_pqn_enemy_learning_diagnostic_2026-09-21/README.md)
  completed with zero new games or training. Few death transitions, substantial
  TD residual around death, and shifting food/boost behavior leave experience
  coverage and adaptation stability unresolved.

The qualified opt-in S2 lifecycle and its tests were integrated on main at
`fe3f89c` and `542c5da`. The three integrated files are byte-identical to the
qualified experiment checkout; no additional test rerun was needed for that
unchanged integration. CZ results/plots were committed at `575a748`.

## Historical resumption information

The current durable campaign ledger is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/campaign.json`.
At the stopping point, the newest artifact roots were `solo-food-pqn-enemy-access`
(CZ) and `solo-food-pqn-enemy-learning-diagnostic` (DA). Subsequent completed roots
are `pqn-capacity-20260921`, `solo-food-pqn-enemy-dose`, and
`controlled-encounter-calibration`, under that same directory.

Frozen training source: `runs/pqn-enemy-experience-source`, branch
`codex/pqn-enemy-experience`, commit `1f32d2de1987d97441fa74c1898ed9039b535f24`.
Frozen evaluation source: `/Users/josenunez/Projects/ml/snake-dqn-ambient-objective`,
commit `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

This is a historical stopping point. The later unchanged-recipe dose is now
completed and documented in the [dose record](solo_food_pqn_enemy_dose_2026-09-21/README.md):
all-three cumulative, retention, and incremental packages were false. The later
[controlled encounter calibration](controlled_encounter_calibration_2026-09-21/README.md)
also completed and supplies a finite task bank. The six-checkpoint
[encounter dose probe](controlled_encounter_dose_probe_2026-09-21/README.md) is
also complete. Subsequent [fixed-priority controls](controlled_encounter_static_controls_2026-09-21/README.md)
showed that unconditional turning can solve the encounter survival criterion.
A separate food-and-survival imitation study has therefore been frozen and its
pipeline qualified. Fresh teacher-data collection is underway; this historical
note records no scientific training result for that study.

The previous resource envelope used shared CPU locks and serial jobs, two CPU/BLAS threads,
4-GiB RSS cap, 12-GiB available-memory reserve, and 8-GiB MPS driver cap.
The resumed capacity benchmark retains shared locks and serial numerical jobs;
its prospectively declared envelope uses 2/4/8 CPU threads, a 16-GiB RSS cap,
24-GiB MPS-driver cap, and at least 15% system memory reserved (9.6 GiB on this
64-GiB Mac). RSS and MPS allocations overlap in unified memory. Actual system
availability is checked at admission and throughout every child job.
Keep Apex incumbent pending the shared tournament gate. Its larger historical
training dose does not demonstrate inherent algorithmic superiority.
