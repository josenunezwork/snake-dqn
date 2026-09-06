# Project history and durable findings

This concise evidence record separates dated observations from current promotion
status. For the September operational cleanup, see
[Codebase quality handoff](codebase_quality_2026-09.md).

## Current status

Apex with `vector61` is the incumbent. Raster/PQN uses `raster31v2` and is an
unpromoted candidate path. [`tournament_eval.py`](../src/scripts/tournament_eval.py)
is the promotion authority: a candidate needs a positive paired mass-integral
confidence interval on at least two opponent mixes and no regression against the
scripted anchor. The [redesign blueprint](ml_redesign_blueprint_2026-07.md) is
the detailed design; the [SIMD specification](simd_env_spec.md) is the executable
dynamics contract.

## Findings that remain relevant

| Date and source | Observation | What it means now |
| --- | --- | --- |
| 2026-06-21, `logs/OVERNIGHT_REPORT.md` | Free-space candidate A5 reported 139.53 versus 41.37 champion mass. Its label named food 1.5, while checkpoint metadata and training evidence record food 3.0. | This is a dated result on the retired alive-conditioned, narrow-opponent gate. The label is not reliable reward provenance. Free-space addressed self-trapping; it is not current promotion proof. |
| 2026-06, deleted DRQN reports and blueprint appendix | GRU/DRQN was trained and did not beat the feedforward model in old paired frozen-opponent evaluations. | Evidence against that recurrent setup, not a repaired-gate result. It does not settle future recurrence experiments. |
| Git archaeology in blueprint Appendix B | CNN code existed briefly but was never trained and produced no checkpoints. | Never say CNN variants lost a benchmark. The spatial/convolutional question remains open and raster/PQN tests it. |
| 2026-07-05, `logs/P0_REPORT.md` | The gate changed to total-horizon mass integral, paired deltas, opponent mixes, and scripted anchors. A5 cleared scripted but not mixed; a 3% live-engine MDE required about 6,450–9,637 seeds. | Old wins are not comparable to current results. Use the repaired gate and report its uncertainty. |
| 2026-07-05, `logs/P1_NOTES.md` | Mechanics and reward v2 are opt-in; v1 was retained. The planned three-arm kill diagnosis has no recorded verdict. | Do not report the diagnosis as completed or infer its cause from the plan. |
| 2026-07-06, `logs/P2_NOTES.md` | Initial parity covered 600,000 frames for rectangular, v2, train-mode terminal-death runs. | A strong result with that dated scope, not a claim that every branch was proven then. |
| 2026-07-06, `logs/P5_NOTES.md` | The optimized simulator reported 109,114 agent-steps/s at 256×8; raster serving and checkpoint SIMD gating worked. A 2.4M-step local raster run lost to `scripted:random_safe`. | The loop is demonstrated, but the local raster policy is not a champion. Keep Apex until a candidate passes the current gate. |

## Historical source map

Raw ignored reports, JSON results, checkpoints, configs, and logs remain the
provenance for these observations. The tracked narratives below were removed
because their plans, paths, or metrics are no longer current. Retrieve exact
contents from the pre-consolidation tree when necessary:

| Removed item | Git provenance | Replacement |
| --- | --- | --- |
| `archive/ARCHITECTURE.md` | `git show bf4d147:archive/ARCHITECTURE.md` | Current source, README, and the algorithm document |
| Dated roadmap and run narratives | Exact paths listed below | This record, blueprint, and raw dated reports |
| `colab/h100_apex_v1.py` | `git show bf4d147:colab/h100_apex_v1.py` | Blueprint grid-fork caution and current SIMD specification |

The removed Colab fork used a grid, cell-equality environment and diverged from
the production contract. It is not a continuous-dynamics precedent. Its history
does not establish anything about CNN training; the CNN finding has separate
checkpoint and history provenance.

The five deleted history sources are reproducible from the same tree:

```bash
git show bf4d147:docs/history/autonomous_run_20260615.md
git show bf4d147:docs/history/cleanup_plan_2026-04.md
git show bf4d147:docs/history/gpu_drqn_log_20260620.md
git show bf4d147:docs/history/research_roadmap_2026-02.md
git show bf4d147:docs/history/slither_training_log_20260615.md
```

## September cleanup decisions

- Removed the write-only, unbounded `MetricsTracker` pipeline. Bounded learner
  statistics and TensorBoard remain live.
- Removed no-caller factories, mixins, wrappers, and key lists without changing
  checkpoint envelopes.
- Removed the old Colab fork, archive, and planning narratives after preserving
  the findings and source provenance above.
- Removed 57.6 MiB of generated caches and builds with scoped `make clean`.
  Real models, replay data, raw logs, and registered worktrees were retained.

## Boundaries for future claims

- Call numeric experiment results *dated observations* unless a current,
  reproducible gate run accompanies them.
- Keep blueprint proposals, implementation milestones, and champion-gated
  promotions distinct.
- Do not revive alive-frame mass, one-opponent rankings, or self-play reward as
  a promotion decision.
- Apex, offline tooling, and curriculum code remain live while raster is
  unpromoted. Blueprint deletion milestones are conditional plans, not completed work.
