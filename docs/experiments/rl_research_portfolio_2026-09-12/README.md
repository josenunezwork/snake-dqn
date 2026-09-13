# RL research portfolio — 2026-09-12

This directory is the durable index for the current RL research portfolio. It
answers a simple maintenance question: what was tried, what actually happened,
and what is still only a proposal or a prepared run.

Read [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) before treating an experiment
name, checkpoint, or research proposal as evidence. The ledger distinguishes
engineering verification, completed scientific executions, retained harness
failures, and unrun work.

## Current decision boundary

The `vector61` Apex checkpoint remains the incumbent. Raster/PQN observations
(`raster31v2` and experimental `raster31v3`) are candidate paths. Only
`src/scripts/tournament_eval.py` can promote a candidate; none of the entries
in this portfolio does so.

The phase witness led to an opt-in PQN decision-phase repair. It is implemented
on clean source `b760b40`, where legacy `pre_transition_v1` remains the default
and `watch_pre_move_v1` is restricted to corrected-v3 `raster31v3` runs. The
repair's focused contract checks and the non-slow suite passed; hardware smoke
receipts also passed on CPU and MPS. These checks qualify the implementation
boundary; they do not establish policy quality or promotion eligibility.

This bounded wave is complete. Its aggregate scientific-child-compute budget
was 600 seconds. The retained VC1 run consumed 28.425 seconds; engineering tests
and short hardware qualifications are recorded separately. One MPS learner may run at a time with two native CPU threads, a 4 GiB
RSS cap, an 8 GiB MPS-driver cap, and a 6 GiB available-memory reserve. These
are resource bounds, not a promise that a pending run will complete.

## Execution and outcome vocabulary

| Status | Meaning |
| --- | --- |
| `COMPLETED` | The frozen execution has an accepted terminal record and result. Pair it with a positive, negative, or inconclusive outcome. |
| `ENGINEERING_ACCEPTED` | Contract, test, or compatibility evidence passed; this is not a learning result. |
| `INCONCLUSIVE_STOP` | A protocol stopped and does not support its planned comparison. |
| `INCONCLUSIVE_NOT_ADVANCED` | A completed probe failed its predeclared advancement rule. |
| `PREPARED_NOT_EXECUTED` | Inputs or an executable draft exist, but no scientific numerical run has occurred. |
| `UNRUN_DEFERRED` | A prepared numerical run was intentionally not launched because a prerequisite decision contract needs repair and qualification. |
| `RESOURCE_CAP` | A run reached its frozen time or resource boundary; partial metrics remain retained but cannot satisfy a complete-run gate. |
| `PROPOSAL` | A research recommendation that has not been frozen or run. |

Outcome classes are `POSITIVE`, `NEGATIVE`, `INCONCLUSIVE`,
`HARNESS_FAILURE`, `UNRUN`, or `ENGINEERING_ONLY`. A positive diagnostic still
does not promote a model. A harness failure does not become a negative learning
result. This two-part vocabulary is intentional: it prevents an operational
stop from being mistaken for an RL comparison.

## Updating the record

The owner of a new execution should update its row in the ledger only after its
terminal receipt, source/config closure, resource receipt, and analysis are
written. Use this row template: **question; method; source and configuration;
seed identity; budget and observed resource use; terminal result; outcome class
and interpretation; next action; artifact links and hashes.** Record an
interrupted or failed attempt as its own row; do not replace it with a later
run. Preserve absolute artifact paths and SHA-256 values when they are
available. State whether the result changes promotion status (normally it will
not).

The portfolio campaign root is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/research-portfolio-20260912`.
Its `campaign.json` is the current resource-policy source. Treat terminal and
supervisor receipts as execution authority; the campaign's `scientific_runs`
list is planning metadata and can lag a completed artifact write.

The [research synthesis and parallel waves](RESEARCH_AND_WAVES.md) assigns
implementation ownership and prerequisites for the next studies. The
[decision-phase execution closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/decision-phase-20260912/root-execution-closure-v1.json>)
links the tested source, both hardware runs, and independent audit.

## Related records

- [Project history and durable findings](../../project_history_and_findings.md)
  is the concise promotion-status record.
- [PQN sampler results](../../pqn_sampler_results_2026-09-06.md) records the
  completed five-seed sampler diagnostic.
- [PQN follow-up execution report](../pqn_followup_2026-09-12.md) is the B5
  implementation and terminal-run record.
- [H1-A observation-value report](../observation_value_2026-09-12/REPORT.md)
  records the completed remote-heading decision-relevance probe.
- [Decision-phase design note](../../pqn_decision_phase_2026-09-12.md) explains
  the opt-in observation timing contract and its compatibility boundary.
