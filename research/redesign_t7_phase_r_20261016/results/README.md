# T7 published evidence

The scientific run and isolated audit used clean commit
`7e1734357115d44b11038634a0ce1cb18afddfa3`. The later publication commit also
contains a coordinator-only lifecycle repair; it is not the run identity.

`decision.json` contains the sole full-N decision, all 45 prefix-control results,
and the path/SHA256 inventory for all 5,805 raw records. `audit.json` is the
separate frozen-source audit. `independent_result_review.json` records the
independent post-run review. `plan.json` and `start.json` retain the complete
registration, source, checkpoint, profile, roster and world identities.

Raw records remain at:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t7-phase-r-b8-20261016/records/`.
Paths inside the decision inventory are relative to that artifact root, not this
publication directory. Reproduction of the original audit requires the frozen
run commit, pinned local checkpoints and original artifact tree. The runner
intentionally rejects a different source/commit identity.

The first coordinator's log/exit and `operational_recovery_1.json` explain the
observed process-group SIGCONT permission error. `coordinator_resume.py` is the
reviewed coordinator adapter used to resume the same run after the old workers
finished. The scientific workers and rules stayed frozen. `ownership*.json`,
launch scripts and final run/audit logs retain task-owned execution evidence.

`postrun_applied_validation.json` records tests of the permanent lifecycle repair
applied only after completion and both result reviews. No gameplay was rerun.
`postrun_owned_signals.patch.gz` preserves the reviewed patch bytes without
changing the original patch hash retained in the validation receipt.
`publication_manifest.json` hashes the copies in this directory and the report.

The decision report is `docs/research/t7_phase_r_result_2026-10-16.md`.
Serving defaults were not changed; qualification/release is a separate step.
