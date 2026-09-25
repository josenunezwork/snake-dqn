# Remaining bridge qualification, 2026-09-25

Status: **CLOSED_INVALID_STOP_AUDIT_FAILED_LIVE_RESULTS_PROVISIONAL**.

The approved recovery ran once. The seven new heartbeat cases, 12-cell shakedown,
Watch and Play stages all completed with controller-accepted PASS reports. The
independent audit then exited with a fixture-report validation error, so the bridge
is **not independently qualified**. Its completed live records are preserved as
provisional evidence. No stage or audit was retried, and no model was promoted.

The user directly authorized the reviewed recovery: “do the reviewed recovery
and stop asking for my input.” Admission v4 preserves the three closed attempts,
their original costs and failures, the captured checkpoint files, and the 79
already passed synthetic checks. It runs only seven new heartbeat cases, the
unfinished live interface qualification, and its independent saved-evidence audit.

The repair publishes the numeric monotonic heartbeat expected by the existing
supervisor, checks compatibility with that reader before model/game work, and
keeps publishing every five seconds. The startup check is local parser
compatibility, not an acknowledgement from the parent process. Watchdog limits
are unchanged.

## Admission and scope

The frozen source is isolated at
`/Users/josenunez/.codex/worktrees/task-aligned-heartbeat-repair/snake-dqn`, commit
`344a7974544669194889caca93f8ef8c4590a664`. The reviewed source is immutable during
execution. Main's numerical source is unchanged.

Evidence directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-bridge-qualification-20260924/admission-v4`.

| Binding | SHA-256 |
|---|---|
| Reviewed plan | `3171149a4db0b4402c13240b861e76dc1f4073ebe76da96b8b82193a2cfaac62` |
| Exact plan review | `17f6fd19d0369caea632d5f6ce5ada77733cecc2eeef900acff47007d9337726` |
| Direct authorization | `afa531499db6cc7643ebdafd373fc8a46227b649676942f7b78841a695eec592` |
| Admitted intent | `8bda18fb6e276063b958ca3bc4c439d49e28b3bf2523945c2efca2db1a16d66d` |
| Original nominated 6102 checkpoint | `8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f` |

The new 7,200-second window began at **2026-09-25 20:18:38 UTC**, with deadline
**22:18:38 UTC**. One controller started at 20:22:14 UTC, PID 44630. Stage caps are
60 seconds for heartbeat checks, 3,600 for shakedown, 600 each for Watch and Play,
and 900 for audit: 5,760 seconds total, plus a separate 120-second handoff reserve.
Every stage must fit all remaining caps and the handoff, including after lock
acquisition. No automatic recovery or shortened endpoint is allowed.

The live ceiling is 14 H5000 episodes, 70,000 world frames, 370,000 forward calls,
720,000 forwarded rows, and zero optimizer updates. CPU remains two intraop and
one interop thread, RSS at most 8 GiB, MPS at most 24 GiB, and available RAM at
least 9.6 GiB, enforced through the existing shared locks and Mac supervisor.

## Recorded outcome

| Stage | Outcome | Guarded seconds | New test cases | Episodes | Completed frames |
|---|---|---:|---:|---:|---:|
| Heartbeat / carry-forward | Natural exit 0; PASS | 0.409259 | 7 | 0 | 0 |
| Shakedown | Natural exit 0; PASS | 334.410171 | 0 | 12 | 60,000 |
| Watch | Natural exit 0; PASS | 52.816155 | 0 | 1 | 5,000 |
| Play | Natural exit 0; PASS | 1.030974 | 0 | 1 | 24 |
| Independent audit | Natural exit 1; INVALID_STOP | 0.207952 | — | — | — |

H5000 is a maximum horizon. The Play episode terminated after 24 frames; its
completed receipt is not a claim of 5,000-frame survival. The 14 live episodes
produced **65,024 frame attempts and completed frames**, 202,199 forward calls,
377,319 forwarded rows, 90 checkpoint loads, 64 model constructions (55 Apex and
nine raster), and 15 game constructions. The latter includes Play's initial
unstepped Watch construction. No checkpoint was saved and no optimizer update ran.

Other completed-stage counts are 20,314 hero transition attempts, 390,144 actor
transition attempts, 128 checkpoint file opens, 9,688 checkpoint read calls,
921,606,875 checkpoint bytes read/reserved, eight parity frames and 17 parity
actions. These totals come from the four completed controller reports; they are
not a replacement for the failed independent audit. The carried 79 cases and old
capture/failure costs are separate from these new counts.

All five children exited naturally, with total guarded time
**388.87451012514066 seconds**. Sampled peak child RSS was 533,446,656 bytes;
minimum sampled available RAM was 31,207,030,784 bytes. The isolated heartbeat
test worker separately recorded 57,245,696 bytes peak aggregate RSS and exited
normally in 0.2625054169911891 seconds. These are sampled measurements.

The controller exited 1 after sealing `run/failure.json` at
**2026-09-25 20:28:44.583503 UTC**. Its absence was then confirmed by process
inspection. The window is closed and no numerical job remains active.

## Audit failure and preserved uncertainty

The audit raised `ValueError: qualification audit: fixture repeated or missing
phase` in `validate_fixtures` (`audit.py:970`), called from the carried-record
validation. Its frozen rule requires exactly `[setup, call, teardown]` for each
of 75 collected fixture node IDs.

The saved fixture JSON has 75 distinct nodes, 75 setup reports, 75 teardown
reports and **82 passed call reports**. Two tests contribute the extra seven call
reports: `test_rejects_duplicate_and_cross_namespace_ids` contributes three and
`test_rejects_narrowing_and_noninteger_ids_before_reads` contributes four. The
corresponding frozen test bodies contain three and four `unittest.subTest`
iterations, matching those extra zero-duration passed reports. Source inspection
therefore identifies a plausible subtest reporting explanation; it does not
directly instrument the runtime hook. The saved multiplicity is not evidence of
seven repeated fixture executions.

This report-shape incompatibility escaped source review. The frozen audit still
failed; it cannot be relabeled PASS. It emitted a failure record but no audit
report, `audit/completed.json`, or root `completed.json`. Some preceding
authentication checks returned by control flow, but there is no separate durable
audit success seal or saved audit read-count log. Audit read counts are unknown.
No suffix audit, corrected audit, test or gameplay retry was run.

Root's one administrative receipt reconciliation authenticated the four completed
stage report/supervisor bindings and preserved the audit's natural exit 1. It read
no checkpoint payload or raw physical journal. This closure does not substitute
for the missing independent audit and does not change any prior attempt's status.

| Closure record | SHA-256 |
|---|---|
| Administrative failure attestation | `b71dd5b148edc3d060e55cb0db283b73a35c42df82c3f7d4e56be000205e4280` |
| Administrative closeout | `b25fe110b178ba3e850d498211816e8d1866b59c2b44c0359b2af7932fa9c226` |

## Evidence boundaries

The original nominated 6102/global35613 remains the candidate. The separately
continued child is not substituted. The fixed pool, deployment profile, restricted
raster inputs and Q scale are unchanged. These are serving and accounting checks;
they do not train the model or measure a new learning curve.

Exactly 24 named administrative records, 1,288,737 bytes, are carried forward.
The prior four boundary and 75 fixture passes retain their original executable
provenance; they are not rerun or counted as newly tested. Old checkpoint capture,
history extraction, namespaces and raw journals are not repeated. Live checkpoint
authentication/loading is charged to the new stages.

The shakedown uses already materialized indices 2 and 3. Watch and Play retain
those same assigned IDs, so the interfaces share two worlds. Old shakedown indices
0 and 1 are retired, and the interrupted cell is not replayed. This reduces unique
world coverage and supplies no independent performance or generalization evidence.
Future strict challenge namespaces remain unchanged.

The preceding interrupted shakedown remains invalid: its persisted lower bounds
are 1,967 frame attempts, 1,966 completed frames and 11,802 forward calls. Its
unflushed tail and full elapsed time through termination remain unknown. New
complete work cannot erase these costs or turn an old attempt into a pass.

The independent audit is designed to authenticate saved metadata and source, with
no checkpoint reads, model calls, gameplay or numerical reproduction. It failed
before issuing a PASS. The expected audit success receipt does not exist, so root
reconciled the failure rather than claiming successful qualification. Apex remains
incumbent; only the shared strict tournament gate can authorize promotion. Nothing
is published or pushed by this recovery.

## Related records

- [Reviewed recovery decision](../../research/task_aligned_bridge_heartbeat_recovery_decision_2026-09-25.md)
- [Closed saved-prefix recovery](../task_aligned_bridge_saved_prefix_recovery_2026-09-25/README.md)

This attempt and its zero-recovery window are closed. The successful stage records
and all prior failures remain intact. The active-source checkout was not changed
after admission. No unexecuted repair is represented as runtime evidence.
