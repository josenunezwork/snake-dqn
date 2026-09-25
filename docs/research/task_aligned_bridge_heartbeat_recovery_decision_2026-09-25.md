# Decision: one remaining bridge qualification attempt

Status: **REVIEWED PLAN READY; AWAITING DIRECT USER DECISION; NOT ADMITTED**.
No new tests, model loads, games,
training, or scientific audit have run. The previous three attempts remain
closed, including the interrupted shakedown. This proposal requires a new direct
decision because the previous approval covered one attempt with no further retry.

## Recommendation

Repair the heartbeat contract, retain the 79 passed synthetic checks and captured
files, and run only seven new heartbeat cases plus the unfinished live interface
qualification and audit. Keep the original nominated 6102 checkpoint. A successful
qualification would permit preparing its separate comparison against Apex; it
would not establish better gameplay, promote the checkpoint, or deploy it.

The failure was a tooling defect: the stage wrote a wall-clock `time` field while
the supervisor expected `monotonic`. The replacement helper atomically publishes
a monotonic record and checks it with the actual supervisor reader before model
or game work starts. It keeps updating every five seconds, and a writer failure
prevents the stage from reporting success. The initial check establishes local
parser compatibility; it does not claim the parent supervisor has acknowledged
the file. The existing watchdog limits remain unchanged.

The helper and its seven prospective tests passed independent **source review**.
The tests have not run. They isolate the actual reader and monitor loop with fake
clocks and processes, without importing the Torch-dependent monitor module.

## Explicit change to world coverage

The failed first shakedown cell has no resumable world/RNG state. Its partial work
will not repeat. Retire the old shakedown pair at indices 0 and 1.

Use the already materialized, previously unplayed indices **2 and 3** for the new
12-episode paired shakedown. These indices were already assigned to Watch and
Play, respectively; they are **not spare worlds**. Preserve those Watch/Play
assignments. This deliberately shares two seed IDs across the three interface
checks and reduces unique-world coverage from the original design. It is limited
interface qualification, not independent performance or generalization evidence.
No new IDs, search, resampling, or historical-source rereads are proposed. The
separate public challenge's development, pilot, final and serving namespaces stay
unchanged.

No gameplay result was used to choose this pair. Its unexecuted status must remain
supported by the closed receipts; no all-history or geometric novelty is claimed.

## Finite scope

| New stage | Maximum seconds | Scope |
|---|---:|---|
| Heartbeat boundary | 60 | Seven new cases and bounded metadata carry-forward |
| Shakedown | 3,600 | Candidate/incumbent × three mixes × two shared worlds |
| Watch | 600 | Original one-episode serving check |
| Play | 600 | Original one-episode serving check, including initial Watch construction |
| Independent audit | 900 | Saved evidence and source only |
| **Stages** | **5,760** | No automatic retry |
| Separate administrative handoff reserve | 120 | Closeout and durable handoff |

The prospective window is at most 7,200 seconds and starts only after explicit
authorization and review of the exact source and plan. Each stage must fit all
remaining caps, audit time and the separate handoff reserve, including after lock
acquisition. Old deadlines cannot be reused.

The seven new cases cover legacy-schema rejection, atomic publication, startup
failure, startup ordering, repeated updates and staleness, invalid timestamps,
and background-writer failure. Their successful-path source accounting is eleven
tiny heartbeat writes, twenty-one heartbeat reads, and nine monitor-source reads
bounded to 2 MiB plus one overflow-probe byte each. The planned test invocation
disables external pytest plugin loading, clears inherited pytest argument/plugin
variables, and bypasses project conftest discovery. No old case is included. The
cases simulate process actions and use no model, checkpoint or game. One isolated
pytest worker has a 45-second ceiling inside the same 60-second stage and process
group, with combined parent/descendant RSS capped at 8 GiB. It is harness overhead,
not another controller.

Live work stays within **14 H5000 episodes, 70,000 world frames, 370,000 forward
calls and 720,000 forwarded rows**, with zero optimizer updates. Preserve every
existing per-stage construction, checkpoint I/O, parity and lifecycle cap. CPU
remains two intraop threads and one interop thread; RSS is capped at 8 GiB, MPS at
24 GiB, and available RAM must remain at least 9.6 GiB. Numerical jobs use the
existing shared locks and Mac supervisor, with one controller.

## Preserved evidence and limits

Reuse the successful metadata handoff and the four boundary plus 75 fixture
passes through their original receipts. Do not rerun capture, namespace
materialization, the historical-source extraction, that handoff, or any of those
79 cases. Bounded administrative authentication is a new carry-forward join,
not a relabeled completion of the failed attempt. Original capture and fixture
executable provenance remains distinct from the new source closure. Later live
checkpoint authentication and loading remains charged to its new stages.

The carry-forward inventory contains exactly 24 named administrative records,
1,288,737 expected bytes in total, each at most 1 MiB. Its hard read ceiling is
1,288,761 bytes, including one extra growth-detection byte per file. Source review
has inspected these administrative shapes; there is no analyst-blindness claim.
No old raw journal is part of this inventory.

The old failed cell remains in exposure and cost accounting, with persisted lower
bounds of 1,967 frame attempts, 1,966 completed frames and 11,802 forward calls.
Its unflushed tail and full elapsed time through termination are unknown. Preserve
those unknowns and the old physical ceilings; new complete cells cannot erase
them. No failed attempt is changed to PASS, and administrative closure does not
replace the missing scientific audit.

The executable carry-forward/controller/audit integration is complete and passed
independent source review. The exact unadmitted plan also passed independent
review. Static AST and whitespace checks passed; runtime checks have not run.
No admission or new deadline exists. Any new failure would close the proposed
attempt with zero automatic recovery.

## Reviewed implementation and plan

The isolated source commit is `344a7974544669194889caca93f8ef8c4590a664`, in
`/Users/josenunez/.codex/worktrees/task-aligned-heartbeat-repair/snake-dqn`.
Its checkout is clean. Main's numerical source and the three closed attempts
were not changed. No push occurred.

The preparation directory is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-bridge-qualification-20260924/preparation-v4`.

- Exact plan: `3171149a4db0b4402c13240b861e76dc1f4073ebe76da96b8b82193a2cfaac62`.
- Exact plan review: `17f6fd19d0369caea632d5f6ce5ada77733cecc2eeef900acff47007d9337726`.
- Implementation source review: `1546232d28db04b25b619facad91822db1d2e670cc851f47c93cb8e443b6cc5d`.
- Audit source review: `b723b6977d609d069e72803c90d2dea5b4ef9fe58b608f413b34a184da934f08`.
- Metadata declaration: `7d018a158cb7a0c7fab4e069c914a4a15f75d368fc6f51cfed52b39e68aa9d9d`.

The plan binds 286 source/configuration files. It has `admitted: false`,
`authorization: null`, no deadline, and an absent `admission-v4/run` output root.
Preparation will require a new actual approval record tied to this plan's SHA,
matching unused campaign authorization, and another active-job/resource check.
The reviewed plan is not permission to execute it.

## Related records

- [Closed saved-prefix recovery](../experiments/task_aligned_bridge_saved_prefix_recovery_2026-09-25/README.md)
- [Previous approved recovery decision](task_aligned_bridge_saved_prefix_decision_2026-09-24.md)

The alternative is to retain the repaired source and stop this qualification
branch. Apex remains the operational incumbent; only the shared strict tournament
gate can authorize promotion.
