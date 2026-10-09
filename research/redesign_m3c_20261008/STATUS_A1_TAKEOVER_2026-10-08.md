# M3-C A1 takeover and recovery status — 2026-10-08

This is a sibling status/deviation record. It does not modify or ratify the pinned
`PREREGISTRATION_M3C.md`, authorize additional spend, select an arm, or promote a policy.

## Verified run outcome

Codex took over from the user-provided handoff on October 8, Pacific time. Worktree
`m3c` was clean at `1614b335a5e16b8bbe8d2b80bb26fc0008653965`; unrelated untracked
work in the primary checkout was preserved.

A1 pod `tbnjau6oku1czb` launched at 20:53:27 PT. G2 passed at 14,842.53
transitions/s, 5.043 times the Mac rate, with projected total pod wall time
21,322 seconds. Six concurrent jobs ran (A/B/L, seeds 0 and 1); nine remained queued.
The runner recorded `stalled` and deleted the pod at 22:45:31 PT. Its cleanup receipt
records `pods_deleted_verified: true`. A fresh provider listing during takeover was
empty. The runner/watchdog exited; both Mac watchers drained and exited following
`PROBES_STOP`.

Estimated cost: first attempt $0.75493, A1 $1.66254, total $2.41747. These are
runner estimates, not final provider billing. No new pod was created during takeover.

## Preserved evidence and limitations

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`.

- `redesign-m3c-gpu-20261008-a1/attempt00/state.json`, `progress.jsonl`,
  `telemetry.jsonl`, and `landed.jsonl` preserve the original run evidence.
- All 22 landed checkpoints matched their recorded SHA-256 hashes during takeover.
  A_s0 and B_s1 have checkpoints through 7.5M; A_s1, B_s0, L_s0 and L_s1 through 10M.
- No 20M final exists. Selection, P2, Phase R, strict gates and serving qualification
  have not been completed for M3-C.
- `redesign-m3c-takeover-20261008/takeover-inventory.json` preserves the checked inventory.
- `redesign-m3c-probes-20261008/A_s0__ckpt_5000k__none.jsonl` is INVALID: its
  scripted/frozen rows use first-attempt checkpoint SHA `ef114113…`, while the mixed
  row uses A1 SHA `6d059248…`. The original file is preserved and must be excluded.
  A corrected A1-only probe completed under
  `redesign-m3c-takeover-20261008/corrected-probes/`. All three mixes contain 16 worlds
  and the verified A1 checkpoint hash; pooled mean mass integral 157.756, descriptive only.
  The corrected probe process exited successfully.

The remaining 5M/10M probes are descriptive only. No early stopping or arm selection
is justified by these incomplete development results.

## Failure diagnosis

The last telemetry interval was 1,132 seconds. That row contains fresh transition
increases for four runs relative to the previous sample, but the row is written
AFTER serial checkpoint/log pulls, so its timestamp is not the sample time.
The final download sweep reports DNS resolution error `Errno 8`.

The runner checks training progress, then performs unbounded serial transfers,
then compares elapsed wall time against the 900-second global training-stall limit.
Thus transfer delay can trigger a false training-stall decision. The recorded
control flow and timing establish this defect; the precise blocked request was not
logged, so the complete network failure sequence cannot be reconstructed.

Probe output reuse checks mix names without validating checkpoint identity, allowing
same-named checkpoints from different attempts to share output rows.

## Recovery boundary

Repairs to runner transfer budgeting and probe identity validation are complete;
validation receipts follow below. These are changes to pinned operational
code and must be recorded in a new launch's source commit and deviation record.
The training algorithm, arms, seeds, 20M target and selection rules stay unchanged.

Intermediate checkpoints save network weights and metadata, not optimizer, replay,
actor or full RNG state. `--init` reconstructs optimizer/replay/target and restarts
counters and schedules. It is not an exact resume. A protocol-preserving retry must
restart all A/B/L x five seeds from the original pinned M2b initialization, in new
training and probe directories. No intermediate checkpoint is a registered final.

A further paid attempt is pending explicit approval: prior authorization specifies
one pod run and the specific A1 relaunch. The proposed retry retains G2 >=4x,
<= $0.95/hour, <=8.5 hours and <=$8.30 for the newly approved attempt, with cleanup and
Mac-side watchdog. The maximum hourly/lifetime product is $8.075; adding A1's estimated
$1.66254 is $9.73754, within the original A1 $10 allowance (first attempt separate).
No approval or amendment is implied by this status record.


## Repair and verification receipt

- Runner: fresh progress observations determine stall decisions before artifact pulls.
  Unreadable logs trigger a distinct bounded observation-loss outcome rather than a
  false trainer kill. A shared 45-second transfer allowance is capped by the job
  deadline; deferred pulls retain SHA verification and atomic landing. Rotating
  checkpoint/log access and alternating log-first polls prevent run starvation.
- Watcher: manifest SHA witnesses are retained and verified against local checkpoint
  bytes. Conflicting same-name checkpoint identities and incompatible existing probe
  rows fail closed, with separate output directories required. Public result names
  remain unchanged for selection compatibility.
- Integrated runner/watcher regression suite: 34 passed in 6.66 seconds.
- Selection and rule-spec regression suite: 10 passed in 0.91 seconds.
- Scoped Black, isort, Flake8, and whitespace checks passed.
- Independent reviews covered both repairs. The runner review specifically reproduced
  and verified correction of six-run transfer starvation and the false stall ordering.
- Corrected A_s0 5M probe: all three 16-world mixes completed, exact A1 SHA verified.

Residual limitation: urllib socket timeout is not a hard wall-clock interruption of
DNS or every response-body read. The allowance bounds subsequent requests, not work
already blocked inside the HTTP library. The independent Mac watchdog remains required.
No new GPU qualification, training completion, selection or promotion is claimed.

Dry-run receipt and the exact committed recovery source are saved outside the pinned
research tree under `redesign-m3c-takeover-20261008/`. The retry proposal remains
unapproved. Repairs and this status record are local only; nothing is pushed.
