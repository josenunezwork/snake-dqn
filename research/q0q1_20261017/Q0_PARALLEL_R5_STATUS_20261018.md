# Q0 parallel execution and r5 launch preparation — 2026-10-18

This is an implementation and local timing record, not a Q0 result or authorization
to enqueue. No queue or Colab runtime was touched. Existing registration, inventory
receipt, world bank, checkpoint pins, episode cap and scientific rules are unchanged.

The mass audit now accepts `--workers` (default `max(1, min(cpu_count - 1, 10))`).
Independent H5000 worlds run in spawned processes, each with one Torch thread and the
existing evaluator/probe/observer guards. The parent merges in the exact old
seed/B8/2M/4M/world order. Each B8 world returns its candidate observation stream;
the parent replays those streams through the original reservoir in that same order,
preserving protected witnesses and RNG draws. Completion order, worker count, and
wall-clock progress never enter the scientific receipt. `--workers 1` remains the
serial reference path. The eight-hour runtime and 480-episode caps remain in place.

Progress is flushed at startup, every 120 seconds while worlds are active, and at
completion: episodes done/total, elapsed seconds, ETA and reused count. B8-init only
validates saved observations and now announces its phases. Q0b emits periodic
parent-side heartbeats even while child output is captured, plus stage/cell updates.
Its four GPU measurement cells and four serial CPU parity worlds remain sequential:
parallelizing these would change the capacity measurement and CPU calibration.

## Durable partial results and explicit recovery

After fresh-output admission, a mass job writes
`QUEUE_OUTPUT_DIR/mass-partials/episode-NNNN.json` for every completed world. The
parent atomically publishes each shard and then updates `mass-partials/manifest.json`
with its digest. Partial results survive a queue timeout independently of final export.
Only episodes referenced by the last complete manifest are accepted; an interrupted
unindexed shard is rerun. No automatic retry or queue-output reuse was added.

An operator-reviewed retry must have a **fresh job ID/output directory**. Set both:

```text
Q0Q1_MASS_RESUME_PARTIALS=/absolute/path/to/prior-output/mass-partials/manifest.json
Q0Q1_MASS_RESUME_PARTIALS_SHA256=<verified SHA-256 of that manifest>
```

The old manifest and shards must remain available. The pointer must be outside the
new output directory. Resume checks the explicit manifest hash, every shard hash,
full task identity/order, B8/snapshot/final hashes, bank and admission evidence,
source manifests and CPU/software runtime identity. The deployment config/profile
and opponent pins are revalidated even when every episode is reused. Any mismatch
fails closed. Changed code/runtime/inputs require fresh episodes; this is deliberately
conservative. Do not edit a partial manifest or synthesize missing shards. The r5
manifests do not preselect a resume source because the timed-out r4 code wrote none.
`Q0Q1_MASS_WORKERS` optionally overrides the default worker count.

## Mac measurements and queue budgets

[Mac timing evidence](../../docs/research/q0_mass_mac_timing_20261018.json) records three actual
H5000 B8 episodes on the 18-logical-CPU arm64 Mac, Python 3.12.3 / Torch 2.9.1,
with all numerical thread limits set to one. The fresh-bank collision audit and
released B8 checksum passed. Context initialization took 0.0243 seconds.

| Mix | World seed | Episode seconds | Hero result |
|---|---:|---:|---|
| scripted | 3531715465 | 13.038 | alive at H5000 |
| frozen | 3214373914 | 4.134 | died frame 353; terminal padding to H5000 |
| mixed | 808428031 | 14.591 | alive at H5000 |

These measurements are B8-only; T23 snapshots were unavailable locally. At the
registered 16/8/8 proportions, the weighted mean is 11.200 seconds. Applying that
mean to 480 episodes gives 89.60 minutes serial or 8.96 minutes at ideal ten-worker
scaling. Applying the slowest sample to every episode gives 11.67 minutes at ideal
ten-worker scaling. These are projections, not measured cloud throughput.

The new r5 mass manifest requests at least 12 vCPUs and leaves the default ten workers.
A fourfold slowdown from the slowest-sample projection plus ten minutes for setup,
Drive reads/writes and export is about 57 minutes. The 60-minute estimate and
120-minute queue timeout leave additional margin for CPU differences, cold JIT,
contention, and longer candidate trajectories. Reassess using real cloud progress
when separately authorized; this preparation does not qualify cloud throughput.

| Packet | Estimate minutes | Queue timeout minutes | Rationale |
|---|---:|---:|---|
| mass A100 r5 | 60 | 120 | Conservative ten-worker projection plus setup and timeout margin |
| B8-init A100 r5 | 10 | 30 | Saved-corpus validation, no gameplay; allowance for Drive inventory reads |
| Q0b L4 r5 | 50 | 60 | Four mandatory eight-minute cells plus fixtures, parity, compile/setup/export |
| Q0b A100 r5 | 50 | 60 | Same full packet; existing inclusive 60-minute hard cap retained |

The Q0b timeout is a hard scientific cap, not a promise that an overloaded runtime
will qualify. Slow packets remain INCOMPLETE. All four r5 manifests retain the r4
release/inventory/input/pricing/parity bindings, use fresh IDs and output paths,
update dependency IDs and audit archive locations to r5, and pin the repaired code.
The previous r2/r3/r4 manifests remain unchanged.

## Verification

The regression fixture compares exact receipt bytes for serial and two spawned
workers, including more candidates than reservoir capacity and late coverage
witnesses. It also tests interrupted recovery, no duplicate completed worlds,
changed manifests/shards/tasks/plans/input bindings, fresh destinations, parent pin
guards, and live progress during a long serial world. Queue and Q0b tests cover
argument wiring, strict fresh-output admission and preserved captured subprocess
bytes. The full relevant Q0/Q1/queue suite is run before publication.

The engineering timing JSON lives under `docs/research/`: the historical collision
scanner treats numeric world IDs in `research/**/*.json` as prior study receipts.
Keeping descriptive timing evidence with documentation preserves the existing
scanner and avoids misclassifying this same-bank local check as a foreign study.
The full Mac collision audit is rechecked after recording this evidence.

A separate [real-world parity record](../../docs/research/q0_mass_worker_parity_20261018.json)
compares the old serial closure, new serial worker and two spawned workers on the
three measured H5000 worlds: all episode rows match, the old and merged retained
reservoirs match, the new serial/pool candidate streams match, and all paths count
one excluded trapped state. Canonical scientific row SHA-256 is
`9c64bd27b40e920b5a17aaf2cd807fa509b3f2d02b40f3ad29fc6eba9f549cbf`.
The full Mac Q0 collision audit passes after evidence recording: 160 candidate
worlds, zero collisions, 29,216 registered historical identities. The final relevant
Q0/Q1/queue suite passed **213 tests**; Black, isort, flake8 and diff checks passed.
No CUDA/cloud qualification or full 480-episode scientific packet was run.
