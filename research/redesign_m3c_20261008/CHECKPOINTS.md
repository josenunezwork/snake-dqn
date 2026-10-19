# M3-C full-state checkpoints

Full-state checkpointing is opt-in. The existing weights-only checkpoint cadence,
checkpoint format and default training path are unchanged. Full-state files are
trusted Python/PyTorch training artifacts; load only snapshots from your own run.

Add these infrastructure flags to an already registered training command:

```text
--full-state-every <registered transition interval>
--full-state-replay full
--ckpt-local-dir <fast local scratch directory>
--ckpt-durable-dir <durable mounted directory for this run>
```

Resume with the same registered recipe and a fresh `--out` directory, adding
`--resume-full-state <snapshot root>`. The loader validates manifests and SHA-256
before loading the newest complete snapshot. Synchronous and multi-actor snapshots
are distinct contracts; the recipe and storage layout must match. Keep league
checkpoint paths and input artifact identities unchanged across resumes. A corrupt newest snapshot is skipped
when an older verified snapshot remains. A directory with no valid snapshot fails
closed. The Colab notebook also refuses to restart a named run silently when its
snapshot directory contains only corrupt or incomplete named snapshots.

The snapshot includes online and target networks, optimizer, counters and schedule
positions, learner RNGs, actor simulator/RNG/seed streams and pending n-step state,
actor weight lag, league seats, label-buffer state, and demonstration-stream state.
Full replay includes both transition arrays and the observation rings referenced by
transition and label IDs. Restoring indices without their observations is invalid.

Multi-actor barriers hold learner ingestion at the completed decision boundary while
actors finish their queued frames and publish FIFO snapshot markers. Those unconsumed
frames belong to the snapshot too: the learner restores and consumes them before new
actor messages. The learner counter therefore stays within the registered
`actors * worlds * learner_seats - 1` overshoot bound, including the terminal boundary,
without discarding actor work or changing replay sampling and optimizer schedules.

`--full-state-replay none` omits those large replay/observation arrays. It is a
continuation with replay refill, marked `resumed-without-replay`; it is not a bitwise
resume. Empty replay and stale label references must refill before learner updates
resume. The existing minimum replay threshold controls the refill, while transition
counters and schedule positions continue from the snapshot.

## Storage and latency

At replay capacity 500,000, two actors and 64 worlds, transition arrays plus rings
occupy approximately:

| Observation layout | Raw | With `--obs-pack` |
| --- | ---: | ---: |
| ego2s-b | 9.31 GB | 2.42 GB |
| ego2s-bv8 | 9.40 GB | 2.51 GB |

These are decimal GB, excluding models, Adam state, labels, actor simulator state
and serialization overhead. A private default demonstration buffer can add about
3.71 GB raw or 0.97 GB packed; a mapped demonstration store instead saves its index
and validates the store layout. Keep enough RAM for live state plus one detached
snapshot, and disk for retained snapshots and an in-progress atomic write.

CUDA model tensors are copied into pinned CPU storage and fenced before training
can mutate their source. CPU arrays are detached while writers are held at a
consistent boundary. `capture_stall_ms` includes boundary synchronization and state
capture; actor-barrier latency is recorded separately as `actor_barrier_ms`. Local
serialization/fsync and durable copy/retry durations are recorded
separately. The small-state CPU test requires capture below 50 ms. **A full copy of
multi-gigabyte host observation rings does not carry a 50 ms guarantee.** The code
reports the measured pause; it does not hide it in background-copy timing. CUDA and
full-size Colab latency require measurement on the intended runtime.

Local serialization/fsync and durable copying use separate background workers.
Neither Drive speed nor copy retry sleeps block a learner checkpoint submission.
Only one in-memory snapshot can be awaiting serialization; a busy interval is
skipped. Waiting durable uploads coalesce to the newest snapshot. The last two
complete snapshots are retained, with at most one extra local snapshot protected
while an older upload is actively reading it. Exhausted copy failures are logged
and remain locally recoverable subject to the same retention bound; a newer
snapshot attempts durable copying again.

Each snapshot directory contains `state.pt` and a `manifest.json` recording its
size, SHA-256, transition count and metadata. The local writer uses temporary files,
fsync and rename. The durable worker publishes the manifest last, after copying and
verifying the payload. Partially written directories are never resume candidates. An exclusive local
writer lease allows a new writer to reclaim temporary directories left by an
interrupted process without touching another active writer.
A final snapshot is requested at orderly exit, including a configured wall-time
stop. After training ends, the process stays alive while queued I/O drains; this
may take minutes for a large Drive upload. Failed final writes
make the summary's `full_state_complete` false; the long-run notebook then keeps
the runtime assigned. Abrupt VM loss can lose work since the last completed durable
snapshot, so choose checkpoint cadence and Drive capacity accordingly.

The infrastructure does not select or authorize a training recipe. See the separate
[T16 long-run notebook guide](../t16_colab/long_run.md).

Bit-identical continuation is tested on the deterministic synchronous CPU path.
The asynchronous multi-actor trainer already depends on OS scheduling and actor
weight lag: restoring complete state does not make that execution order, compiled
kernels, or different GPU architectures bit-reproducible. No GPU latency claim is
inferred from CPU tests.
