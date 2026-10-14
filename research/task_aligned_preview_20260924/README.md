# First-lineage preview during second-lineage training

This separate exploratory preview answers the user's request to see the first
completed model playing while the second lineage continues training. It does not
replace the official three-lineage comparison or change its training dose,
checkpoint selection, worlds, criteria, source, or controller.

The first completed lineage, `2026096101`, is fixed before seeing preview results.
Its source mark-4608 parent and final-31,005-rollout candidate each play the same
32 new solo worlds and 32 new mixed-opponent worlds for up to 3,000 frames.
The candidate's global update count is 35,613. All worlds are retained.

The immutable original adapter supplies the already-qualified E32 CPU float32
inference, action masks, native mechanics, opponent controllers and terminal-hero
lifecycle. Death ends a hero's episode; missing horizon frames contribute zero
living mass. The full-horizon mass integral, food, time alive, endpoint survival,
boost and encounter metrics are reported from saved frames. Future trajectories
and RNG consumption may differ despite identical initial worlds.

The preview uses a separate seed namespace, with exact seed-ID exclusion against
the official held IDs and the first lineage's completed training ledger. It does
not claim exhaustive exclusion against unfinished future training seeds or
all-history initial geometry.

## Bounds and concurrency

| Item | Maximum |
|---|---:|
| Hero episodes | 128 |
| Native frames / hero transitions | 384,000 |
| Batched model calls | 12,000 |
| Evaluated model rows, including inactive lanes | 384,000 |
| Opponent transitions | 960,000 |
| Checkpoint loads | 2 |
| Optimizer updates | 0 |

The user explicitly authorized this concurrency exception. A dedicated one-shot
preview lock leaves the original study's two locks unchanged. Preview children
run sequentially on CPU with two intra-op threads, one inter-op thread, and lower
process priority. They do not allocate models on MPS. The preview stops if the
bound second-training process ends or becomes unhealthy, available RAM falls
below 20 GiB, preview RSS exceeds 3 GiB, combined RSS exceeds 8 GiB, or global CPU
utilization exceeds 80% for ten seconds. Admission requires 24 GiB available.
Only preview processes may be terminated by this supervisor.

These margins reduce contention risk; sampled guards cannot guarantee which of
two independent monitors reacts first to every abrupt external resource loss.

Each policy evaluation has a 900-second cap. The independent saved-output audit
has a 180-second cap. A new 2,400-second prospective window includes a separate
120-second handoff reserve. There are no retries or recovery runs. Implementation
review and exact source/authority bindings must precede admission.

The first-lineage result is exploratory. It cannot establish reliability across
training seeds, select a different endpoint, alter the remaining frozen study,
or promote a model.
