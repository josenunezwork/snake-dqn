# Phase 0 teacher speed profile — 2026-10-19

Outcome: the requested batching and input reuse are implemented and fixture parity passes,
but the **20x Mac CPU target is not met**. No Phase 0 selection episodes were run.
The registered N=8/H=128/epsilon=.02, L>=600 OR margin<0 trigger and leaf scale=1
are unchanged. All profile rows below use disjoint L800 valid-serpentine timing
fixtures, real strict-profile opponents, one native CPU thread, and three rounds
with warm-up excluded. These fixtures include early rollout deaths.

## Profile before optimization

The old teacher was profiled before editing it. Root cProfile measured 5.879,
6.045 and 6.824 seconds for scripted/frozen/mixed, with 6,315/6,340/7,516
`build_ego_raster` and B8 forward calls respectively. Observation construction
and B8 forwards dominated; simulation and replication were small contributors.
The original manual-timer receipt is preserved as
[PROFILE_BASELINE_RECEIPT.json](PROFILE_BASELINE_RECEIPT.json).

The following corrected, reusable-harness reference profile uses the unchanged
old scorer (`_reference_scores`) retained in the implementation. Each row is
its median-total-time round, so its category times sum to the same total.
The frozen oracle in `tests/fixtures/phase0_teacher_speed` independently pins
and verifies the old implementation bytes from commit `45069720`.

| Mix | Total s | Sim | Obs including v8 | B8 forward | Frozen forward | Scripted | Clone | Other host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scripted | 4.4209 | 0.1394 | 2.2717 | 1.8809 | 0.0000 | 0.0092 | 0.0192 | 0.1005 |
| frozen | 4.8183 | 0.1695 | 2.3267 | 1.8317 | 0.0982 | 0.0000 | 0.0185 | 0.3737 |
| mixed | 5.5517 | 0.1688 | 2.7554 | 2.1963 | 0.0738 | 0.0145 | 0.0192 | 0.3237 |

Reference evidence: [PROFILE_CPU_V1.json](PROFILE_CPU_V1.json). Its observation
instrumentation patches both aliases of `build_ego_raster`. Here, other host
includes Vector61 carry featurization, dispatch and Python overhead.

## Final exclusive profiles

Nested v8 feature time is removed from the raster/scalar category; Vector61
prepare/observe is separated from the remaining host overhead. The rows use
median-total-time rounds, rather than adding independently selected medians.
B8 timings include tensor conversion, transfers and synchronization. MPS moves
only the batched B8 network; simulation and frozen opponents stay on CPU.

| Device / mix | Total s | Sim | Raster/scalars | v8 features | B8 forward | Frozen forward | Scripted | Clone | V61 carry | Other host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU / scripted | 1.0420 | 0.1320 | 0.0757 | 0.0940 | 0.6773 | 0.0000 | 0.0062 | 0.0187 | 0.0026 | 0.0356 |
| CPU / frozen | 1.4017 | 0.1582 | 0.0812 | 0.0989 | 0.6642 | 0.1023 | 0.0000 | 0.0182 | 0.2326 | 0.0460 |
| CPU / mixed | 1.6572 | 0.1614 | 0.0874 | 0.1191 | 0.9582 | 0.0644 | 0.0136 | 0.0188 | 0.1859 | 0.0485 |
| MPS / scripted | 0.7346 | 0.1311 | 0.0754 | 0.0931 | 0.3680 | 0.0000 | 0.0060 | 0.0183 | 0.0027 | 0.0399 |
| MPS / frozen | 1.1693 | 0.1601 | 0.0814 | 0.0985 | 0.4429 | 0.0783 | 0.0000 | 0.0182 | 0.2337 | 0.0563 |
| MPS / mixed | 1.1713 | 0.1648 | 0.0898 | 0.1202 | 0.4546 | 0.0600 | 0.0140 | 0.0184 | 0.1896 | 0.0600 |

Evidence: [PROFILE_CPU_FINAL.json](PROFILE_CPU_FINAL.json),
[PROFILE_MPS_FINAL.json](PROFILE_MPS_FINAL.json). The final CPU makes 131 raster
builds and 131–136 B8 forward calls per search, including the outer decision,
leaf and occasional rowwise tie checks. It replicates the world once. The
rollout work remains N times legal actions; only equal neural inputs share a
forward result. There is one batched B8 forward per rollout step, with additional
CPU batch-1 forwards solely for close Q ties. Frozen networks retain their
shared grouped dispatch.

| Mix | Reference CPU s | Final CPU s | CPU speedup | Final MPS s | MPS vs reference | MPS vs final CPU |
|---|---:|---:|---:|---:|---:|---:|
| scripted | 4.421 | 1.042 | 4.24x | 0.735 | 6.02x | 1.42x |
| frozen | 4.818 | 1.402 | 3.44x | 1.169 | 4.12x | 1.20x |
| mixed | 5.552 | 1.657 | 3.35x | 1.171 | 4.74x | 1.41x |

The remaining CPU bottleneck is B8 inference, followed by simulator/observation
work and frozen runtime carry. A channels-last CPU layout probe did not help:
48-row forwards took 19.28 ms contiguous versus 19.31 ms channels-last (ten
measured calls, three warm-ups, one thread; probe fixture Q difference zero).
That layout change was not adopted. Intermediate batching-only and input-reuse
receipts are retained as `PROFILE_{CPU,MPS}_V{1,2}.json`.

## Equivalence and limitations

All 18 CPU/MPS mix-by-length fixtures use N8/H128/epsilon=.02 and match the
independently frozen CPU teacher's chosen action and scores within absolute
1e-3, with no relative tolerance. The L800 raw receipts show score differences
below 1e-6. Repeated CPU runs are exactly deterministic. Search does not mutate
the prepared source simulator or runtime arrays.

Batching and removal of identical neural inputs can reassociate float32 math.
The 1e-3 near-Q comparison threshold reruns close rows on the original CPU
batch-1 path. Final scores separated by at most `2*1e-3*leaf_scale` rerun the
entire reference search. Exact score ties retain the first legal action. This
threshold is a tested fixture safeguard, not a proven bound for every state or
accelerator. No universal trajectory-identity or cross-architecture claim is made.
The CPU oracle weights are immutable; the optional device cache refreshes when
the source network object changes and assumes no in-place weight updates.

Other host overhead includes input deduplication and allocations, not just
Python bytecode. No otherwise-idle-host claim is made. Native pools were capped
at one per worker; local scaling used at most six workers. Operational guards
checked AC, lid and thermals. CUDA is unavailable and untested.

## Reproduce

Run from `/Users/josenunez/Projects/ml/snake-dqn-search-teacher` and choose a fresh
output path; the harness refuses existing receipts:

```bash
nice -n 10 ./venv/bin/python -m research.phase0_search_20261019.profile_teacher \
  --implementation both --device cpu --length 800 --rounds 3 \
  --out docs/agent_notes/phase0-speed-repeat/profile-cpu.json
nice -n 10 ./venv/bin/python -m research.phase0_search_20261019.profile_teacher \
  --implementation batched --device mps --length 800 --rounds 3 \
  --out docs/agent_notes/phase0-speed-repeat/profile-mps.json
```

## Initial receipt errata

The first manual timer patched only one raster-function alias. Its observation
category therefore undercounts raster work; that time remains in its residual.
Use the corrected reference table above for category comparisons. The initial
receipt's Python version `3.12.2` is a metadata transcription error; the current
venv and reusable receipts report `3.12.3`. An earlier prose statement that all
three reference winners were action 3 was also incorrect: the original raw
score vectors already show scripted=3, frozen=3, mixed=1. The immutable receipt
is unchanged; the reusable reference reproduces all three original score vectors.
