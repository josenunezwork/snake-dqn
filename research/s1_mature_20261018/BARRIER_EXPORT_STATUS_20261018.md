# S1 barrier correction and checkpoint liability — 2026-10-18

**Operational engineering evidence only. Admission remains REFUSED; registration remains DRAFT.**
This sibling status supersedes the counter-blocker description in
`ADMISSION_STATUS_FINAL.json` and `IMPLEMENTATION_RECEIPT.md` only where the new
source-bound verification receipt explicitly passes. Original evidence, the
registered overshoot bound, scientific rules, endpoints, budgets and approvals
remain unchanged. No live Colab operation or study training was performed.

## Barrier behavior

The learner stops ingesting ordinary actor packets while the full-state barrier
is pending. It drains transport queues through each actor's FIFO snapshot marker
and saves the unconsumed packets with the actor/ring state. A nonterminal release
returns those packets to learner-local per-actor FIFOs. Full-replay resume consumes
these FIFOs before new actor traffic, without preloading bounded transport queues.
Remaining FIFO packets survive another barrier. Replay-none continuation omits
these ring-dependent packets and records that discard; it is not an exact resume.

The existing learner round can ingest at most `actors * worlds * learner_seats`
decisions before requesting the barrier. No decisions are ingested during the
barrier. Thus the last round crosses a positive target by at most
`actors * worlds * learner_seats - 1`, including the final 8M target. The bound
has not been raised. The adapter's smoke target is restored from 12,001 to the
original 12,000.

## Measured checkpoint capacity and serializer liability

`CHECKPOINT_SIZE_MEASUREMENT.json` derives allocation sizes from the actual S1
commands, ring-capacity function, dtypes and shapes, and measures `torch.save` of
production-capacity arrays tiled from the trusted real Mac fixture. It avoids
training 1–2M operational decisions simply to fill fixed-size allocated arrays.
S1 bv8 has no private demonstration buffer (`demos=None`).

| Quantity | N3 (6 actors, 128 worlds, 3 seats) | N1 (3 actors, 128 worlds, 1 seat) |
| --- | ---: | ---: |
| Exact fixed replay/ring/label raw bytes | 10,103,294,896 | 9,481,840,816 |
| Measured production-capacity standalone array serialization bytes | 11,178,149,683 | 10,490,632,471 |
| Projected full-state raw bytes including actors/models/optimizer/pending packets | 11,105,121,360 | 10,000,057,788 |
| Provisional serialized snapshot planning bound | 22,680,004,768 | 20,268,551,032 |

The array serialization measurements use the real protocol-2 serializer and
representative fixture values. They are not a measurement of a complete
production checkpoint. The total planning bounds allow up to 2× raw NumPy bytes,
project actor arrays from 8 to 128 worlds, include deferred-packet allowance, and
reserve 64 MiB per actor plus learner for object/framing overhead. That last
allowance and linear projection remain assumptions needing full-size validation;
they do not qualify an L4 runtime or a complete upper bound.

For one job, the prospective export liability in seconds is:

```
[4 * (ceil(snapshot_bytes_upper_bound * 1.01) + 1 MiB)
 + final_archive_bytes_upper_bound]
 / measured_export_bytes_per_second_lower_bound
 + 4 * snapshot_prepare_seconds_upper_bound
 + final_archive_prepare_seconds_upper_bound
```

All four 2/4/6/8M snapshot exports count, even though local snapshot retention is
two. The 1% plus 1 MiB allowance covers archive/chunk framing with no compression
credit. At the provisional snapshot sizes, the snapshot-only export bytes are
91,631,413,568 per N3 job and 81,889,140,476 per N1 job; all ten jobs total
867,602,770,220 bytes, plus final archives. Throughput is deliberately null until
bound to real CLI acceptance evidence.

Admission requires `s1_full_replay_transfer` inside the hash-bound CLI acceptance
receipt: passing L4 chunk protocol, Mac verification, current transport source
hash, exported bytes, elapsed seconds and a finite positive measured conservative
`bytes_per_second_lower_bound`. A complete verified 1.5 GiB chunk is the minimum
sample. It also requires source-bound production-shape N3/N1 size and preparation
measurements, covering capture/save/hash/archive/chunk-copy and final archives.
A timeout alone does not establish liability. A separate independently validated,
source-bound `full_horizon_size_bound` receipt must cover all four checkpoints,
all additional state and framing, and worst-case protocol-2 expansion of fixed
arrays. Admission requires at least twice fixed-capacity raw bytes plus that
validated additional-state allowance. Training (30/68 minutes) plus all
measured preparation and transfer must fit the unchanged 60/136-minute job caps.

## Conditional forecast and admission dependencies

The conditional reservations retain the ten 60/136-minute job timeouts, five
cleanup minutes per job, four acceptance units/two conservative GPU hours, and
existing disjoint Mac evaluation timing. They are not a complete admitted bound:

| Dimension | Conditional reservation | Usable cap |
| --- | ---: | ---: |
| Units | 38.333333 | 48 |
| GPU allocation hours | 19.166667 | 20 |
| Mac worker hours | 15.865673 | 19.2 |
| Elapsed hours | 27.099503 | 38.4 |

The complete upper bound remains null until full-size preparation and measured
CLI transfer evidence qualify the formula inside these reservations. The
independent CLI acceptance/review, fresh account evidence, complete registration
approval and clean published admission plan are still required. This work does
not launch or authorize any of them.

## Verification evidence

The new source-bound counter preflight and final local verification receipt are
linked from the README. Historical failed fixtures remain preserved. A first
intermediate development patch was rejected during review because it discarded
held packets on normal release; its local artifacts are marked
`barrier-fix-01/DIAGNOSTIC_ONLY.json` and are not qualification evidence.

`TRAIN_PREFLIGHT_BARRIER_FIX.json` binds the final trainer and adapter bytes to
the real 12,000-decision run: 12,000 decisions, zero overshoot, clean actors,
zero overruns, and a recovered complete full state. Its 11,760 emitted rows equal
45 updates times 256 plus 240 pending update debt; the 240 decision/replay delta
equals pending n-step rows. The saved terminal snapshot retains another 48 actor
decisions as unconsumed packets. They do not enter the published final counter.

The frozen no-barrier CPU oracle compares all replay draws and every learner
update's online/target/optimizer/replay/RNG bytes against the original
`474b3c2e` trainer. The modeled 8M endpoint test preloads counters; it does not
claim to have trained 8M decisions locally. The focused integrated suite passed
270 tests with six slow tests deselected. Additional transport-bound regression
coverage was then added and verified separately; exact commands, later checks,
and broad-suite limitations are recorded in `BARRIER_FIX_VERIFICATION.json`.
