# S1 full-state checkpoint capacity and export liability

Outcome: **CAPACITY_MEASURED_TRANSPORT_UNQUALIFIED**. These local measurements make
the size liability explicit. They authorize no launch, spending, retry, promotion
or change to the registered counter bound. The frozen preregistration and original
failed-counter receipts remain unchanged.

The machine is a Mac with 64 GiB RAM. The registered recipe is raw observations,
500,000 replay rows, 300,000 on-policy labels, 60 observation scalars, 128 worlds,
and two queued actor frames. N3 has six actors and three learner seats; N1 has
three actors and one learner seat. Although a demo directory is required for the
margin and hold-out input, `bv8` sets `demos=None`: the large private demonstration
buffer is **absent** from these snapshots.

| Quantity | N3 | N1 |
|---|---:|---:|
| Observation rows per actor | 134,984 | 253,328 |
| Bytes per raw observation row | 12,449 | 12,449 |
| Exact saved ring/replay/label array capacity | 10,103,294,896 B | 9,481,840,816 B |
| Real serializer measurement, production-shaped capacity arrays | 11,178,149,683 B | 10,490,632,471 B |
| Total raw array/tensor projection including actors and learner | 11,105,121,360 B | 10,000,057,788 B |
| Total serialized checkpoint planning upper allowance | 22,680,004,768 B (21.122 GiB) | 20,268,551,032 B (18.877 GiB) |
| Four snapshot archives, including framing allowance | 91,631,413,568 B | 81,889,140,476 B |

The exact capacity formula is
`actors * (floor(1.5 * 500000 / actors) + 128 * seats * (2 + 8 + 16)) * 12449`
plus `500000 * 32` replay bytes and `300000 * 16` label bytes.
The full allocated arrays are serialized, so their size does not depend on replay
occupancy after one or two million decisions. The eight-byte live shared-ring
counter is represented by a Python integer in the saved state and belongs to the
framing allowance rather than another full array.

For the real serializer measurement, each array is resized to the production
shape by repeating the actual values from the preserved 12,048-decision fixture,
then passed through the same default `torch.save` serializer as
`AsyncCheckpointer`. Arrays are measured one at a time using a counting sink to
keep RAM bounded. One actor's ring layout is serialized and multiplied by actor
count. Summing the standalone array archives includes their individual pickle
and zip framing. These are real serialized production-capacity arrays populated
with observed values; they are not a production trajectory or a measured whole
production `state.pt`. The counting-sink time is not disk, archive, export, or L4
preparation throughput.

The preserved fixture provides an actual whole `state.pt` of **144,477,261 B** and
a gzip export of **46,008,697 B**. This observation is hash-bound in the JSON
receipt. The production total projection adds its exact learner/network/optimizer
array sizes, scales non-network actor arrays from 8 to 128 worlds, retains one
network per actor, and allows up to three deferred frame packets per actor at
1 KiB per driven row. Default protocol-2 NumPy pickles can expand high bytes to
two bytes each, demonstrated by the serializer test; the total planning allowance
therefore uses twice the projected raw bytes plus **64 MiB per actor and learner**
for Python/RNG/pickle framing. That framing allowance and the actor scaling are
explicit provisional planning assumptions. They require full-size validation;
they are not a proved universal bound on every mutable Python object.

All four registered 2M/4M/6M/8M exports count even though local snapshot retention
is two. Published immutable chunks remain in the export directory. No compression
credit is taken: each snapshot archive is bounded for planning by
`ceil(snapshot_bytes * 1.01) + 1 MiB`, including tar/gzip/chunk/index framing.
The ten runs therefore have a snapshot archive planning subtotal of
**867,602,770,220 B**, plus all ten final weights/summary archives.

Export time is `export_bytes / measured_export_bytes_per_second_lower_bound`.
The snapshot subtotal alone would take 9.19 hours at 25 MiB/s, 4.60 hours at
50 MiB/s, or 2.30 hours at 100 MiB/s. These are sensitivities, not measurements.
Full-size capture/save/hash/archive/chunk/copy preparation and final archives are
additional. A timeout cap does not establish that any of this work fits.

`price.py --transport-receipt` and admission require a hash-bound qualified CLI
acceptance receipt with `s1_full_replay_transfer` containing:

```json
{
  "status": "PASS",
  "hardware": "L4",
  "protocol": "s1-chunked-export/v1",
  "verified_on_mac": true,
  "transport_source_sha256": "<current S1 train.py sha256>",
  "transport_source_manifest": {"<current CLI transport source path>": "<sha256>"},
  "exported_bytes": "<number, at least one 1.5 GiB chunk>",
  "elapsed_seconds": "<positive finite measured duration including verification>",
  "bytes_per_second_lower_bound": "<positive finite bound no faster than measured average>"
}
```

The enclosing acceptance receipt must have `qualified: true`. The CLI source
manifest must match the current source delivery, supervisor, transfers, runtime, bootstrap,
acceptance, and acceptance-probe bytes; a throughput measurement from an older
transfer implementation cannot qualify new transport source. Separate bound
N3/N1 full-size measurements must identify L4 hardware, the current source and
capacity contracts, deferred barrier packets, complete preparation scope, and
measured snapshot/final-archive bytes and preparation seconds. The supplied
conservative bounds must dominate those observations. All four preparation and
export costs plus the 30/68-minute training planning base must fit the unchanged
60/136-minute job timeout; cleanup remains a separate five-minute reservation.

Each arm also needs a hash-bound `full_horizon_size_bound` independent validation
receipt (`s1-full-horizon-checkpoint-size-bound/v1`, `status: PASS`). It must bind
the current source, arm and all four snapshot targets, affirm independent
validation and coverage of every noncapacity state object plus framing, and give
a positive `additional_serialized_bytes_upper_bound`. Admission enforces
`snapshot_bytes_upper_bound >= 2 * fixed_capacity_array_bytes + additional_bound`.
A single measured 2M snapshot cannot bound later protocol-2 byte expansion or
growing actor/Python state. The provisional allowances above do not supply this
validation receipt.

Until these measurements exist, the conditional reservation remains
**38.333333333333336 units / 19.166666666666668 GPU-hours** versus usable
**48 units / 20 GPU-hours**, and `complete_upper_bound` remains null. Passing the
counter repair does not qualify the transport. Registration approval and the
existing dispatch/admission guards remain required separately.

Reproduce locally, without contacting Colab:

```bash
./venv/bin/python -m research.s1_mature_20261018.checkpoint_sizing \
  --fixture-archive /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/preflight/failed-counter/output/exports/full-state-snapshot-00000000000000012048-81553026.part-000 \
  --measure-capacity-serialization \
  --out /tmp/s1-checkpoint-size-measurement-new.json
```

The command is create-only. Source hashes in a published receipt describe the
source present at measurement. Changes to serialization, capacity, actor state,
or deferred packet contracts require a fresh measurement rather than modifying
an already hash-bound receipt.
