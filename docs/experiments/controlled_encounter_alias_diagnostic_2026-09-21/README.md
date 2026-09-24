# Controlled-encounter alias diagnostic — 2026-09-21

This diagnostic investigated the two exact input aliases that rejected the
controlled-encounter imitation collection. It reconstructed the six affected
states and compared the saved arrays, native frames, and original teacher labels.
It is a data-admission diagnostic only: it contains no learner, promotion, or
architecture result.

## Reconstruction result

The canonical third diagnostic passed. Each of the six reconstructed states
matched its saved NPZ tensors, native frames, and original labels. The two
groups have these shared food-only action sets:

| Group | States | Shared food-only actions |
| --- | ---: | --- |
| 1 | 4 | `[0, 1, 2]` |
| 2 | 2 | `[0, 1]` |

The old teacher’s clearance tiebreak distinguishes the labels through a hidden,
far respawned enemy. The raw hero tactical inputs are equal within each group;
the raw strategic and scalar inputs differ. After the established transform,
however, all network inputs are identical. That establishes why the original
no-alias admission gate correctly treated the rows as conflicting despite their
different native states.

## RNG check and limit

Each group had one unique donor RNG state. The diagnostic performed 20 donor
swaps, but every swap was non-contrast: the available donor states do not vary
the relevant hidden factor within either alias group. The future RNG question is
therefore **inconclusive**, rather than evidence for or against an RNG-based
repair.

The next experiment drops clearance from the label objective for every saved
state, keeps the original teacher trajectories and gameplay baselines, and
reapplies all admission gates before three-seed learning. This tests whether
food-optimal sets provide a learnable target. Full H4 food search still uses the
native realized future; these six cases do not establish that all such labels
are observation-compatible.

## Execution record

The first two guarded attempts failed before producing a report: the first after
1.6714 seconds with a packed-observation key error, and the second after 2.0738
seconds because whole-object pickle hashes differed after an equivalent
self-swap. The third attempt compared canonical state values and completed in
5.2059 seconds. Five of six self-swaps had equal canonical values but different
pickle hashes, confirming that the pickle bytes were unsuitable for comparing
these equivalent objects.

All three receipts are retained under the frozen diagnostic root and remain
within the 90-second total budget: **8.9512 seconds** in total. Peak RSS was
492,765,184 bytes (about 470 MiB), and the minimum available-memory sample was
31,992,020,992 bytes (about 29.8 GiB). The successful reconstruction does not reopen
the imitation study, authorize training, or change the rejected original
criteria.

## Source records

- `controlled-encounter-alias-diagnostic/diagnose-v3/report.json`
- `controlled-encounter-alias-diagnostic/supervisor-runs/diagnose/receipt.json`
- `controlled-encounter-alias-diagnostic/supervisor-runs/diagnose-v2/receipt.json`
- `controlled-encounter-alias-diagnostic/supervisor-runs/diagnose-v3/receipt.json`
- [Prior imitation admission closeout](../controlled_encounter_imitation_2026-09-21/README.md)
