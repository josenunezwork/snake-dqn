development diagnostic, not gate evidence

# T1 Part A: exact-replay commitment and contrast study (2026-10-15)

## Routing rules (fixed before the commitment numbers)

- R1: >= 60% of the 49 late deaths (L >= 800) have a surviving alternative within 150 decisions AND
  median H_LAD <= 100 -> "T3 eligible". R1': < 30% have any alternative within 300, or median
  H_LAD > 300 -> "T3 off".
- R2: Q-gap at LAD < 2 Q units in a majority -> "value learning is the limit" (Part C's probe also
  feeds R2). R2': Q-gap > 5 favouring the fatal action in a majority -> "perception is the limit"
  (decided jointly with Part C).
- R3: the surviving action differs from the chosen one on tail-in-region or on boundary TTL in >= 70%
  of LADs -> tail-topology inputs GO; < 50% -> NO_GO.
- R4: a boost at LAD survives in >= 20% of late deaths -> include boost labels in T3.

## Method and status

Implementation and the mandatory full-state clone precondition are in progress. No
commitment result or rule verdict is claimed in this initial, pre-result record.

The census's `fatal_t` and `trap_onset.t0` are zero-based decision indices; its `frame`
is one-based after preparation. Scans use decision indices. At `d = fatal_t - 300`,
300 transitions stop one transition before the fatal one; the recorded-action
control continues one extra transition to check death reproduction. The control
is excluded from the surviving *alternative* set even if alive at the 300-transition
endpoint. Advisory-unsafe recorded actions remain controls, never legal alternatives.

The optional incumbent/champion reference deaths require their stateful vector61
runtime and v8 wrapper. They will be reported separately if run; B8 has no veto.

Small counts limit interpretation. "No survivor" means "no survivor under B8's own
greedy continuation" at the scanned decisions, not proof that no possible policy
can escape. R2/R2' remain subject to Part C's independent probe.
