# S1 amendment 01: parallel L4 execution (2026-10-19 PT)

**Status: APPROVED_S1_PARALLEL_20261019.** Owner-approved operational amendment,
recorded prospectively by `docs/research/specs/2026-10-19_s1_parallel_lanes.md`.
The original approved `PREREGISTRATION.md` bytes remain unchanged.

The owner instructed at approximately **2026-10-19 10:25 PT**, in chat:
“how many can u run thru colab cli at once do as much as u need to work faster”.
The planner's specification records that **no S1 training metric or efficacy
data had been observed by anyone** when this amendment was authorized. This
statement records that pre-data authorization; implementation does not inspect
training metrics to establish it.

This amendment supersedes only the original requirements for “one L4 run at a
time”, no “parallel L4 study run”, and serialized primary/replacement runs.
An amended plan may use **one, two, or three concurrent L4 lanes**, with at most
**three account-wide assignments including every other assignment**. Each lane
uses the existing qualified Colab CLI supervisor. Parallel admission binds this
file by SHA-256 and requires fresh evidence identifying every active assignment
as an outstanding study lane with its measured positive rate. Unknown or
unreconciled assignments block admission. The supervisor remains the sole
allocator and stop broker; this amendment does not authorize dispatch itself.

The arms use independent, already registered seed blocks and fresh per-run
optimizer/replay state. Scheduling cannot change the fixed 8,000,000 learner
decision endpoint, B8 initializer, recipe, inputs, world banks, metrics,
statistics, thresholds, final-only selection, or single complete Mac efficacy
look. Interleaving N3 and N1 changes only execution order. There is no efficacy
selection, interim look, seed replacement, automatic retry, extension, or serving
change. A verified tripwire halts future work for its arm; healthy lanes and the
other arm continue. Already running attempts retain their immutable evidence.
Operational failures retain the original once-per-arm, explicitly prepared,
from-scratch rerun rule and INVALID/INCOMPLETE precedence.

All resource caps and the twenty-percent withheld reserve are unchanged:
nominal **60 units / 25 GPU-hours / 24 Mac worker-hours / 48 elapsed hours**;
usable **48 / 20 / 19.2 / 38.4**, respectively. Acceptance, setup, idle,
export, failures and cleanup remain charged. Units, GPU-allocation hours and
Mac worker-hours remain additive across all attempts. Only the prospective
elapsed upper bound is recomputed for the admitted lane count using a
conservative scheduling bound; it includes serial acceptance and Mac evaluation.
Elapsed time already consumed is never divided by the lane count or reset.
Spent plus outstanding plus all remaining liabilities must fit every usable
cap under the ledger lock before any allocation. First cap still wins.

The already-running **dispatch-04** is governed by its original immutable plan
and loaded serial dispatcher. This amendment does not migrate its outstanding
reservation, alter its files, stop it, or authorize an overlapping dispatcher.
An operator must establish a quiescent handoff before using the new scheduler;
one job's settlement alone does not prove the old dispatcher has stopped.
