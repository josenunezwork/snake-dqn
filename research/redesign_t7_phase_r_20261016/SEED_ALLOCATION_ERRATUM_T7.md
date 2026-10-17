# T7 seed-allocation erratum — before any gameplay

2026-10-16. The first proposed round-74 allocation failed freshness. Its seed-4 bank,
index 129, contains world seed `1595692102`, already reserved by the 20,000-world T3
label namespace. `SEED_ALLOCATION_REJECTED_T7.json` preserves all rejected banks and
the exact prior namespace path/hash. No T7 gameplay occurred on this allocation.

To fulfill the spec's mandatory fresh-bank requirement, allocate **round 75** under
the same namespace `redesign-t7-phase-r-b8/v1`, with keys `round75-seed<S>` and the
same deterministic `screen_seeds(192, namespace, key)` construction. Reject the
entire round-74 allocation; do not replace only the colliding world. The fresh round
must pass the complete seed audit before launch. This is a deterministic allocation
repair before any gameplay or outcome observation, not selection based on outcomes.

This sibling supersedes only the round/key in the immutable pre-registration and
its prohibition on discretionary replacement insofar as a failed freshness audit
requires re-registration. No thresholds, clauses, sample sizes, candidate identity,
one-look discipline, serving defaults or scope change. No second survival-guard
amendment is made. The original pre-registration bytes remain unchanged.

The run plan binds this erratum as well as the original registration and passing
audit. The audit must check the rejected round-74 allocation as an additional
exclusion, and record its exact documentary receipt if a repeated identity occurs.
