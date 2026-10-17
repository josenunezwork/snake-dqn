# T7 final seed-allocation erratum — before any gameplay

2026-10-16. Round 75 also fails the full historical freshness scan: world seed
`1193253620` appears in the earlier FRP training exclusion inventories. Preserve
the allocation and exact prior-inventory hash in `SEED_ALLOCATION_REJECTED_75_T7.json`.
Two other raw text matches were transition counters and timestamp integer parts,
not world IDs; the actual exclusion collision alone rejects the entire allocation.
No gameplay has occurred in T7.

Allocate **round 76**, namespace `redesign-t7-phase-r-b8/v1`, keys
`round76-seed<S>`, using the identical 192-world deterministic construction.
This is the next consecutive round, not an outcome-dependent choice. Its initial
comparison to the union of 1,182,309 registered/historical exclusion values finds
no overlap. A passing full repository/artifact seed audit is still required.
The audit additionally excludes both rejected allocations and binds all historical
FRP exclusion inventories, T3/T4 collections and M3/M4 banks.

This sibling supersedes the bank round/key only. Preserve the original registration,
first allocation erratum and both rejected receipts unchanged. Bind all five files
in the final plan. The 192/96/96 design, checkpoint identities, one-look rule,
thresholds and all reporting/serving restrictions are unchanged. There has been no
new gameplay outcome on which to amend the survival guard; no second guard
amendment is made.
