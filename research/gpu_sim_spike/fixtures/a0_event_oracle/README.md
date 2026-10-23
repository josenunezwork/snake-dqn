# A0 event-path oracle fixture

This fixture freezes the unedited `DeviceSim` source at revision
`01334e1ed04b49ebf61ab16a6df0cf6c7e4c2884` (the `01334e1e` baseline supplied to
the A0 work package). The source was captured with `git show` into the sibling
`research/gpu_sim_spike/device_sim_frozen.py`; `test_a0_event_oracle.py` loads
that file under a private module name and checks its bytes against the SHA-256
recorded in `ORACLE_RECEIPT.json`. It does not import the working-tree
simulator for the oracle arm.

Every fixture allocates the production body capacity `cap=10400`. The bounded
capacity fixture avoids a baseline full `S * cap` tick by invoking the frozen
and candidate append paths with a bounded list of actual corpse cells. The
single and simultaneous death fixtures each execute one full tick at this
capacity, which verifies compaction from actual dead segments without turning
the 1,000-tick food check into a throughput measurement. The 1,000-tick food
fixture compares ordered food, corpse flags, state bytes and the counter-RNG
stream on every isolated maintenance tick.

The production-capacity food case also compares four worlds in one call: full,
refill, no free lattice space, and exhausted refill trials. It checks the
active-row scatter result and the fixed three-draw counter-RNG advance for
every world.

The receipt is a scope and identity record, not a benchmark result or dispatch
approval.
