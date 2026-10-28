# Historical TRAIN-01 mechanical evidence

`attempt-08` and `attempt-09` contain byte-for-byte copies of the complete retained
output roots and external operations logs from their real historical runs. Their
receipts were produced by the qualified supervisor, not assembled by a fixture
helper. Each stopped with `RPC_EXEC_POLL_TIMEOUT`, verified VM release, and no
child terminal receipt.

The captures preserve failed renewal-heartbeat transfer telemetry and the native
operations-log `INCOMPLETE: ...; receipt: ...` / `EXIT=1` footer. These are positive
admission cases. Scientific faults injected into a copy must still refuse.

Each `provenance.json` records the original historical paths and a complete
size/SHA-256 inventory. The historical absolute paths remain unchanged inside the
captured evidence. `pinned-commit.raw` preserves the exact Git commit object named
by the original manifest so its identity can be checked and restored if needed.
No original evidence was changed. No attempt15 or r6 data was captured.
