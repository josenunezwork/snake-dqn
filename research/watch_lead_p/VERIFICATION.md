# P packet-04 verification

Implementer Astra, 2026-10-28. This note replaces packet-03's verification claims
after the operator supplied Argon's REJECT verdict (one blocker, three major,
three minor). No independent acceptance review of the corrections has been
received. No dispatch, allocation, main-bank session, push or release is performed.

The implementation fixes all seven reported boundaries:

- Completion sealing accepts a verified ledger and unit root, ships every sealed
  payload/sidecar with its study tag, and requires exports only for missing units.
  Empty and all-complete completion ledgers refuse before packet creation.
- Scientific identity uses science-sources.json; attempt source pins separately
  bind operational code. Operational fixes preserve completed units; scientific
  dependency changes refuse the old recipe. This boundary is frozen before data.
- Interrupt handlers do no I/O, disable repeated stop signals, and defer unwinding
  while submit is registering a child. Cleanup discovers all owned processes,
  terminates them, lets the executor manager reap, verifies PID disappearance,
  and only then publishes one interruption record. No process-group kills occur.
- Twelve arm workers replace ten whole-session workers. The explicit indivisible
  task forecast includes tail capacity and setup/export; only full matched
  three-arm sessions become completion units. The base/effective gates stay 4/8 h.
- Tests exercise bare death, rescue, descending-Q/ascending-index alternatives,
  all-die fallback and the H300 fatal transition without post-respawn credit.
- Admission verifies committed seal bytes, its exact six-member inventory, ledger
  location, upload destinations and missing-unit exports. Dispatch verifies and
  loads the committed runtime inventory once, retaining traversal/prefix checks.
- LiveWorld preserves caller RNG on construction, refuses negative/out-of-range
  and masked forced actions, and refuses forced actions on non-acting ticks.

Final test counts, forecast receipt, detached checkout identities, sparse-source
proof and exact dry-run command are appended after sealing. Packet-03 and its
hash-bound files remain unchanged. No new judgment or approval is claimed.
