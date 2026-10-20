# Full-state digest v2

The engine-fanout optimization replaces `full_digest`'s recursive scalar/repr
encoding with typed contiguous buffers. Its schema is
`t34-shadow-full-digest/v2` (`FULL_DIGEST_SCHEMA`). Unversioned historical
full-state digest strings use v1 and cannot be compared to v2 strings.

The fixed top-level field order is ordered live bodies, ordered food and corpse
cells with environment offsets, simulator attributes in lexical order,
per-environment MT streams, runtime, hero, opponents, checkpoint, and isolated
global MT stream. Arrays include dtype, shape and contiguous payload bytes;
containers have explicit boundaries. The prior semantic field surface and
timer/address-cache exclusions are retained. The unchanged `_update_value`
encoder remains the authority for historical M1 evidence chains.

Teacher world exports and labels use `exports.state_hash`, which is unchanged.
Full-state digests remain internal to their shadow confirmation path. Path B
controller diagnostics do export confirmation bindings: each binding now includes
`digest_schema` beside `digest`. Existing receipts and registrations are never
rewritten. Source-bound Path B preparations must be regenerated before running
this source; this change does not qualify or release an assisted controller.

The independently loaded v1 oracle is pinned in
`tests/fixtures/t34_fast_digest/README.md`. Divergence tests, frozen teacher export
comparisons, and fresh M1 identity diagnostics accompany the implementation
receipt in `research/t34_exit_b8/ENGINE_FANOUT.md`.
