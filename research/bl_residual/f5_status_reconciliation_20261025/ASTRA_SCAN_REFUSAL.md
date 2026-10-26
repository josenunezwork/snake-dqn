**Structural refusal, but not pre-access; renewed authorization is required.** At `f5_status_extractor.py:300–301`, `set(lives) != set(bindings)` failed against 969 frozen IDs (960 selected + nine siblings).

The frozen inventory does not record the NPZ’s observed ID set, so the exact missing/extra IDs remain unproven. Historical `_table` preserves life IDs unchanged but emits only retained rows: a sibling with zero ordinary rows disappears. That is a plausible coverage mismatch, not an established diagnosis; canonicalization is unsupported.

No outcome value was decoded: only eight allowlisted metadata members were opened; whole-file hashing was opaque. Metadata, including statuses, **was** decoded before refusal. The zero-byte reservation and absent sidecar are verified.

**Sol task:** preserve the reservation; prepare synthetic-tested, metadata-only missing/extra-ID reconciliation, proving zero-row omissions without weakening coverage or inventing aliases. Submit revised pins for independent acceptance and one fresh output path. The prior explicit “never retry” authorization cannot transfer.