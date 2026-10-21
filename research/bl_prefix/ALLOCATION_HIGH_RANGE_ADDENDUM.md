# Pre-data high-range seed family (v2)

No gameplay exists. The v1 family is retired after whole-round historical
collisions in rounds 0 through 23. All rejection/operational audit files are
retained outside Git; no individual replacement seed was selected. The incomplete
first scan and receipt-alias inspection attempts are operational failures, not
experimental data.

Freeze a new WHOLE family before data:

- Domain: `bl-prefix-e6e7-development/20261021/v2`.
- Round 0 keys: source-world / future-rng / exploration / preflight.
  Later rounds use the same keys suffixed `/roundN`.
- Derive via existing `dev_screen.screen_seeds(count, domain, key)`, then add
  exactly `2**40` to EVERY value. Preserve order. This stays below `2**63`.
- Counts remain source4096 / future3000 / exploration4000 / burned preflight128.
- Audit consecutive whole rounds 0 through100 and choose the first collision-free
  allocation. Internal, historical and visible collisions still reject the
  entire round. Structural/missing-inventory errors stop the audit.

The offset avoids the densely occupied historical 32-bit identity range; it is
not a substitute for a complete collision audit. CPython EnvRng preserves the
full integer, and scripted anchors support signed-int64 seeds. The actual Mac
preflight must prove high-range source/future RNG and snapshot round trips on
the real entrypoints. If any runtime truncates or rejects the seeds, stop
INCOMPLETE and register a prospective repair before gameplay.

No controller, mix, matching rule, horizon, sample size, statistic, kill/continue
threshold, or resource cap changes. The v1 registrations remain byte-identical;
this sibling addendum supersedes only their seed-family allocation recipe.
