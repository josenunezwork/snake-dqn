# B-3 packet 02: Argon fixes

This sibling note supersedes the affected implementation descriptions in `B3.md`
for packet 02. Packet 01 remains historical evidence with its original bytes.
Argon's `review-sbx-b3/answer.md` (2026-10-26, ACCEPT-WITH-FIXES) requested these
changes before dispatch; no A100 execution or scientific result is claimed here.

The local geometry flood treats the previous head as newly occupied own body
until absolute time `ceil(length)` and a boosted intermediate cell until
`ceil(length) + 1`. Both use the original observation's time origin. They block
backward traversal through the neck until release and cannot create a visible
tail-reachability claim, even if the original raster marked either cell TTL-one.
A legal landing on an original TTL-one cell still counts as reaching that
tail-release cell. Enemy occupancy stays fixed, and the crop remains a local
approximation. The unused Python flood implementation was removed. Geometry
outputs and the runtime geometry tensor explicitly use float32.

Retrieval omits unknown/illegal/unselectable training labels. With zero known
votes for the query's base action, its probability slot uses `1.0` as a **no-veto
sentinel**, not an estimated survival rate. This also covers an absent same-base
training pool. Alternatives with zero known votes retain zero, preventing their
selection without evidence. Query masks cannot erase the base's sentinel.

All veto variants now require a strictly safer selectable alternative:
`p_surv(base) < tau_veto` and `p_surv(alternative) > p_surv(base)`. The `.1` and
`.25` advantage variants also require the registered strict advantage. Equal,
worse, and all-zero alternatives preserve base; threshold zero remains the
exact B8 control. This rule replaces the null-advantage wording in `B3.md`.

Packet 02 retains the admitted D1024 input, frozen B8, input registry, namespace
`a100-dev/sandbox-b/heads-v3`, allocation `split-v1`, parent split salt
`a100-dev/sandbox-b/split-v1`, 12 configurations, 432 points, 6,000 full-sweep
neural updates, and 60/120-minute base/effective runaway guard. Its regenerated
empty ledger binds the fixed source recipe; packet-01 ledger units cannot be
reused under it. See `b3-packet-02/OPERATOR.md` and `VERIFICATION.json` for exact
commands and current verification evidence. Actual dispatch remains the
operator's step.
