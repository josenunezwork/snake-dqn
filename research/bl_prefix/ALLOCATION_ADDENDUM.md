# Pre-data allocator and terminal-prefix clarifications

Round 0's entire scientific allocation is rejected. Registered inventory hits:
1412664698 and 2562925658; visible-history hits additionally 80930 and 6326472.
No gameplay was collected. The earlier eight-seed preflight audit is superseded
because it preceded the final published namespace coverage check; it cannot
admit gameplay. Its immutable files remain as operational history.

Before data, freeze this allocator: try consecutive WHOLE rounds 1 through 100,
choose the first with zero internal, registered and visible-history collisions.
Each uses the same domain `bl-prefix-e6e7-development/20261021/v1` and
`screen_seeds(count, domain, '<kind>/roundN')` keys, where kind is source-world,
future-rng, exploration or preflight. Preserve every rejection. Never replace
individual seeds. Stop INCOMPLETE if no round passes. Source/future/exploration
counts remain 4096/3000/4000; the separate burned preflight allocation is 128
seeds, enough to cover source, fixture-model initialisation and policy/RNG tests.
The chosen round and full ordered vectors/digests are bound in the audit receipt.
Preflight uses only that passing round's burned preflight vector.

The E7 selected prefix can end at terminal hero death before decision 64.
Confirmation replays only the actions that actually existed (at most 64), then
uses greedy B8. Record the actual prefix length and training death; do not
invent actions after death or classify an ordinary failed search as a mechanical
failure. The two confirmation streams and +20/survival rule are unchanged.

Source-bank construction may replay each selected source once to materialise
its ancestor roots/history. This work counts within E6's existing 4u cap, along
with all primary/secondary continuations. Cached first-pass length trajectories
choose ordinary matches; never replay every survivor for every death. No sample
size, endpoint, confidence bound or decision rule changes here.
