# T13 product-roster descriptive measurement

Authority: `docs/research/specs/t13_roster_spec_2026-10-17.md` and advice item 2.
This is descriptive Watch measurement, not a promotion or serving qualification.
Serving defaults, checkpoints, pins, policy math and game mechanics stay unchanged.

- 25 same fresh world seeds per roster; 100 episodes; H5000; 12 snakes; CPU.
- Real T12 `GameSession.step()` then `snapshot()` path, continuous Watch respawns.
- `all_b8`: released default (no checkpoint override), no veto.
- `all_frp3`: `SNAKE_SERVE_CHECKPOINT=frp3-s12` with other serving settings at defaults.
- `all_champion`: `SNAKE_SERVE_CHECKPOINT=champion` with other serving settings at defaults.
- `mixed`: B8 hero and 11 FRP3 opponents with v8 installed through an explicit,
  research-only hook. No product configuration or default changes.

The named FRP3/champion Watch configurations currently wrap **only the hero** with
v8. Their opponents share the selected checkpoint but are unwrapped. Preserve
that actual served behavior; record identities and wrapped IDs rather than label
all twelve snakes as veto-wrapped. The mixed roster explicitly wraps its opponents.

Freshness: SHA-derived 25 world seeds and a separate smoke seed, checked against
T7 registered history, final and both rejected allocations, both T12 audit receipts,
historical FRP exclusion inventories, T3 namespace, M3/M4 Phase R plans, and visible
JSON/JSONL research/artifact evidence. Mandatory T7 evidence is hash-validated.
No opaque/unrecorded/external experiment freshness claim.

Metrics: count decisions and pre-decision logical lengths at each actual snake
update (after respawn); count deaths and kills from authoritative per-frame events;
count boost at actual movement and consumed food items; record post-step live mass,
with dead frames zero. Hero mass is its length sum / 5000; arena mass is the sum over
all twelve snakes / 5000, with per-snake mean also reported. Hero and pooled arena
length distributions are weighted over decisions, not episode medians. Kills taken
means attributed victim events, distinct from all deaths.

Report raw counts and rates per 1,000 decisions, boost fraction, length median/p95/max,
H5000 mean mass, cause counts (head-on/body/wall/self plus any unexpected causes).
Report all six roster pairs for hero and arena, using a paired seed bootstrap:
10,000 resamples of 25 seed indices, identical indices for both sides, 95% percentile
intervals, fixed NumPy RNG seed 20261017. Recompute pooled rates/distributions on
each resample. No multiplicity adjustment; descriptive intervals, no gate verdict.

Execution: one coordinator and four single-thread CPU subprocesses (five processes
at peak); episode files are exclusive-create; no automatic retries; logs and exit
codes retained. No tmux interaction and no operations on unowned processes. Each
commit is followed by fetch, merge origin/main, push HEAD:main, and push the branch.

Implementation details: the current mechanics YAML has six snakes. T13 explicitly
uses T12's constructor-local Watch12 loader hook to meet the spec's twelve-snake
roster; it does not edit the YAML. Record this effective override in episode
identities. Offline release-monitor logging is suppressed, while the real step,
snapshot, collision, food, respawn, policy and veto paths remain in use. The existing
live food-consumption method returns immediately after the first consumed item, so
its per-frame boolean counts actual consumed items (at most one per snake/frame).
Length quantiles follow T12's nearest order statistic: sorted sample index
`round((N - 1) * q)`, with Python/NumPy ties-to-even rounding.
