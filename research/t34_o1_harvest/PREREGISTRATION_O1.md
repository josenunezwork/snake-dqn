# O1 harvest-branch census — frozen before data

Authority: owner's explicit O1 implementation instruction; adopted O1 in
`docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md` §3 governs over
Astra's prospective proposals. Tier 1 mechanism screen, not efficacy, promotion,
serving qualification or release. No dispatch is authorized by this package.
Implementation, frozen registration JSON, source bindings and seed audit must be
committed and published before any gameplay, including the excluded Mac fixture.
One fixed-N look only; no extension, automatic retry, replacement or second look.

## Hypothesis and design

At decisions where bare B8 s3 survives its H500 continuation, a single legal
first-action deviation followed by B8 yields ≥ +5 mean branch mass
(`delta_A500 = (Σ len_alt − Σ len_bare)/500`, dead frames zero) with equal horizon
survival, and the gain is attributable to death-drop intake, in at least 8 of 48
worlds including ≥4 frozen/mixed.

48 FRESH terminal-hero H5000 worlds, 16 scripted / 16 frozen / 16 mixed; hero slot
0; other snakes respawn. Use Round 0's `_rows` and `trajectory.build_world`
read-only: same ordered opponent rosters, v8 opponents, profile and released bare
B8 s3 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
Assert mechanics v2 and record `asdict(sim.cfg)`; correctly label v2. No rescue
trigger, rescue ranking, hindsight intervention, training or mechanics changes.

Seed namespace `t34-o1-harvest-20261021/v1`, separate study/timing/mac-preflight/
parity subspaces. Reuse Round 0-D's fail-closed seed-audit tooling including its
positive controls, explicit registered banks, T3 census, source/docs/artifact
text and CSV/TSV, and concrete historical FRP exclusions. Any collision rejects
the entire allocation, without rerolling. Never open the sealed
`snake-dqn-artifacts/t34-round0-study-01/`; its identities are checked from
`research/t34_round0/BANK_ROUND0.json`. Coverage limits are retained in the audit.

At most eight roots/world: four time roots at prepared decisions 1000, 2000,
3000, 4000 if hero alive, plus at most four event roots. An event is an opponent
death whose post-movement head is within Manhattan 30 GRID CELLS of hero head,
with the hero alive. Record every event, deduplicate prospective event windows
64 decisions apart (first in window wins), then choose up to four by a fixed
SHA-256 priority reservoir keyed by world seed and decision. This is a uniform
hash reservoir independent of outcomes; log eligible count and inclusion
probability min(1,4/count). Coincident time/event roots merge and count once.
Events are captured at the next prepared decision. Branches never alter the base
trajectory or its reservoir. Frames identify completed transitions (decision
1000 means 1000 transitions completed, before transition 1001).

Each root enumerates all advisory-legal first actions (≤6), records illegal
actions without executing them, and follows unmodified B8 for exactly H500
steps or terminal hero death. Opponents react on each branch using cloned
policy/carry/RNG state; no recorded future opponent actions. Dead suffix mass is
zero. Record all branch outcomes; rank descending delta_A500, exact ties to bare,
then ascending action index. Bare is evaluated even if advisory fallback is used.
Only roots with a surviving bare H500 branch can qualify. Do not sum local gains
into an H5000 efficacy claim. The unit counted for the decision is the world.

## Provenance and attribution fixed before measurement

Observer-only side state tags successful food additions as ambient, death_drop,
or boost_pellet, preserving deduplication and native corpse FIFO eviction. The
original simulator methods execute exactly once. Each branch inherits the root's
food-source map; its ledger resets intake counters and initial stored length at
the root. Check food/corpse-set identity and native consumption counters every
step, and initial stored length + intake − boost loss = stored length, including
fatal transitions. Missing or unexplained sources invalidate the run.

For surviving paired branches, decompose the mass difference exactly by the
integral of cumulative food intake by source minus cumulative boost costs.
`death_delta_A500` is the alternative-minus-bare integral of cumulative death-drop
intake /500. Operationally, "death-drop-attributed" means
`death_delta_A500 >= delta_A500 >= 5` (integer numerator comparison, no floating
threshold rounding): additional death food accounts for the whole net positive
gain; ambient or boost savings alone cannot qualify. Report all components and
gross counts, including offsetting ambient/trail/boost effects. Both branches
must survive; rescue branches never qualify.

Parity: three additional excluded worlds, one/mix, instrumented versus plain
full terminal H5000 continuations, compare actions and exact physical/carry/RNG
state digests every step. Also test changed-action reacting opponents, clone
isolation, FIFO overflow, deduplication, boosts, death/respawn and contested food.
These are bounded same-platform observer checks, not Mac/x86 or serving proof.

## Frozen KILL/GO rule (verbatim from the judgment)

INVALID/INCOMPLETE first (parity, provenance, accounting, missing worlds,
cap). **KILL this controller family** if < 8/48 worlds (or < 4 frozen+mixed) contain a surviving branch with
`delta_A500 ≥ 5` whose gain is death-drop-attributed. Otherwise GO to O2. A KILL closes single-action H500
harvesting, not "offense"; no extension, no second look.

Analysis runs only after all 48 world payloads, three parity receipts and timing
admission are complete and hash-verified. Operational telemetry contains counts,
time, errors and hashes only. Invalid identity/parity/provenance/accounting →
INVALID; missing work or cap expiry → INCOMPLETE; never turn these into KILL.
A create-only one-look marker is written before analysis; no partial decision.
O1 GO authorizes only preparation of separately registered O2. O2 must compare
D+O against rescue-only D, not only B8. No serving/release changes.

## Caps and timing admission

ONE L4 via qualified `research/colab_cli` supervisor only, CPU rowwise fp32
policies, deterministic process workers (one world/process, ordered merge),
≤12 workers and one thread each, min 12 vCPUs. Account-wide assignment cap three
including notebooks; actual endpoint inventory/rates supplied by the operator.
No direct Colab/Drive calls, allocation/job retries, purchases or RunPod.

Total ≤8 units /4 runtime hours including setup, input delivery, fixture, parity,
census and export; stop at first cap. Reserve 20% INSIDE this envelope (1.6 u,
48 min), ordinary admission/runner allowance ≤6.4 u/192 min. A disjoint L4 timing
fixture uses two worlds, two fixed roots/world (1000/3000), full H500, ≤1 u/30 min
including its work. Before study-bank admission, use measured whole fixture and
branch/base component costs to forecast worst-case full48 + three parity worlds
+ setup/export, conservatively across the fixed workers. Require it to fit the
REMAINING ordinary cap; otherwise INCOMPLETE without study gameplay. Runtime
supervisor enforces the outer wall/unit/lease limits. Estimation uses 2 u/hour as
a price ceiling; reprice/refuse if the provider's reviewed rate is higher.
Progress at least every two minutes, create-only output, bounded immutable export
≤64 MiB, lease600s, full source/checkpoint/config/input SHA verification.
Branch-work upper bound 48×8×6×500=1,152,000 row-steps plus ≤240,000 base steps;
parity and fixtures are charged separately. No exact training-resume claim.

The owner specifically requests one mandatory excluded Mac entrypoint preflight,
which takes precedence over the judgment's "Mac: none": two preallocated excluded
worlds, 40 frames, roots10/20, H8, nice10, one numerical thread, CPU, no Path B
calendar/timing claims. Save command, platform, actual niceness/thread settings,
wall and phase timing plus explicit extrapolated (not measured) L4 forecast under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/`.
Only targeted tests at nice10/one thread. No study gameplay or Colab calls during
implementation. Freeze registration and source before this excluded preflight.
