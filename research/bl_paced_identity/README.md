# Round-07 X7 paced identity

Build-only implementation of advisor 03's three-oracle package, adopted by
round-07 JUDGMENT §b03/X7, with round-06 §a7 delivery semantics and round-07 §c4/5
corrections. This package does not change serving, run a paced study, allocate a
bank, invoke Colab, or ratify a registration. Serving remains on this Mac.

The preflight demonstrates the plumbing on injected fixtures. Production
semantic qualification, quiet-window recorder admission and fresh paced evidence
remain prerequisites for L3/L4 claims. A fixture PASS cannot produce an overall
three-oracle PASS.

## Files and integration

| File | Responsibility |
| --- | --- |
| `events.py` | Canonical finite JSON, immutable request/config snapshots and integer-nanosecond events |
| `record.py` | Bounded on-tick ring, explicit off-tick draining, create-only chained chunks/seal and verified imports |
| `controller.py` | Request admission, exact target/identity/deadline fences, validation/consumption, bare fallback and intervention invalidation |
| `transport.py` | Mac-only spawned worker processes, one math thread each, bounded EDF queue and W/K accounting |
| `replay.py` | Separately implemented reducer; independent request/dispatch expectations, validation, game actions, state and score replay |
| `semantic.py` | Six first-action branches, greedy continuations, exact state/observation/action/event/mass comparison and full-horizon clone isolation |
| `semantic_fixture.py` | Tiny GridBatchSim-versus-BatchSim fixture with reacting opponents; partial fixture state schema, no candidate qualification |
| `receipt.py` | Immutable joined evidence, fail-closed admission, fresh-paced inventory and final-family freeze policy |
| `fixtures.py` / `preflight.py` | Synthetic interleavings and one spawned-worker/eight-tick Mac preflight |

The caller provides an immutable root and a separately pinned planner worker
`worker(request_dict, input_payload) -> result_dict`. Result JSON contains the
entire request binding, `action`, `bare`, `mask`, and all planner outputs required
by its frozen semantic oracle. A request binds session, life, source/target
frame, checkpoint SHA, shadow epoch, root digest, request ID, input-payload SHA
and absolute cutoff. Root selection, two-tick cutoff margin, the 32/96/192/300
horizon checkpoints and a lead cap of 432 decisions belong to the frozen
candidate adapter. The package preserves those declarations; it does not invent
the future candidate's roots or claim the T3 planner has been qualified.

Before a tick, call `collect()` to observe ready IPC; it never waits. `submit()`
admits at most K roots. Off tick, `dispatch()` starts the earliest cutoff, then
target, then request ID among eligible queued roots, at most W workers. Process
creation, joining and journal draining are explicit off-tick operations. Record
worker progress reads and clock observations before using them for root selection.
The initial journal state must include the entire restorable game, opponent
carry, RNG, source/fixture identities and frozen authorization, not just a seed.

At a tick, `poll(context, bare, mask, validate)` consumes only a visible response
for that exact context. Completion time, coordinator visibility and consumption
are distinct events. Validation must finish and consumption must occur strictly
before the request's own cutoff; equality is a miss. A result cannot borrow a
later root's deadline. Every rejected result uses the current bare action.
Interventions advance the shadow epoch, fencing other answers. Respawn,
checkpoint changes and host pause/resume call `invalidate()` with an advancing
epoch. Cancelling unfinished work keeps its K slot occupied until collection.

W defaults to 1 and K to 1. All integer configurations 1≤W≤4 and W≤K≤8 are
representable and covered by model fixtures. **Actual multi-worker launch
requires `owner_multi_worker=True` only after the owner's explicit OK.** This
flag is not authorization itself; save the authorization in the prospective
manifest. The preflight launches only W=1/K=1. K includes queued, running and
completed-but-unconsumed work. No latency claim follows from a contended Phase-0
fixture.

## Oracle contracts

1. **Semantic:** supply 18 declared mix/length fixtures (three mixes × six length
   strata) and 96 preselected unique pinned roots including ordinary/decisive,
   near-tie, long-body and frozen-opponent-carry cases. Each root compares all
   six first actions followed by independent greedy continuation through H300.
   Fatal transitions do not count as alive transitions. Rank by alive
   transitions, bare/B8 preference, root Q, then index. Mass never ranks branches.
   Compare canonical observation bytes, ordered food/bodies, RNG/carry, hero and
   opponent actions, terminal events and integer hero mass. Clone isolation is
   checked through every continuation step on both implementations. Production
   runtimes must expose full `t34-shadow-full-digest/v3` semantic state; the tiny
   adapter explicitly exposes only `partial-fixture-state/v1`.

   `SemanticFixture` documents mandatory metadata and source/input pins. The
   reference must be independently frozen and may not import modified planner
   helpers. The independent-reference attestation is auditable evidence; it
   does not mechanically prove source independence. Do not use the score-free
   fixture ranking mode for production; the checker blocks it.

2. **Availability replay:** read only sealed `read_journal()` exports, then use
   `replay_journal(directory, game_factory)`. The reducer does not import the
   controller or planner. External clocks, visibility and progress are supplied
   in original coordinator sequence and integer time order. Replay recomputes
   admission/refusal, EDF dispatch, K/W, validation, rejection/fallback and
   executed actions. It never drives the game with recorded decisions.

   The independent `ReplayGame` adapter implements `reset`, `context`, `bare`,
   `mask`, `step`, `observe`, `submission`, `validate`, and
   `finish_expectations`. `observe` receives only exogenous observations, never
   recorded submissions or executed actions. `finish_expectations` supplies the
   complete request and dispatch sequences generated from the independent
   scheduler's clock/progress/game opportunities, including requests absent
   from the recorded trace. This detects whole omitted submission/dispatch
   blocks. Each `game_state` is regenerated, including state digest and integer
   terminal-hero `mass_sum`, `alive_frames`, and `deaths`. Missing adapter
   capabilities are INCOMPLETE. A scheduler-only pass is also INCOMPLETE.

   Production replay initial state carries nonempty `game_state`,
   `opponent_state`, `rng_state`, and `source_snapshot`, with
   `state_digest_schema="t34-shadow-full-digest/v3"` and `synthetic=false`.
   Its `replay_contract` binds `independent_scheduler=true` and a valid frozen
   `replay_reference_sha256`. Every replayed game-state record uses the same
   exact v3 schema. Missing this contract yields PRECHECK_ONLY. Pass the
   prospective manifest's binding hash to `Journal(binding_sha256=...)`.

3. **Fresh paced:** every scored attempt must replay its own full trace before
   aggregate efficacy is unblinded. L3 retains 16 paired worlds per mix and six
   paired ten-minute Watch6 sessions, a frozen chronological order/load profile,
   all misses/fallbacks/errors, and at least one naturally exercised
   intervention. No best-of-N, discarded worlds, replay-score substitution, or
   synthetic DELIVERY_READY. Timing/efficacy STOP rules remain the future
   registration's responsibility. Planner-over-net additionally requires all
   four E13 receipts, not just these three oracles.

The bounded ring performs no file I/O in `emit()`. Its canonical serialization
cost is part of the controller and must be measured with recording enabled in
the quiet-window admission. Overflow, malformed events, missing chunks, broken
chains or a missing seal cannot be silently dropped. All JSON exports have
`.complete` and `.sha256` sidecars and are create-only; use a fresh directory
for each attempt. The first mismatch suspends the whole evaluation and preserves
request/frame/event, both compared values and action/state/score prefixes.
Missing evidence blocks admission. A verifier-only repair needs independent
audit and prospective permission to recover unchanged raw evidence; an
action-affecting repair creates a new artifact and requires fresh banks.

## Final family and forecasts

There is one final α=.05 family, frozen before opening the first strict bank.
Allowed slots are bare R, assisted B8, and a reserved planner-over-R slot. Only
slots with a frozen nominee at that moment enter the family; an absent reserved
slot cannot be added later. Three nominees use Holm .05/3, .05/2, .05; two use
.025, .05. Composite p is the maximum required-clause p; bare wins unresolved
ties. `freeze_family` exports this identity; it does not open a bank or make a
release decision. The X6 campaign ledger remains a separate authority.

Engineering estimates from the adopted spec: 24 builder-hours + 8 independent
verification-hours; reserve 2 unpaced Mac-hours for production semantic fixtures
and 8 for replay before admitting paced windows. Those reservations are not
measured forecasts and have not been consumed by this build. L3≤9 and L4≤32
paced Mac-hours require their own prospective authorization/readiness. No study
manifest is dispatchable here; Colab units = 0.

At 24 fps, 640 full H5000 **candidate** episodes would occupy 37.037 paced hours
before overhead if every episode advances the whole horizon. Terminal-hero
padding can reduce elapsed time. The bare comparator may be unpaced only under
the applicable registered amendment. 640 is an illustration from the E13 bank,
not a sized strict design; actual strict N, bare-comparator time and production
replay/semantic forecasts remain unknown until measured prospectively.

## Bounded Mac preflight

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-bl-x7
nice -n 10 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.bl_paced_identity.preflight \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-x7/NEW-preflight
```

Use a nonexistent absolute output path. This creates the manifest, receipt,
semantic/replay exports and complete per-fixture journals. It runs no paced
worlds, loads no served checkpoint and opens no sealed study. The fixture
checkpoint SHA is deliberately synthetic. The retained preflight records source
hashes, libraries, load declaration, event/chunk counts and elapsed functional
time; it leaves the paced oracle NOT_RUN and three-oracle admission BLOCKED.
