# Tier-1 screen: Apex + v8 (space + head, lambda = 8) vs Apex + v7 (lambda = 4) (pre-registration)

Label: **screen (Tier-1, non-authoritative)**. Governance: Tier 1 in
`docs/research/governance_tiers_2026-09-26.md`. Nothing here can change a default, the
champion, a deployment profile or the Watch-hero veto. Written 2026-10-03 (UTC), before any
episode of this screen ran. The only episodes allowed before GO are plumbing smokes on the smoke
namespace `apex-veto-v8-screen-smoke-v1` (at most 2 episodes x 500 frames each).

**Amendment 1 (2026-10-03, before any real episode; only the one plumbing smoke had run).**
After an independent review of this package: a real run now needs a fully clean tree
(untracked files too) with `screen.py` and this file committed, records both files' sha256
in every intent and re-checks them at merge; `--out` and the slot lock root are pinned; the
thermal backoff, max backoffs and episode budget are fixed; a 3-shard start barrier replaces
the open-ended slot wait; a shard that crashes still writes its summary; `merge` checks
records against plans, recomputes the seeds, and checks pairing. Design, metric and decision
rule are unchanged.

- `screen_id`: `apex-veto-v8-screen-v1` (the namespace reserved for it in the v8 DEV sweep's
  protocol, `research/apex_veto_v8_lambda_sweep_20261002/protocol.md`)
- Harness: `screen.py` beside this file. It reuses `research/apex_safety_20260926/dev_screen.py`
  (seed recipe, disjointness and roster-parity preflight, strict balanced rosters, checkpoint
  snapshots, `run_episode`, `acquire_cpu_slots(..., pool=3)`, `make_thermal_guard`, AC check)
  and the v8 sweep's installers, episode loop (`sweep._play`: deadline budget, then the
  thermal-guard admission, then one episode) and guard tallies, unchanged. No existing file
  changes.
- Owner: Apex safety lane.

## Question and the decision it informs

Does `free-space-veto/v8-space-and-head(lambda=8.0)` (`src/evaluation/safety_veto_v8.py`)
raise the Apex champion's H5000 mass integral over the incumbent
`free-space-veto/v7-space-preference(lambda=4.0)` (v7(4) has a Tier-2 STRICT_PASS vs v5,
`apex-veto-v7-strict-20261002/run-v1`), without a significant loss in any opponent mix?
Decision informed: design a Tier-2 strict gate for v8(8) against v7(4) (`ADVANCE`) or stop
(`NOT_ADVANCED`). Two sub-questions are named in advance because the DEV sweep raised them:
is v8 worse than v7 in the frozen mix (sweep: -14.1, 0 wins / 2 losses of 8), and is the
scripted gain real (sweep: +137.1, 4 / 0)?

### Disclosed: where lambda = 8 came from

`lambda` is not a free parameter here. The pre-declared DEV sweep `apex-veto-v8-dev-v1`
(8 worlds per mix, arms v7(4) and v8 at 4 / 8 / 16) returned `SELECTED`, `lambda = 8.0`
(per-mix mean delta vs v7(4): frozen -14.1, mixed +19.5, scripted +137.1; pooled +47.5,
SD 123.5 over 24 pairs). `screen.py` binds to that summary by sha256 (`2a3eb8b3...4720b2`,
`snake-dqn-artifacts/apex-veto-v8-lambda-sweep-20261002/run-v1/merged/summary.json`) and
refuses a real run unless the file still has those bytes, is not a smoke, and says
`SELECTED` / `passes` / `selected_lambda = 8.0`. The sweep's deltas are biased upward by
the selection and are not evidence here; its worlds are excluded.

## Arms (paired: A and B play every world)

| Arm | Hero | Veto |
|---|---|---|
| A | `champion_a5_freespace_20260621.pth` (sha256 `43d4e2c5...d747ac93`) | v7, `lambda = 4.0` (`safety_veto_v7.py`), the incumbent; installed by the v8 sweep's `install_a` |
| B | same | v8, `lambda = 8.0`, head layer on, with the diagnostic-only `reference_lambda = 4.0` (counts decisions where B differs from A's rule; never decides); installed by the sweep's `install_v8` (the sweep's H800 installer object) |
| C | same as A | determinism control: A repeated on the first 2 worlds per mix, in another shard process |
| D | same as B | replay control: B repeated on the first 2 worlds per mix, in another shard process |

Vetoes are installed through `dev_screen.hero_veto_installer` after rollout's built-in
install (whose vector61 guard still runs).

## Worlds

- Profile `promotion-v2-watch-rect`, horizon 5000, digest `d396d3ed...0e8b`; config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3...715aa5`).
- Mixes `frozen`, `scripted`, `mixed`. **60 worlds per mix** (180 world-mix pairs), the same
  seeds in every mix: `uint32(sha256("apex-veto-v8-screen-v1|worlds|<i>")[:4])`, i = 0..59.
- Construction-time disjointness (fail closed, in code, before any slot, output or episode)
  against: (1) the first 1000 seeds of every domain/purpose the v8 DEV sweep excluded
  (all `apex-safety`, `apex-veto-strict-*`, web-serving watch/play/parity/worlds, the v3-v7
  screens and smokes, v5 strict and serving, trap-horizon v2/v5, the v7 DEV sweep, the v7
  strict banks `apex-veto-v7-strict-{dev,final,serving,smoke}-v1`), the v8 DEV sweep and its
  smoke (`apex-veto-v8-dev-v1`, `apex-veto-v8-dev-smoke-v1`: burned), this screen's other
  domain, and the v7 web serving lane `apex-veto-v7-web-serving-v1` and its smoke (purposes
  watch/play/parity/worlds); (2) **observed** seeds read from the saved artifacts (every
  `world_seed` / `seeds` / `paired_seeds` integer): the v7 strict run's `rosters.json`
  (calibration, final and serving banks), the three v8 sweep shard intents and its merged
  summary, the v7 sweep and v7 screen intents, and the v7 web serving run's intent. A
  missing source file is a refusal. dev_screen adds the task-aligned challenger namespaces,
  the strict pilot's observed seeds and seeds 0..999. The strict pilot recipe and roster
  parity checks must run and pass.

## Shards (three processes, one CPU slot each)

The 180 world-mix rows, in dev_screen's (mix-major) order j = 0..179, go to shard `j % 3`
(60 rows each, 20 per mix). A shard plays its rows ordered by (world index, mix), A then B
per row, then its controls: C and D for a mix's first 2 rows (j0, j0 + 1) run in shard
`(j + 1) % 3`, never the shard of the episode they repeat (a cross-process determinism
check). Shard 0 plays 120 episodes, shards 1 and 2 play 126 each; 372 in total.

Start barrier (the v8 sweep's `start_barrier`, in code): after taking its slot
(`--slot-timeout-seconds`, default 180) and before `--out` exists, each real shard writes
`<run root>/.barrier/shard-K.json` and waits at most 120 s until all three markers are
present, fresh, from live processes, with the same commit, protocol sha256, `screen.py`
sha256 and shard count, and hold the three distinct pool-3 slots; otherwise it exits 2
before any output or episode. So the operator launches only after the sweep's `slots-free`
check passes (no other holder, for example the v7 web serving qualification, on any pool-3
slot); the three shards then run together.

`merge` refuses (exit 2) shard sets that are not exactly shards 0, 1, 2 of 3 of this screen,
are smokes, differ in protocol bytes, worlds, mixes or commit, whose seeds are not the
recomputed `apex-veto-v8-screen-v1` seeds, whose recorded `screen.py`/`protocol.md` sha256
differ from the merging package's, whose slot lock root, operating parameters or `--out`
directory are not the pinned ones, without a passed start barrier, whose plan differs from
this assignment, or whose record keys are not a prefix of the shard's plan; and it writes
only to `<run root>/merged`. It records the merging commit (`merge_source`).

## Primary metric, estimator and decision rule (computed by `screen.summarize` in `merge`)

Primary: per world and mix the paired delta `d = mass_integral(B) - mass_integral(A)`.
Per mix m (n = 60): mean `dbar_m`, SD `s_m`, one-sided 90% bounds
`dbar_m -/+ t(0.90, n - 1) * s_m / sqrt(n)` (`t` from `src.scripts.eval_stats.student_t_isf`).
Pooled: the equal-weight mean of the three mix means (= the mean of all 180 deltas, since
n is equal) with `SE^2 = (1/9) * sum_m s_m^2 / n` and Welch-Satterthwaite df; one-sided 90%
lower bound `LB = mean - t(0.90, df) * SE` (equal to the lower end of
`screen_stats.stratified_mean_of_means` at two-sided 80%; tested). If every SE term is 0 the
bound is the mean itself.

Decision, in order:

1. `SMOKE_NO_DECISION` for a smoke (a smoke is never merged).
2. `INVALID_SELF_CHECK_FAILED` if any record fails a gating self-check (below), the arms of
   one (mix, seed) differ in roster or `world_identity`, or a shard's checkpoint snapshot
   changed.
3. `INCOMPLETE` if any planned episode is missing (for example a deadline, a thermal stop or
   a crash: a crashing shard still writes `shard_summary.json` with `stopped_reason =
   "error: ..."` and exits 4).
4. `INVALID_NONDETERMINISTIC` if any C record differs from its A record or any D record from
   its B record (canonical JSON).
5. **`ADVANCE`** iff the pooled mean delta is > 0 **and** its one-sided 90% lower bound is
   > 0 **and** no mix shows a significant loss, where a mix's loss is a one-sided 90% upper
   bound < 0.
6. `NOT_ADVANCED` otherwise.

Reported only (never gating): per-mix and pooled means, SDs, both bounds, wins / losses /
ties; the crossed mixes x worlds and random-effects pools (dev_screen's informational
`pooled_summaries`, which model the shared-seed covariance that the decision's stratified SE
ignores); survival fraction per arm and the paired survival delta; hero death causes per arm
and mix, with head-on deaths (`death_cause == "head_on"`) counted separately; B's v8
counters (head-risky decisions and vetoes, `action_differs_from_reference` vs A's rule,
replacement and cost counters) and A's v7 counters; episode wall times and thermal-guard
events. Say "harm" or "no effect" only when a bound excludes the effect of interest;
otherwise "not established".

### Size and power (stated before any episode)

60 worlds per mix. The only variance estimates are the DEV sweep's (8 worlds, mostly exact
ties, so unreliable): SD of the lambda = 8 deltas 30.9 / 57.9 / 177.1 (frozen / mixed /
scripted). With those, a frozen SE is about 4.0, so the frozen mix is flagged as a loss when
its mean is below about -5.2: if the sweep's -14.1 were the truth the screen would call it
a loss with probability about 0.99 (about 0.76 at a true -8). The pooled SE is about 8.1,
so the lower bound is positive when the pooled mean exceeds about 10.5: power about 0.88 at
a true pooled +20 (the sweep's +47.5 shrunk by more than half for selection) and near 1 at
+47.5. The no-loss condition is conservative by design: a mix can block `ADVANCE` with a
small true loss. 60 rather than 40 worlds because the frozen question needs it and three
slots make it affordable.

## Self-checks (gating; `screen.check_entry`)

On every record: `probes.safety_veto` equals the arm's descriptor (A/C: v7(4)'s; B/D:
v8(8)'s, head layer on) plus exactly the seven v2 counters
(`strict_promotion._validate_candidate_wrapper_probe`); `record.seed` is the entry's world
seed, the seed is a planned seed and no (arm, mix, seed) appears twice; `mass_integral` is
finite; outside a smoke, `evaluation_profile_digest` is the profile's, `world_identity`
matches the roster, `denominators.scored_frames = 5000` and `counters.decisions =
denominators.decision_frames`; entry schema, screen id, hero sha256, `safety_veto: true` and
the arm's method; on B/D the reference diagnostic is present with `reference_lambda = 4.0`
and `reference_decisions = counters.decisions`. Reported only: the v8 counter identities
(`sweep.v8_identities_hold`) and presence of A/C's v7 counters.

## Compute cap and operations

- 372 episodes, three shard processes. Expected about 70-80 min per shard when all three
  run together (the v8 sweep measured about 35 s per episode with three concurrent shards).
  A shard that waits for a slot starts later.
- Cap: 4 h (14400 s) per shard. `screen.py run` refuses a `--deadline-utc` more than 4 h
  after launch, or less than 5733 s (126 episodes x 35 s x 1.3) after it, and stops at the deadline (an episode starts only with max(45 s, 2 x mean
  episode time) left); a stop gives `INCOMPLETE`.
- Required flags: `screen.py run` refuses (exit 2, before any output, lock or episode)
  unless given `--use-slot-locks --slot-pool 3 --thermal-guard`; AC power is always
  required (refuses to start on battery; the guard re-checks AC before every episode).
  `--require-ac-power` is accepted for symmetry. Outside a smoke, `--thermal-backoff-seconds`
  (60), `--thermal-max-backoffs` (5) and `--min-episode-budget-seconds` (45) must be the
  defaults, `--slot-lock-root` must be the default root, and `--out` must be
  `<run root>/shard-K` with `<run root>` = `snake-dqn-artifacts/apex-veto-v8-screen-20261002/run-v1`
  or its `recovery-1/`.
- Size: outside a smoke it refuses any `--worlds-per-mix` other than 60 and any `--shards`
  other than 3. A smoke is one shard (`--shard 0 --shards 1`) playing A and B on one world
  of one mix (2 episodes, <= 500 frames, legacy path, smoke namespace), no controls.
- Clean tree: outside a smoke it refuses unless `git status --porcelain
  --untracked-files=all` is empty and `screen.py` and `protocol.md` are tracked.
- Slot lock: each shard takes ONE slot from pool 3 (`cpu-slot-3.lock`, then
  `cpu-slot-1.lock`, then `cpu-slot-2.lock` under
  `snake-dqn-artifacts/pqn-followup-20260909`, opened read-only, never created), taken
  before `--out` exists and held until `shard_summary.json` is written. Locks are never
  bypassed; with the start barrier, a launch while another process holds a pool-3 slot
  plays nothing (relaunch after `slots-free` passes).
- Thermal guard (compute policy, 3-slot calibration PASS in the v8 DEV sweep): before each
  episode, `admit_next_episode` with dev_screen's defaults (pmset thermal/performance
  warning levels 0, CPU speed/scheduler limits not below 100, AC power; slowdown baseline 12
  / window 8 / 60 %; 60 s backoff, at most 5 in a row; a persistent slowdown or a
  still-not-ok guard stops the shard with a `thermal: ...` reason). As in the sweep (whose
  calibration measured this setting: 0 pauses, largest in-run ratio 1.14), the slowdown key
  is the whole shard (`shard`). A thermal stop is never relabelled: the decision sees the
  missing episodes (`INCOMPLETE`).
- Output: real runs write only under
  `snake-dqn-artifacts/apex-veto-v8-screen-20261002/run-v1/{shard-0,shard-1,shard-2,merged}`;
  smokes stay outside `snake-dqn-artifacts`.
- Test safety: every unit test of this package replaces `tournament_eval.rollout` and
  `dev_screen.run_episode` with functions that raise, and uses temporary slot lock roots.
- Recovery: one mechanical-defect recovery under `recovery-1/` with the same protocol, as
  governance Tier 1 allows. Anything else needs a new namespace.

## Operator commands

```bash
cd <this worktree>   # the commit that holds this file; tree clean for the real run
R=/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909
A=/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v8-screen-20261002/run-v1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
pmset -g batt | head -1          # must say 'AC Power'
# 1. One plumbing smoke (2 episodes x 500 frames, smoke namespace, outside artifacts).
SNAKE_DQN_DEVICE=cpu $PY research/apex_veto_v8_screen_20261002/screen.py run \
  --shard 0 --shards 1 --out <scratch>/v8-screen-smoke-$(date +%s) \
  --deadline-utc $(date -u -v+20M +%Y-%m-%dT%H:%M:%S+00:00) \
  --smoke-frames 500 --worlds-per-mix 1 \
  --use-slot-locks --slot-pool 3 --thermal-guard --require-ac-power
# 2. The real screen (in tmux): abort unless all 3 pool-3 slots are free now, then three
#    shards at once (start barrier), then merge.
$PY research/apex_veto_v8_lambda_sweep_20261002/sweep.py slots-free --slot-lock-root $R || exit 1
mkdir -p $(dirname $A)
D=$(date -u -v+4H -v-5M +%Y-%m-%dT%H:%M:%S+00:00)
for K in 0 1 2; do
  SNAKE_DQN_DEVICE=cpu caffeinate -dimsu $PY research/apex_veto_v8_screen_20261002/screen.py run \
    --shard $K --out $A/shard-$K --deadline-utc $D \
    --use-slot-locks --slot-pool 3 --thermal-guard --require-ac-power > $A-shard-$K.log 2>&1 &
done; wait
$PY research/apex_veto_v8_screen_20261002/screen.py merge \
  --shard-dirs $A/shard-0 $A/shard-1 $A/shard-2 --out $A/merged
```

## Non-claims

- `ADVANCE` is not promotion: v8 would still need a Tier-2 strict gate against v7(4) on
  fresh worlds with its own source binding, plus a serving qualification (and a SIMD port or
  a live-engine serving path).
- `NOT_ADVANCED` does not show v8 is worse; it says this rule at this size did not find a
  pooled gain free of a significant loss in some mix.
- At `lambda = 8` the head layer and the stronger space preference are confounded (B differs
  from A in both); this screen cannot attribute a difference to either.
- The released configuration is unaffected: no veto module, dev_screen, sweep or strict
  package changes.
