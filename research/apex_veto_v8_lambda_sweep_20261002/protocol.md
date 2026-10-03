# Tier-1 DEV lambda sweep: v8 (space + head) vs v7(lambda=4), and the 3-slot calibration (pre-declaration)

Label: **development sweep (Tier-1 dev, non-authoritative)**. Governance: Tier 1 in
`docs/research/governance_tiers_2026-09-26.md`. It cannot change a default, the champion, a
deployment profile or the released Watch-hero veto, and it reaches no screen decision. It has
two outputs, both pre-declared here: (1) the choice of ONE `lambda` for a future, separately
pre-registered v8 Tier-1 screen (not built in this package), or a stop; (2) the verdict of the
3-slot thermal calibration in `docs/research/compute_policy_2026-10-02.md`. Written
2026-10-02, before any episode of this sweep ran. The only episodes allowed before GO are one
plumbing smoke on the smoke namespace `apex-veto-v8-dev-smoke-v1` (at most 2 episodes x 500
frames), run by the operator only after the v7 strict run (`apex-veto-v7-strict-20261002`)
has closed.

- `sweep_id`: `apex-veto-v8-dev-v1`
- Harness: `sweep.py` beside this file. It reuses `research/apex_safety_20260926/dev_screen.py`
  unchanged (seed recipe, disjointness and roster-parity preflight, strict balanced rosters,
  checkpoint snapshots, `run_episode` with a `ScreenSpec` mapping every arm to its veto
  installer, `acquire_cpu_slots(..., pool=3)`, `make_thermal_guard`, AC check) and the
  `research/compute/thermal_guard.py` admission gate (`admit_next_episode`), plus the v7
  sweep's statistics helpers. No existing file changes.
- Output per shard: `intent.json` (before any episode: git commit and dirty paths, this
  file's sha256, the arms, the shard plan, the selection and calibration rules, the slot held,
  the guard configuration), `records/`, `events.jsonl`, `shard_summary.json`. `merge` writes
  one `summary.json` from the three shard directories.

## Question

v7 (`free-space-veto/v7-space-preference(lambda=...)`) re-ranks v5's eligible moves by
`Qn + lambda * g(area)`. Its dev sweep (`apex-veto-v7-dev-v1`) gave pooled mean deltas vs v5
of +13 / +104 / +117 at `lambda` 1 / 2 / 4, still rising at the top of the grid, and its
Tier-1 screen at `lambda = 4` gave +135 to +154 vs v5; v7(4) is now under the strict gate.
Separately, v6's layer (a) (opponent-head avoidance, `src/evaluation/safety_veto_v6.py`) cut
head-on deaths 10 -> 2 and gave +30 mass in the scripted mix vs v5
(`docs/experiments/apex_veto_v6_2026-10-02`). Two questions, one sweep:

1. Does adding v6's head layer on top of v7(4) help (arm H400 vs A)?
2. Does a larger `lambda` (8, 16), with the head layer, help further (H800, H1600 vs A)?

v8 (`src/evaluation/safety_veto_v8.py`, sha256 `faf3695f...4ac05`) is v7's decision at
`lambda` (an unchanged `SpacePreferenceVeto`), then v6's layer (a) on v7's choice: if that
choice is head-risky (a cell an unbeaten opponent head can reach next frame, v6's reach and
1.15x rule) and a masked-legal, v5-eligible, non-risky alternative exists, the choice is
replaced by v7's own re-rank over those alternatives (same speed mode first; `Qn` normalized
over them; ties to the highest-Q one). With the head layer off v8 is v7 exactly (tested), so
H400 differs from A only on head vetoes.

## Arms (paired: every arm plays every world)

| Arm | Hero | Veto |
|---|---|---|
| A | `champion_a5_freespace_20260621.pth` (sha256 `43d4e2c5...d747ac93`) | v7, `lambda = 4.0` (`safety_veto_v7.py` sha256 `56ff7009...e2980`), the strict-gated candidate |
| H400 | same | v8, `lambda = 4.0` (= A + head layer) |
| H800 | same | v8, `lambda = 8.0` |
| H1600 | same | v8, `lambda = 16.0` |
| R | same | replay control: H800 repeated on the first world of each mix, in a different shard process from its H800 episode. Each R record must equal its H800 record exactly. |

The v8 arms are installed with `reference_lambda = 4.0`: each v8 decision also runs a
separate v7(4) on the same inputs and counts `action_differs_from_reference` (decisions where
the arm differs from arm A's rule). It never decides, is not part of the probe descriptor, and
costs one more v7 decision per call. Vetoes are installed through
`dev_screen.hero_veto_installer` after rollout's built-in install (whose vector61 guard still
runs). Opponent pool and roster construction are dev_screen's (strict balanced rosters).

## Worlds

- Profile `promotion-v2-watch-rect`, horizon 5000, digest `d396d3ed...0e8b`. Config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3...715aa5`).
- Mixes `frozen`, `scripted`, `mixed`. **8 worlds per mix** (24 world-mix pairs), the same
  seeds in every mix: `uint32(sha256("apex-veto-v8-dev-v1|worlds|<i>")[:4])`, i = 0..7.
- Construction-time disjointness (fail closed, in code) against the first 1000 seeds of
  every earlier domain: everything the v7 screen excluded (all `apex-safety`,
  `apex-veto-strict-*`, web-serving with purposes watch/play/parity/worlds, v3-v6 screens,
  v5 strict and serving, trap-horizon v2/v5, the v7 dev sweep and the v7 screen, each with
  its smoke), the v7 strict banks (`apex-veto-v7-strict-{dev,final,serving,smoke}-v1`), this
  sweep's smoke domain, and the future v8 screen domains `apex-veto-v8-screen-v1` and
  `apex-veto-v8-screen-smoke-v1`. dev_screen adds the task-aligned challenger namespaces, the
  strict pilot's observed seeds and seeds 0..999. The strict pilot recipe and roster parity
  checks must run and pass (they did in a no-episode preflight while writing this file).

## Shards (the run is three concurrent processes)

The 24 world-mix rows, in dev_screen's (mix-major) order j = 0..23, go to shard `j % 3`
(8 rows each). A shard plays its rows ordered by (world index, mix), the four arms per row in
the order A, H400, H800, H1600, then its R episode: R for a mix replays that mix's first world
in shard `(j0 + 1) % 3`, so every replay runs in a different process from its H800 episode
(a cross-process determinism check). 33 episodes per shard, 99 in total. `merge` refuses
shard directories whose recorded plan differs from this assignment, that come from different
commits or protocol bytes, or that are not exactly shards 0, 1 and 2 of 3.

## Selection rule (pre-declared; computed by `sweep.summarize` in `merge`)

For each `lambda` in {4, 8, 16} (arms H400, H800, H1600): per world and mix, the paired delta
`mass_integral(v8_lambda) - mass_integral(A)`; per mix the mean and SD over its 8 worlds;
pooled the mean over all 24 pairs. Activity is read from each v8 episode's
`veto_diagnostics` (`action_differs_from_reference`, `decisions`).

1. `SMOKE_NO_SELECTION` for a smoke (a smoke is never merged).
2. `INVALID_SELF_CHECK_FAILED` if any record fails a gating self-check (below).
3. `INCOMPLETE` if any planned episode is missing (for example a deadline or thermal stop).
4. `INVALID_NONDETERMINISTIC` if any R record differs from its H800 record.
5. A `lambda` is **active** iff in EACH mix at least **1** of its episodes has
   `action_differs_from_reference > 0` and at least 1 world has a nonzero paired delta, and
   its pooled `action_differs_from_reference / decisions` is at least `1e-4`.
   These are v7's three conditions with the per-mix episode minimum lowered from 2 to 1, for
   a disclosed reason: H400 differs from A only through head vetoes, which are rare. In the
   v6 screen, layer (a) vetoed 109 times in 372,591 decisions (2.9e-4, above the rate floor)
   and in 35 of 120 episodes (29%; 13/12/10 of 40 per mix). With 8 episodes per mix and
   p = 0.29, a minimum of 2 fails a mix with probability about 0.28 and all three mixes pass
   with probability only about 0.38, so v7's minimum would usually call a real head layer
   inert; a minimum of 1 fails a mix with probability about 0.065 (all pass about 0.82).
   An inactive `lambda` can never be selected.
6. A `lambda` is a **clear loser** in a mix iff the one-sided 90% upper bound of its mean
   delta there, `mean + t(0.90, n-1) * sd / sqrt(n)` (`t = 1.4149` for 8 worlds), is below 0.
7. A `lambda` **qualifies** iff it is active and a clear loser in no mix.
8. `SELECTED`: the qualifying `lambda` with the highest pooled mean delta (ties: the
   SMALLER `lambda`, the one closer to the strict-gated A). The pooled mean may be negative;
   a later screen decides, not this sweep.
9. `NONE_ACTIVE` if no `lambda` is active, else `NONE_QUALIFIES` if none qualifies: stop.

Only `SELECTED` allows a v8 screen to be written (`summary.selection.passes = true`). The
summary also records `source.commit` and `source.dirty_paths`. Everything else (per-mix means
of each arm, wins and losses, death causes, v8 and nested v7 counters, head veto rates, wall
time) is reported only. No significance test is run.

## Self-checks (gating)

On every record: `probes.safety_veto` equals the arm's descriptor (A: v7(4)'s; H*/R: v8's at
that `lambda`, head layer on) plus exactly the seven v2 counters
(`strict_promotion._validate_candidate_wrapper_probe`); `record.seed` is the planned world
seed and no episode appears twice; `mass_integral` is finite; outside a smoke,
`evaluation_profile_digest` is the profile's, `world_identity` matches the roster,
`denominators.scored_frames = 5000` and `counters.decisions = denominators.decision_frames`;
entry schema, sweep id, hero sha256, `safety_veto: true` and the arm's method; on every v8
arm the reference diagnostic is present with `reference_lambda = 4.0` and
`reference_decisions = counters.decisions` (the activity gate reads it). Reported only: the v8
counter identities against the probe (`sweep.v8_identities_hold`) and arm A's v7 counters.

## Compute cap and operations

- 99 episodes in 3 shards of 33. Expected about 20-35 min per shard: about 30 s per live
  v7(4) episode in the v7 sweep (arm L400: 40.6 / 26.8 / 23.9 s per frozen / mixed /
  scripted episode), v8 adds v6's cheap head check, rare head-veto areas and one reference
  v7 decision per call on its arms, and 3 concurrent processes may slow each other (that is
  what the calibration measures).
- Cap: 3 h (10800 s) per shard. `sweep.py run` refuses a `--deadline-utc` more than 3 h after
  launch and stops at the deadline (an episode starts only with max(45 s, 2 x mean episode
  time) left); a stop gives `INCOMPLETE`.
- Required flags: `sweep.py run` refuses (exit 2, before any output, lock or episode) unless
  given `--use-slot-locks --slot-pool 3 --thermal-guard`. AC power is always required
  (refuses to start on battery; the guard also re-checks AC before every episode).
  `--require-ac-power` is accepted for symmetry with dev_screen.
- Size: outside a smoke it refuses any `--worlds-per-mix` other than 8 and any `--shards`
  other than 3. A smoke is one shard (`--shard 0 --shards 1`) playing arms A and H400 on one
  world of one mix (2 episodes, <= 500 frames, legacy path, smoke namespace), never R.
- Clean tree: outside a smoke it refuses a tree with tracked modifications.
- Slot lock: each shard takes ONE slot from pool 3 (`cpu-slot-3.lock`, then
  `cpu-slot-1.lock`, then `cpu-slot-2.lock` under
  `snake-dqn-artifacts/pqn-followup-20260909`, opened read-only, never created; slot 3 exists
  only after `research/compute/slot_setup.py --create`), taken before `--out` exists and held
  until `shard_summary.json` is written. A shard that cannot get a slot within 180 s refuses.
- Start barrier (non-smoke runs, in code): after taking its slot and before `--out` exists,
  each shard writes a marker `<parent of --out>/.barrier/shard-K.json` (commit, this file's
  sha256, `shards = 3`, slot held, pid, time) and waits at most 120 s
  (`--barrier-timeout-seconds`) until all three markers are present, written within 120 s
  of its own, from live processes, with the same commit, protocol sha256 and shard count,
  and hold the three distinct pool-3 slot files. Otherwise it prints the refusal and exits
  2 before any episode or output. Since the three markers mean the three shards hold every
  pool-3 slot at once, no other holder (in particular the v7 strict run on slots 1 and 2)
  can run beside the sweep, and a partial launch (one shard on slot 3 while slots 1 and 2
  are held, or a sibling that failed to start) plays nothing. The barrier report is in
  `intent.json` (`start_barrier`). A smoke (one shard) has no barrier.
- Thermal guard: before each episode, `admit_next_episode` with dev_screen's defaults (pmset
  thermal/performance warning levels 0, CPU speed/scheduler limits not below 100, AC power;
  slowdown baseline 12 / window 8 / 60 %; 60 s backoff, at most 5 in a row; a persistent
  slowdown or a still-not-ok guard stops the shard with a `thermal: ...` reason). One
  deviation, disclosed: the slowdown key is the whole shard (`shard`), not `arm/mix`. With
  8 episodes per arm and mix in the WHOLE sweep (at most 3 per shard), an `arm/mix` key never
  reaches the 12-episode baseline, so the detector would be inert. Each shard's order
  interleaves mixes and arms per world, but world content still moves the shard key's wall
  times. Measured before any episode (a proxy replay, no thermal cause): the pinned v7
  sweep's single-process wall times, mapped onto this shard plan (4 arms per row) for all 6
  mix orders and run through the real `ThermalGuard` / `admit_next_episode` at 12 / 1.60 /
  8 with a whole-shard key, paused twice in 3 of 18 shard runs (including shard 1 in the
  pre-declared frozen/scripted/mixed order), 0 times in the other 15, never stopped, and
  the largest reported in-run ratio was 1.71. Slowdown pauses are therefore content-driven
  at a known rate, and the calibration only reports them (below).
- Output: real runs write only under
  `snake-dqn-artifacts/apex-veto-v8-lambda-sweep-20261002/run-v1/{shard-0,shard-1,shard-2,merged}`;
  smokes stay outside `snake-dqn-artifacts`.
- Test safety: every unit test that calls `sweep.main` first replaces
  `tournament_eval.rollout` and `dev_screen.run_episode` with functions that raise, and uses
  a temporary slot lock root.

## 3-slot thermal calibration (pre-declared; computed by `sweep.calibration_report` in `merge`)

This run doubles as the calibration in `docs/research/compute_policy_2026-10-02.md`
("Calibration before 3 slots become the default"): three Tier-1/dev processes at
`--use-slot-locks --slot-pool 3 --thermal-guard --require-ac-power`, each on its own fresh
output directory and a development-only namespace, on AC power with the lid open and
`caffeinate`, while no strict run holds slots 1 and 2.

- **Valid** only if the three shards each passed the start barrier, held the three distinct
  pool-3 slot files, and ran at the same time for at least 90 % of the shortest shard's
  duration (latest start to earliest finish), with all starts within the remaining 10 % of
  it (any positive overlap is not enough: episodes in a long 1- or 2-shard tail are not
  3-slot load, and a short late shard must not pass), and the 2-slot baseline is the
  pinned file: the v7 sweep's `run-v1/events.jsonl` (sha256 `0ea51914...c38ae`), arm L400 =
  v7(lambda=4), the same veto as arm A here, one process holding one slot. The number of
  arm A episodes that finished outside the all-three window is reported.
- **PASS** iff all of: (a) zero thermal-guard stops on any shard; (b) zero guard checks that
  were not ok for a non-slowdown reason (any thermal or performance warning level above 0,
  `CPU_Speed_Limit` or `CPU_Scheduler_Limit` below 100, pmset unknown or failing, battery);
  (c) arm A's mean episode wall time over its 24 episodes is strictly below 1.30 x the
  baseline's over its 24, AND in each mix arm A's mean over its 8 episodes is strictly
  below 1.30 x the baseline's 8 in that mix (the policy's "under 30 %" per mix, for arm A
  only and on other worlds: see the deviations below); and the sweep complete. Reported
  only: slowdown pauses (content-driven at a known rate, see "Thermal guard"; a pause still
  costs 60 s and a persistent-slowdown stop is still a stop under (a)) and the guard's
  in-run slowdown ratios.
- **FAIL** if any criterion fails. A thermal stop is FAIL even though the sweep is then
  incomplete.
- **INVALID** otherwise (for example a deadline stop, a shard without a passed barrier,
  shards overlapping for less than 90 % of the shortest, or a baseline whose bytes
  changed): no conclusion, rerun.
- **Explicit deviations from policy step 4** (`compute_policy_2026-10-02.md`, which asks for
  a slowdown under 30 % "per arm and mix" against a 2-slot baseline "for the same
  episodes"): (1) only arm A is tested, because only v7(4) has a 2-slot baseline (the v8
  arms never ran at 2 slots); (2) the baseline played different worlds (the v7 sweep's), so
  each ratio mixes load with content. Episode wall time has a CV of about 0.35, so a per-mix
  ratio of two 8-episode means has an SE of about 17 % and the pooled 24 vs 24 ratio about
  10 %. With no true slowdown, each mix exceeds 1.30 with probability about 0.07, so the
  per-mix gate gives a false FAIL about 18 % of the time; that error is conservative (it
  keeps 2 slots). A PASS therefore does not meet the policy's literal criterion (same
  episodes, every arm). The separate edit that makes 3 slots the default must cite these
  two deviations and weigh the reported per-mix ratios and the in-run slowdown ratios.
- Consequence: PASS -> 3 slots may become the default for Tier-1/dev runs, as the policy
  requires, in a separate explicit edit with its own note. FAIL -> keep 2 slots for
  Tier-1/dev and record why. Strict (Tier-2) packages stay on slots 1 and 2 either way.
- Disclosed limits: what else ran during the baseline is not recorded; the guard does not
  read the lid.

## Operator commands (after the v7 strict run has closed)

```bash
cd <this worktree>   # the commit that holds this file; tree clean for the real run
R=/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909
A=/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v8-lambda-sweep-20261002/run-v1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
# 0. Confirm the v7 strict run (apex-veto-v7-strict-20261002) has CLOSED: its final
#    summary is written and its process has exited. Do not continue while it runs.
# 1. Create the Tier-1/dev third slot (create-only; status first).
$PY research/compute/slot_setup.py --root $R
$PY research/compute/slot_setup.py --root $R --create
# 2. The one plumbing smoke (2 episodes x 500 frames, smoke namespace, outside artifacts).
SNAKE_DQN_DEVICE=cpu $PY research/apex_veto_v8_lambda_sweep_20261002/sweep.py run \
  --shard 0 --shards 1 --out /tmp/v8-sweep-smoke-$(date +%s) \
  --deadline-utc $(date -u -v+20M +%Y-%m-%dT%H:%M:%S+00:00) \
  --smoke-frames 500 --worlds-per-mix 1 \
  --use-slot-locks --slot-pool 3 --thermal-guard --require-ac-power
# 3. The real sweep: abort unless all 3 pool-3 slots are free now, then three shards at
#    once (each waits at the start barrier for the other two), then merge.
$PY research/apex_veto_v8_lambda_sweep_20261002/sweep.py slots-free --slot-lock-root $R || exit 1
D=$(date -u -v+3H -v-5M +%Y-%m-%dT%H:%M:%S+00:00)
for K in 0 1 2; do
  SNAKE_DQN_DEVICE=cpu caffeinate -dimsu $PY research/apex_veto_v8_lambda_sweep_20261002/sweep.py run \
    --shard $K --out $A/shard-$K --deadline-utc $D \
    --use-slot-locks --slot-pool 3 --thermal-guard --require-ac-power > $A-shard-$K.log 2>&1 &
done; wait
$PY research/apex_veto_v8_lambda_sweep_20261002/sweep.py merge \
  --shard-dirs $A/shard-0 $A/shard-1 $A/shard-2 --out $A/merged
```

(`mkdir -p` the parent of `$A` first; each `--out` must not exist. A barrier refusal
exits 2 with no `--out` and no episode; its markers are ignored once their process is gone
or they are older than 120 s, so the three shards can be relaunched together.)

## Non-claims

The sweep is tuning on development worlds. Its deltas are not evidence that v8 helps; with 8
worlds per mix and a selection over three arms, the selected `lambda`'s delta is biased
upward. Arm A is v7(4), not the released v5: deltas here are increments over the strict-gated
candidate. The head layer's replacement uses v7's score, so at `lambda` 8 and 16 the head
layer and the stronger space preference are confounded; only H400 vs A isolates the head
layer. The released configuration is unaffected: `safety_veto.py`, `safety_veto_v3.py` to
`safety_veto_v7.py` stay byte-identical (tests assert all six), and no strict package is
edited.
