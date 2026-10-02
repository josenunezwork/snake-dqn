# Compute policy for CPU research runs, 2026-10-02

Status: **policy, opt-in tooling landed; 3 slots are not yet the default.** This policy covers
new runs that start on or after 2026-10-02. It does not change any closed study, any strict
(Tier-2) package or any default code path. Tier names follow
[governance_tiers_2026-09-26.md](governance_tiers_2026-09-26.md).

## Why

The host is an Apple M5 Pro (18 CPU cores, 64 GB). Earlier strict runs died from a
thermal emergency and from clamshell sleep on battery. Runs were also capped at the two
shared slot locks (`cpu-slot-1.lock`, `cpu-slot-2.lock`), each one process with 2 torch
threads, so the machine sat mostly idle while one strict run held both slots.

## Policy

| Run kind | Slots | Threads | Guard | Power |
|---|---|---|---|---|
| Tier-2 strict gate (`research/apex_veto_*_strict_*`) | exactly 2 (slots 1 and 2, unchanged) | 2 per worker | strict runner's own AC check | AC, lid open |
| Tier-1 screen or dev sweep | up to 3 shared slots in total (slots 1-3) | 2 per slot (about 6 in total) | `--thermal-guard` required when 3 slots are in use | AC, lid open |
| Tier-0 diagnostic | no slot | short, read-only | none | any |

Rules:

1. **Strict stays on two slots.** Strict packages keep their all-or-nothing hold of
   `cpu-slot-1.lock` and `cpu-slot-2.lock`. They never read `cpu-slot-3.lock`, and this
   change edits no strict package and nothing under `src/evaluation/`.
2. **The third slot is Tier-1/dev only.** It exists only after an explicit setup step:

   ```bash
   ./venv/bin/python research/compute/slot_setup.py \
     --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909          # status only
   ./venv/bin/python research/compute/slot_setup.py \
     --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909 --create # create slot 3
   ```

   The setup is create-only. It refuses a root that lacks slots 1 and 2, and it never
   opens or rewrites slots 1 and 2. `dev_screen` opens lock files read-only and never
   creates them.
3. **Opting in.** `dev_screen.py` (and any screen wrapper that forwards its flags, such as
   the v5, v6 and v7 screens) takes `--use-slot-locks --slot-pool 3`. The pool order is
   slot 3, then slot 1, then slot 2, so a dev run takes the dev-only slot first and leaves
   the strict slots free when it can. Without `--slot-pool 3` the pool is slots 1 and 2 in
   file order, exactly as before. `--slot-pool 3` without slot locks is refused.
4. **Thermal guard.** With 3 slots in use, pass `--thermal-guard` (and keep
   `--require-ac-power`, or a spec that requires it). Before each new episode the guard
   checks:
   - `pmset -g therm`: thermal warning level 0, performance warning level 0, and
     `CPU_Speed_Limit` / `CPU_Scheduler_Limit` not below 100. The note
     "No ... has been recorded" counts as nominal. Unknown or unparseable state, or
     `pmset` missing or failing on macOS, is **not ok** (fail closed).
   - Episode slowdown: per `arm/mix`, the median of the run's first 6 episode wall times is
     the baseline. The key is flagged when the median of its latest 3 later episodes
     is more than 30 % above the baseline.
   - AC power, when the run requires it.

   When the guard is not ok, the run admits no new episode and pauses
   `--thermal-backoff-seconds` (default 60 s), up to `--thermal-max-backoffs` times in a row
   (default 5). It never pauses past the point where the deadline budget would be crossed.
   A pause also answers the slowdown evidence seen so far, so a slowdown flag clears unless
   new slow episodes appear. If the guard is still not ok after the budget, the run stops
   with `stopped_reason` `thermal: ...`. The decision rule is unchanged: a pre-registered run
   with missing episodes reads `INCOMPLETE`, and nothing is relabelled.
   Every check, pause and stop is a line in `events.jsonl` with an `event` key
   (`thermal_guard_check`, `thermal_guard_backoff`, `thermal_guard_stop`). `intent.json` and
   `summary.json` each carry a `thermal_guard` block. Episode lines keep their old shape.
5. **AC power and lid open for every run that plays episodes.** Disable sleep on AC for the
   run (for example `caffeinate -dimsu -w <pid>`), keep the lid open, and do not run on
   battery. The guard reads AC power and thermal state. It does not detect the lid.
6. **Defaults are unchanged.** Without the new flags, `dev_screen` writes the same
   `intent.json`, `summary.json` and `events.jsonl` as before (verified against the base
   commit `df86c32` on a stubbed run, apart from timestamps). The guard modules live in
   `research/compute/`, outside the strict source-closure roots, and are imported lazily
   only when a flag is passed.

## Calibration before 3 slots become the default

Three slots stay opt-in until one calibration run is done:

1. While no strict run holds slots 1 and 2, create slot 3 with the setup step above.
2. Run one 30-minute Tier-1 dev run, three processes at `--use-slot-locks --slot-pool 3
   --thermal-guard --require-ac-power`, each on its own fresh output directory and a
   development-only namespace. Use AC power, keep the lid open, and use `caffeinate`.
3. Read the guard events: how many checks were not ok, which reasons fired, how many pauses
   happened, any `CPU_Speed_Limit < 100`, and the per-`arm/mix` slowdown ratios against a
   2-slot baseline for the same episodes.
4. Make 3 the default only if the run shows no thermal or performance warning, no speed
   limit, no guard stop, and a slowdown under 30 % per arm and mix. Otherwise keep 2 slots
   and record why. Either way, that change is a separate, explicit edit with its own note.

## Gaps

- The v7 lambda sweep (`research/apex_veto_v7_lambda_sweep_20261002/sweep.py`) has its own
  episode loop. It still takes one slot from slots 1 and 2 and has no guard. A future sweep
  that wants 3 slots needs the same opt-in wiring.
- The `pmset -g therm` warning-level line format was only seen in its nominal "No ...
  recorded" form on this host. The numeric forms are parsed by tolerant patterns, and an
  unrecognised relevant line fails closed. The calibration run is the first real-format check.
- The lid (clamshell) state is not read. Rule 5 relies on the operator.
