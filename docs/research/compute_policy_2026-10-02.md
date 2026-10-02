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
| Tier-1 screen or dev sweep | up to 3 shared slots in total (slots 1-3) | 2 per slot (about 6 in total) | `--thermal-guard` required with `--slot-pool 3` (enforced: refused otherwise) | AC, lid open |
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
   the v5, v6 and v7 screens) takes `--use-slot-locks --slot-pool 3 --thermal-guard`.
   The pool order is slot 3, then slot 1, then slot 2, so a dev run takes the dev-only slot first and leaves
   the strict slots free when it can. Without `--slot-pool 3` the pool is slots 1 and 2 in
   file order, exactly as before. `--slot-pool 3` without slot locks, or without
   `--thermal-guard`, is refused (exit code 2, before any lock is taken).
4. **Thermal guard.** With 3 slots in use, `--thermal-guard` is required (and keep
   `--require-ac-power`, or a spec that requires it). Before each new episode the guard
   checks:
   - `pmset -g therm`: thermal warning level 0, performance warning level 0, and
     `CPU_Speed_Limit` / `CPU_Scheduler_Limit` not below 100. The note
     "No ... has been recorded" counts as nominal. A level is read only from pmset's own
     format (`Thermal Warning Level = N`, `Performance Warning Level = N`, from
     `strings /usr/bin/pmset`). Any line starting with `Error` (for example
     `Error:Failed to get thermal warning level with error code 0x...`), any other line
     that names a level, a negative level, unknown state, or `pmset` missing or failing on
     macOS is **not ok** (fail closed).
   - Episode slowdown: per `arm/mix`, the median of the run's first 12 episode wall times
     is the baseline. The key is flagged when the median of its latest 8 later episodes is
     more than 60 % above the baseline. These thresholds are wide on purpose. Episode wall
     time depends on game content (10-51 s, coefficient of variation about 0.35 within a
     run). Replaying the five completed v3-v7 screens' `events.jsonl` wall times through the
     guard (pmset off), the first-draft 6 / 30 % / 3 setting would have paused 8, 5, 24, 10
     and 4 times with no thermal cause. 12 / 60 % / 8 pauses 0 times on all five. On the
     2026-09-26 screen, which has one 271 s episode, it pauses 3 times and does not stop.
   - AC power, when the run requires it.

   When the guard is not ok, the run admits no new episode and pauses
   `--thermal-backoff-seconds` (default 60 s), up to `--thermal-max-backoffs` times in a row
   (default 5). It never pauses past the point where the deadline budget would be crossed.
   A pause also answers the slowdown evidence seen so far, so a slowdown flag clears until a
   new window of 8 fresh episodes for that key is slow again. Slowdown is therefore not only
   a throttle: a key flagged again after 2 consecutive slowdown pauses (no unflagged fresh
   window in between) is a **persistent slowdown**, and the run stops at once with
   `thermal: persistent slowdown (...)`. If the guard is still not ok after the backoff
   budget, the run stops with `stopped_reason` `thermal: still not ok (...)`. The decision rule is unchanged: a pre-registered run
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
   happened, any `CPU_Speed_Limit < 100`, and the per-`arm/mix` slowdown `ratio` values in
   each check's `readings.slowdown.keys`, compared with a 2-slot baseline for the same
   episodes. Read the ratios directly; "no guard stop" alone is not evidence, because the
   wide detector tolerates slowdowns below 60 %.
4. Make 3 the default only if the run shows no thermal or performance warning, no speed
   limit, no guard stop, and a measured slowdown under 30 % per arm and mix against the
   2-slot baseline. Otherwise keep 2 slots
   and record why. Either way, that change is a separate, explicit edit with its own note.

## Gaps

- The v7 lambda sweep (`research/apex_veto_v7_lambda_sweep_20261002/sweep.py`) has its own
  episode loop. It still takes one slot from slots 1 and 2 and has no guard. A future sweep
  that wants 3 slots needs the same opt-in wiring.
- The `pmset -g therm` warning-level lines were only seen in their nominal "No ...
  recorded" form on this host. The numeric form is parsed from pmset's own format string,
  and anything else fails closed. The `CPU Power notify` value lines (`CPU_Speed_Limit = N`)
  are not in pmset's string table, so their pattern stays tolerant. The calibration run is
  the first real-format check.
- The slowdown thresholds come from five past screens, not from a throttled run. A
  slowdown below 60 % per `arm/mix` does not trigger a pause.
- The lid (clamshell) state is not read. Rule 5 relies on the operator.
