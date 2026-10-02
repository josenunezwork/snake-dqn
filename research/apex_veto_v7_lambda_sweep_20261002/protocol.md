# Tier-1 DEV lambda sweep: v7 space-preference veto vs the released v5 veto (pre-declaration)

Label: **development sweep (Tier-1 dev, non-authoritative)**. Governance: Tier 1 in
`docs/research/governance_tiers_2026-09-26.md`. It cannot change a default, the champion, a
deployment profile or the released Watch-hero veto, and it reaches no screen decision. Its
only output is the choice of ONE `lambda` for the separate, pre-registered Tier-1 screen
`research/apex_veto_v7_screen_20261002` (or a stop). Written 2026-10-02, before any episode
of this sweep ran (the only episodes allowed before GO are one plumbing smoke on the smoke
namespace `apex-veto-v7-dev-smoke-v1`, at most 2 episodes x 500 frames).

- `sweep_id`: `apex-veto-v7-dev-v1`
- Harness: `sweep.py` beside this file. It reuses `research/apex_safety_20260926/dev_screen.py`
  pieces unchanged (seed recipe, disjointness and roster-parity preflight, strict balanced
  rosters, checkpoint snapshots, `run_episode` with a `ScreenSpec` that maps every sweep
  arm to its veto installer, slot locks, AC check). No dev_screen change. Output:
  `intent.json` (before any episode), `records/`, `events.jsonl`, `summary.json`.
- Owner: Apex safety lane. `intent.json` records the git commit and dirty paths, this
  file's sha256, the arms, the selection rule and the compute cap before any episode.

## Question

Which `lambda` of `free-space-veto/v7-space-preference(lambda=...)`
(`src/evaluation/safety_veto_v7.py`) should the v7 screen test against the released v5
veto? v7 runs v5 unchanged and, when v5 had a real choice, re-ranks v5's eligible actions in
the speed mode of v5's choice by `Qn + lambda * g(area)`: `Qn` is the Q-value min-max
normalized over those candidates (in `[-1, 0]`), `g = log1p(area)/log1p(cap)` and `area`
the tail-aware (v3 model) reachable count from the action's exact post-move head with cap
`min(max(32, 4*length), 4096, open cells)`. `lambda = 0` is v5 decision for decision.

Motivation (disclosed): the v5 death census (`docs/research/death_census_v5_2026-10-02.md`)
found that most remaining self deaths begin with the long hero (262-807) entering a region
too small for it 20-60 frames before death, while v5 only asks whether a move reaches
`need` (at most 160) cells. Its worlds (`trap-horizon-v5-dev-v1`) are excluded here.

Analytic consequence of the specified normalization (disclosed before any run): with two
candidates `Qn` is exactly `{0, -1}`, so for `lambda <= 1` a two-candidate decision can never
change (the other action would need `lambda * (g_other - g_v5) > 1`, and `g` is in `[0, 1]`).
With three candidates only the second-highest-Q action can displace v5's choice, when its
`Qn` gap is below `lambda` times its `g` advantage. The grid therefore tests a mild
preference.

## Arms (paired: every arm plays every world)

| Arm | Hero | Veto |
|---|---|---|
| A | `champion_a5_freespace_20260621.pth` (sha256 `43d4e2c5...d747ac93`) | v5 (`lambda = 0`), the released `free-space-veto/v5-boost-aware` (`safety_veto_v5.py` sha256 `d86d084e...ec86c`) |
| L025 | same | v7, `lambda = 0.25` |
| L050 | same | v7, `lambda = 0.5` |
| L100 | same | v7, `lambda = 1.0` |
| R | same | replay control: L050 repeated on the first world of each mix, after all other episodes. Each R record must equal its L050 record exactly. |

Vetoes are installed through `dev_screen.hero_veto_installer` after rollout's built-in
install (whose vector61 guard still runs). Opponent pool and roster construction are the
dev_screen ones (strict balanced rosters).

## Worlds

- Profile `promotion-v2-watch-rect`, horizon 5000, digest `d396d3ed...0e8b`. Config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3...715aa5`).
- Mixes `frozen`, `scripted`, `mixed`. **8 worlds per mix** (24 world-mix pairs), the same
  seeds in every mix: `uint32(sha256("apex-veto-v7-dev-v1|worlds|<i>")[:4])`, i = 0..7.
- Construction-time disjointness (fail closed, in code) against the first 1000 seeds of
  every earlier domain the v6 screen excluded (every `apex-safety`, `apex-veto-strict-*`,
  web-serving, v3/v4/v5/v6 screen, v5 strict and serving, trap-horizon v2 and v5 census
  domain, each with its smoke), the v6 screen and its smoke, this sweep's smoke domain, and
  the future v7 screen domains `apex-veto-v7-screen-v1` and `apex-veto-v7-screen-smoke-v1`
  (so the sweep can never consume screen worlds). dev_screen adds the task-aligned
  challenger namespaces, the strict pilot's observed seeds and seeds 0..999. The strict
  pilot recipe and roster parity checks must run and pass.

## Selection rule (pre-declared; computed by `sweep.summarize`)

For each `lambda` in {0.25, 0.5, 1.0}: per world and mix, the paired delta
`mass_integral(v7_lambda) - mass_integral(A)`; per mix the mean over its 8 worlds; pooled the
mean over all 24 world-mix pairs.

1. `SMOKE_NO_SELECTION` for a smoke.
2. `INVALID_SELF_CHECK_FAILED` if any record fails a gating self-check (below).
3. `INCOMPLETE` if any planned episode is missing (for example a deadline stop).
4. `INVALID_NONDETERMINISTIC` if any R record differs from its L050 record.
5. A `lambda` **qualifies** iff its mean delta in EACH mix is `>= -10` mass.
6. `SELECTED`: the qualifying `lambda` with the highest pooled mean delta (ties: the smaller
   `lambda`). The pooled mean may be negative; the screen decides, not this sweep.
7. `NONE_QUALIFIES` if no `lambda` qualifies: stop, no screen is run.

Only `SELECTED` lets the screen run (`summary.selection.passes = true`). The summary also
records `source.commit` and `source.dirty_paths` from `intent.json`; the screen refuses unless
that commit equals its own. Everything else in the summary (per-mix means of each arm, wins
and losses, death causes, v7 counters, wall time) is reported only. No significance test is
run: 8 worlds per mix cannot support one, and this is a tuning step.

## Self-checks (gating)

On every record: `probes.safety_veto` equals the arm's descriptor (A: v5's; L*/R: v7's at
that `lambda`) plus exactly the seven v2 counters
(`strict_promotion._validate_candidate_wrapper_probe`); `record.seed` is the planned world
seed and the entry's arm/mix/seed match its file name; `mass_integral` is finite; outside a
smoke, `evaluation_profile_digest` is the profile's, `world_identity` matches the roster,
`denominators.scored_frames = 5000` and `counters.decisions = denominators.decision_frames`;
entry schema, sweep id, hero sha256, `safety_veto: true` and the arm's method. Reported only:
the v7 diagnostics identities against the probe (`sweep.v7_identities_hold`).

## Compute cap and operations

- 99 episodes: 96 paired (4 arms x 24) plus 3 R.
- Expected time about 50-70 min on CPU with 2 torch threads: about 30 s per live v5 episode
  (census: 1,436 s for 48), and v7 adds at most a few capped breadth-first counts per
  decision (about 0.3 us per cell; measured 0.2 ms per decision without, up to 1.5 ms per
  decision when every area is forced at length 800, usually one evaluation or none).
- Cap: 3 h (10800 s). `sweep.py` refuses a `--deadline-utc` more than 3 h after launch and
  stops at the deadline (an episode starts only with max(45 s, 2 x mean episode time) left);
  a deadline stop gives `INCOMPLETE`.
- Size: outside a smoke, `sweep.py` refuses (exit 2, before any world is played) any
  `--worlds-per-mix` other than 8. A smoke plays only arms A and L100 on one world of one
  mix (2 episodes, <= 500 frames, legacy path, smoke namespace) and never R.
- Slot lock: one shared CPU slot lock (`cpu-slot-{1,2}.lock` under
  `snake-dqn-artifacts/pqn-followup-20260909`, opened read-only, never created), taken before
  `--out` exists and held until `summary.json` is written. Power: refuses to start on
  battery (macOS `pmset`).
- Output: the real run writes only under
  `snake-dqn-artifacts/apex-veto-v7-lambda-sweep-20261002/run-v1`; smokes stay outside
  `snake-dqn-artifacts`.
- Test safety: every unit test that calls `sweep.main` first replaces
  `tournament_eval.rollout` and `dev_screen.run_episode` with functions that raise.

## Non-claims

The sweep is tuning on development worlds. Its deltas are not evidence that v7 helps; with
8 worlds per mix and a selection over three arms, the selected `lambda`'s sweep delta is
biased upward. The screen uses fresh worlds and its own pre-registered decision rule. The
released configuration is unaffected: `safety_veto.py`, `safety_veto_v3.py`, `_v4.py`,
`_v5.py` and `_v6.py` stay byte-identical (tests assert all five).
