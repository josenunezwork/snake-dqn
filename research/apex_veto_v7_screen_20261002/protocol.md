# Tier-1 screen: Apex + v7 space-preference veto vs Apex + v5 veto (pre-registration)

Label: **screen (non-authoritative)**. Governance: Tier 1 in
`docs/research/governance_tiers_2026-09-26.md`. Nothing here can change a default, the
champion, a deployment profile or the released Watch-hero veto. Written 2026-10-02, before
any episode of this screen ran and before any episode of the DEV lambda sweep that feeds it
(the only episodes allowed before GO are plumbing smokes on smoke namespaces, at most 2
episodes x 500 frames each).

- `screen_id`: `apex-veto-v7-screen-v1`
- Harness: `research/apex_safety_20260926/dev_screen.py`, driven by `screen.py` beside this
  file. No dev_screen change: the spec uses only existing opt-in fields (callable arm
  installers for A/B/C/D, `replay_worlds`, `max_wall_seconds`, slot lock, AC guard, Tier-1
  intent fields). Output: `intent.json` (before any episode), `records/`, `events.jsonl`,
  `summary.json`, then `receipt.json` (written by `screen.py`).
- Owner: Apex safety lane. `intent.json` records the git commit and dirty paths, this file's
  `protocol_sha256`, the owner, hypothesis, decision informed, primary metric, estimator,
  decision rule, compute cap and (in the arm descriptions and hypothesis) the selected
  `lambda` with the sweep summary's sha256 and source commit, all before any episode.

## Question and the decision it informs

Does `free-space-veto/v7-space-preference(lambda=<selected>)` (`src/evaluation/
safety_veto_v7.py`) raise the Apex champion's H5000 mass integral over the released v5
boost-aware veto (`free-space-veto/v5-boost-aware`, STRICT_PASS and SERVING_PASS, the
served Watch-hero configuration)? Decision informed: design a Tier-2 strict gate for v7
against v5 (`RECOMMEND_STRICT_GATE`) or stop (`NOT_ADVANCED`; the served v5 configuration
is unchanged either way).

### Disclosed: motivation and the lambda sweep

- **Motivation.** The v5 death census (`docs/research/death_census_v5_2026-10-02.md`, worlds
  `trap-horizon-v5-dev-v1`, excluded here) found most remaining self deaths begin with the
  long hero (length 262-807) entering a region too small for it 20-60 frames before death,
  while v5 only asks whether a move reaches `need` (at most 160) cells.
- **`lambda` was chosen on other worlds.** `lambda` is NOT a free parameter of this screen.
  `screen.py` reads it from the summary of the pre-declared DEV sweep
  `research/apex_veto_v7_lambda_sweep_20261002` (namespace `apex-veto-v7-dev-v1`, 8 worlds
  per mix, arms v5 and v7 at `lambda` in {0.25, 0.5, 1.0}; its selection rule is in its own
  `protocol.md`). Outside a smoke, `screen.py` refuses to start unless:
  - `--sweep-summary` exists with its `intent.json` beside it and matching `intent_sha256`;
  - it is a real (non-smoke) `apex-veto-v7-lambda-sweep/v1` summary of `apex-veto-v7-dev-v1`;
  - `selection.status == SELECTED`, `selection.passes` is true and the lambda is on the grid;
  - the sweep's source commit (summary and intent) equals this run's `HEAD`;
  - neither the sweep's tree nor this run's has tracked modifications (untracked files are
    allowed).

  If the sweep returns `NONE_QUALIFIES` (or anything but `SELECTED`), this screen is not run.
  The sweep's worlds, smoke worlds and deltas are not reused here; its selected lambda's
  sweep delta is biased upward by the selection and is not evidence.

## v7 rule (arm B)

Full specification in the module docstring. Summary: v5 runs unchanged. When v5's outcome
is `kept` or `vetoed`, `lambda > 0` and v5's reason is not a landing veto's same-direction
normal-speed replacement, the candidates are the v5-eligible actions in the speed mode of
v5's choice; with at least two, v7 takes the argmax of
`Qn + lambda * log1p(area)/log1p(cap)` (`Qn` min-max normalized over the candidates; ties
to v5's choice, then the lowest index). `area` is the tail-aware (v3 model) reachable count
from the action's exact post-move head (other live snakes static, own body released by
steps, slack 1) with cap `min(max(32, 4*length), 4096, open cells)`. `no_spacious` keeps
v5's choice. Deterministic. Analytic note: for `lambda <= 1` a two-candidate decision never
changes and only the second-highest-Q candidate can displace v5's choice.

The probe (`probes.safety_veto`) is the descriptor (including `space_preference_lambda`)
plus exactly v2's seven counters; a re-rank change is a `vetoed` decision. v7's counters
(`rerank_changes`, area evaluations and cap hits, cost, nested v5 counters) are stored in
each B/D entry's `veto_diagnostics`.

## Arms

| Arm | Hero | Veto |
|---|---|---|
| A | `champion_a5_freespace_20260621.pth` (sha256 `43d4e2c5...d747ac93`) | v5, released (`safety_veto_v5.py` sha256 `d86d084e...ec86c`), installed through `dev_screen.hero_veto_installer` after rollout's built-in install (whose vector61 guard still runs) |
| B | same | v7 at the sweep-selected `lambda`, installed the same way |
| C | same as A | determinism control: A repeated on the first 8 worlds per mix after all A/B pairs; each C record must equal its A record |
| D | same as B | replay control: B repeated on the first 4 worlds per mix after all C; each D record must equal its B record |

## Worlds

- Profile `promotion-v2-watch-rect`, horizon 5000, digest `d396d3ed...0e8b`; config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3...715aa5`).
- Mixes `frozen`, `scripted`, `mixed`; **40 worlds per mix**, the same seeds in every mix:
  `uint32(sha256("apex-veto-v7-screen-v1|worlds|<i>")[:4])`, i = 0..39.
- Construction-time disjointness (fail closed, in code) against the first 1000 seeds of every
  domain the v6 screen excluded (see its protocol), the v6 screen and its smoke, the v7 DEV
  sweep `apex-veto-v7-dev-v1` and its smoke `apex-veto-v7-dev-smoke-v1`, and this screen's
  smoke domain; dev_screen adds the task-aligned challenger namespaces, the strict pilot's
  observed seeds and seeds 0..999. The strict pilot recipe and roster parity checks must
  run and pass. Smokes use `apex-veto-v7-screen-smoke-v1` only.

## Primary metric, estimator, decision rule

- Primary: paired Δ `mass_integral` (B − A) per world, per mix.
- One-sided paired t test per mix, Holm across the three mixes at α = 0.05, 2 of 3 required.
- Decision (`dev_screen.summarize` with `replay_worlds=4`; `screen.py` adds the self-check
  override), in order: `NON_PREREGISTERED_DESIGN` unless 40/8; `INCOMPLETE` if any planned
  A/B, C or D episode is missing; `INVALID_NONDETERMINISTIC` if any C ≠ A or D ≠ B;
  `RECOMMEND_STRICT_GATE` if Holm rejects in ≥ 2 of 3 mixes; else `NOT_ADVANCED`;
  `INVALID_SELF_CHECK_FAILED` (receipt only) if any record fails a gating self-check.
- Reported only: per-mix means, wins and losses; survival Δ; death causes per arm and mix;
  veto counters for A and B; v7 counters and cost for B (`reported.v7_rerank_arm_B`); arm
  A's v5 landing counters; the stratified mean of means with a 90% Welch CI; random-effects
  and crossed pools; sizing for a later Tier-2 design. Say "harm" or "no effect" only when
  the CI excludes the effect of interest; otherwise "not established".

## Self-checks (gating) and their writers

1. `probes.safety_veto` equals the arm's descriptor (A/C v5's; B/D v7's at the selected
   `lambda`) plus exactly the seven v2 counters with `decisions = kept_base +
   vetoes_applied + fallback_no_spacious` (`rollout` attaches `veto.record()`; checked
   with `strict_promotion._validate_candidate_wrapper_probe`).
2. `record.seed`, `world_identity` and `evaluation_profile_digest` match (`rollout(profile=...)`).
3. `denominators.scored_frames` = 5000 and `counters.decisions` = `decision_frames`.
4. Entry schema `apex-veto-v7-screen/v1`, screen id, hero sha256, `safety_veto: true`, the
   arm's method; file name `<arm>-<mix>-<seed>.json`; seed in the intent.

Reported only (warnings): B/D v7 diagnostics agree with the probe (`sweep.v7_identities_hold`),
A/C v5 diagnostics agree with theirs (`v5screen.probe_identities_hold`).

## Compute cap and operations

- 276 episodes: 240 A/B, 24 C, 12 D. Expected about 2.5-3 h on CPU with 2 torch threads
  (about 30 s per live v5 episode; v7 adds a few capped breadth-first counts per decision,
  measured at up to about 1.5 ms per decision when every area is forced at length 800).
- Cap 4 h (14400 s): `--deadline-utc` beyond 4 h after launch is refused, the run stops at
  the deadline (an episode starts only with max(45 s, 2 × mean episode time) left), and a
  deadline stop gives `INCOMPLETE`.
- Sizes: outside a smoke, `screen.py` refuses (exit 2, before any world is played) any size
  other than 40/8. D is fixed at 4 worlds per mix. Smokes never run D and take
  `--smoke-lambda` (a grid value) instead of a sweep summary.
- Slot lock: one shared CPU slot lock (read-only lock files, never created), taken before
  `--out` exists and held until `summary.json`. Power: refuses to start on battery.
- Output: `snake-dqn-artifacts/apex-veto-v7-screen-20261002/run-v1` only.
- Test safety: every unit test that calls `screen.main`, `sweep.main` or `dev_screen.main`
  first replaces `tournament_eval.rollout` and `dev_screen.run_episode` with functions that
  raise.
- Recovery: one mechanical-defect recovery under `recovery-1/` with the same intent, as
  governance Tier 1 allows. Anything else needs a new namespace.

## Non-claims

- `RECOMMEND_STRICT_GATE` is not promotion: v7 would still need a Tier-2 strict gate against
  v5 on fresh worlds with its own source binding, plus a serving qualification (and a SIMD
  port or a live-engine serving path; the SIMD engine supports v2 and v5 only).
- `NOT_ADVANCED` does not refute the census; it says only that this re-rank at this
  `lambda` did not raise the mass integral detectably at this size.
- The released configuration is unaffected: `safety_veto.py`, `_v3.py`, `_v4.py`, `_v5.py`
  and `_v6.py` stay byte-identical (tests assert all five). No dev_screen file changed.
