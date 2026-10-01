# Tier-1 screen: Apex + v5 boost-aware veto vs Apex + v2 veto (pre-registration)

Label: **screen (non-authoritative)**. Governance: Tier 1 in
`docs/research/governance_tiers_2026-09-26.md`. Nothing here can change a default, the
champion, a deployment profile or the released Watch-hero veto. Written 2026-10-01, before
any screen episode or smoke of this screen ran.

- `screen_id`: `apex-veto-v5-screen-v1`
- Harness: `research/apex_safety_20260926/dev_screen.py`, driven by `screen.py` beside this
  file (`ScreenSpec` `SPEC`). No dev_screen change was needed: the spec uses only existing
  opt-in fields (`replay_worlds`, `max_wall_seconds`, slot lock, AC guard, Tier-1 intent
  fields). Output: `intent.json` (before any episode), `records/`, `events.jsonl`,
  `summary.json`, then `receipt.json` (written by `screen.py`).
- Owner: Apex safety lane. `intent.json` records the git SHA and dirty paths, this file's
  `protocol_sha256`, the owner, hypothesis, decision informed, primary metric, estimator,
  decision rule and compute cap (`compute_cap`: planned episodes, deadline, 4 h maximum),
  all before any episode runs.

## Question and the decision it informs

Does replacing the released v2 free-space veto (`free-space-veto/v2-speed-preserving`,
STRICT_PASS in `apex-veto-strict-20260927` run-v3; SERVING_PASS for the Watch hero) with the
v5 boost-aware veto (`free-space-veto/v5-boost-aware`, `src/evaluation/safety_veto_v5.py`)
raise the Apex champion's H5000 mass integral? The decision informed: design a Tier-2
strict gate for v5 against v2 (`RECOMMEND_STRICT_GATE`) or stop (`NOT_ADVANCED`).

### Motivation (disclosed): the trap-horizon diagnostic

This screen was designed after, and because of, the trap-horizon development diagnostic
(`docs/research/trap_horizon_2026-10-01.md`, `research/trap_horizon_20261001`, commit
`31584a9`). That diagnostic ran champion + v2 on 48 H5000 worlds of a **different**
namespace, `trap-horizon-dev-v1`, which this screen excludes (see Worlds). In 24 of its 42
self-collision deaths the point-of-no-return action was a boost whose two-cell landing state
had 1 to 24 reachable cells (need 160, or 50 for a short hero), while the same direction at
normal speed still had a way out. v2 scores a boost only from its first cell, so it kept
those boosts. The hypothesis was formed on those worlds; none of them are reused here. The
diagnostic did not show that the hero survives longer after switching to normal speed; that
is what this screen asks, on fresh worlds and with the mass integral as the outcome.

## v5 rule (the arm-B configuration; it has no options)

Full specification, with `file:line` references to the boost mechanics, in the module
docstring. Summary:

- **Non-boost actions: exactly v2.** `cap = min(160, max(32, 2 * length))`,
  `need = min(length, cap)`, direction spacious iff `round(feature * cap) >= need` from the
  snake's own `_get_free_space_features` (v2's functions, imported unchanged).
- **Boost actions (3-5): first cell AND landing.** A boost is eligible only if it is
  masked-legal, its direction is spacious by v2's first-cell count AND its two-cell
  landing count is at least `need`. The landing count replays the boost's exact one-frame
  movement (`Snake.move`, `src/game/snake.py:134-177`: two head pushes with the
  `len > length` tail pop, then the boost cost `boost_frames`/`BOOST_LENGTH_COST_FRAMES`
  with its extra tail pop), blocks the resulting body except the new head (so the cell
  between is blocked) plus every other live snake's cells (static, as in v2), and
  flood-fills from the landing cell with v2's `cap`. If the boost would not fire
  (`length < MIN_BOOST_LENGTH`) the landing test is skipped.
- **Choice.** Keep the base if eligible. A boost base that v2 would keep but whose landing
  fails is replaced by the same direction at normal speed when that is masked-legal
  (spacious by construction); otherwise, and in every other case, v2's speed-preserving
  `veto_choice` runs over the v5-eligible actions. Nothing eligible: base unchanged
  (`no_spacious`), as in v2.
- The probe (`probes.safety_veto`) is the descriptor plus exactly v2's seven counters,
  counting the action taken. The v5 counters (`boost_landing_vetoes`,
  `boost_to_normal_same_direction`, `boost_landing_v2_rule`, `boost_landing_no_eligible`,
  landing checks/failures, `action_differs_from_v2`, wall cost) are stored in each B/D
  entry's `veto_diagnostics`.

## Arms

| Arm | Hero | Veto |
|---|---|---|
| A | `champion_a5_freespace_20260621.pth` (sha256 `43d4e2c5...d747ac93`) | v2 via `rollout(hero_safety_veto=True)`, the released configuration (`safety_veto.py` sha256 `1b62d15c...c428`) |
| B | same checkpoint | v5 |
| C | same as A | determinism control: A repeated on the first 8 worlds per mix, after all A/B pairs. Each C record must equal its A record exactly. |
| D | same as B | B replay control: B repeated on the first 4 worlds per mix, after all C episodes. Each D record must equal its B record exactly. |

Opponent pool, roster construction and checkpoint snapshots are the dev_screen ones (strict
balanced rosters).

## Worlds

- Profile `promotion-v2-watch-rect`, horizon 5000, digest `d396d3ed...0e8b`. Config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3...715aa5`).
- Mixes `frozen`, `scripted`, `mixed`. **40 worlds per mix**, the same seeds in every mix.
- Seeds: `uint32(sha256("apex-veto-v5-screen-v1|worlds|<i>")[:4])`, i = 0..39.
- Construction-time disjointness (fail closed, in code) against the first 1000 seeds of:
  - the task-aligned challenger namespaces, the strict pilot's observed seeds and seeds
    0..999 (dev_screen checks);
  - `apex-safety-screen-v1/worlds` (also the worlds the SIMD H5000 parity check replayed);
  - `apex-veto-strict-{dev,final,serving}-v{1,2,3}/worlds` and `apex-veto-strict-smoke-v1`;
  - `apex-veto-web-serving-v1` and `apex-veto-web-serving-smoke-v1` (`watch`, `play`,
    `parity`);
  - `apex-veto-v3-screen-v1` (consumed and quarantined), `apex-veto-v3-screen-v2` and
    `apex-veto-v3-screen-smoke-v1`;
  - `apex-veto-v4-screen-v1` and `apex-veto-v4-screen-smoke-v1`;
  - `trap-horizon-dev-v1` and `trap-horizon-dev-smoke-v1` (the motivating diagnostic);
  - this screen's smoke domain, `apex-veto-v5-screen-smoke-v1`.

  The strict pilot recipe and roster parity checks must run and pass.
- Smoke dry runs use `apex-veto-v5-screen-smoke-v1` only, never these 40 worlds.

## Primary metric, estimator, decision rule

- Primary: paired Δ `mass_integral` (B − A) per world, per mix.
- Test: one-sided paired t test per mix (`paired_delta_test`), Holm across the three mixes at
  α = 0.05 (`holm_three_mix_superiority`, 2 of 3 required).
- Decision (computed by `dev_screen.summarize` with `replay_worlds=4`; `screen.py` adds the
  self-check override), in order:
  - `NON_PREREGISTERED_DESIGN` unless worlds per mix = 40 and determinism worlds = 8;
  - `INCOMPLETE` if any planned A/B pair, C or D episode is missing;
  - `INVALID_NONDETERMINISTIC` if any C record differs from its A record or any D record
    differs from its B record;
  - `RECOMMEND_STRICT_GATE` if Holm rejects H0 in at least 2 of 3 mixes;
  - `NOT_ADVANCED` otherwise;
  - `INVALID_SELF_CHECK_FAILED` (receipt only) if any record fails a gating self-check.
- Reported, not used for the decision:
  - per-mix means, wins and losses; survival fraction Δ; death causes (in particular the
    `self` count, which v5 targets);
  - veto counters for A and B;
  - the v5 landing counters and cost (`receipt.json` `reported.v5_landing`: landing
    vetoes, same-direction normal-speed replacements, landing checks per decision,
    decisions differing from v2, mean wall seconds per decision, mean B episode wall time
    against A's);
  - the mix-stratified mean of means with a 90% Welch CI and clear-loser flags;
  - random-effects and crossed pools;
  - sizing for a later Tier-2 design.

  Wording: say "harm" or "no effect" only when the CI excludes the effect of interest;
  otherwise say "not established".

## Self-checks (gating) and their writers

`screen.py` runs these on every record after the run. They are tested in
`tests/test_safety_veto_v5.py`.

1. `probes.safety_veto` equals the arm's descriptor plus exactly the seven v2 counters, with
   `decisions = kept_base + vetoes_applied + fallback_no_spacious`. Writer: `rollout`
   attaches `veto.record()`. Checked with `strict_promotion._validate_candidate_wrapper_probe`.
2. `record.seed` equals the planned world seed, `world_identity` equals the materialized
   roster's identity, and `evaluation_profile_digest` is the profile's. Writer:
   `rollout(profile=...)`.
3. `denominators.scored_frames` = 5000 and `counters.decisions` =
   `denominators.decision_frames`. Writers: `EvaluationMetricsAccumulator`, and
   `AISnake.update` → `veto.apply`.
4. Entry: schema `apex-veto-v5-screen/v1`, screen id, hero sha256, `safety_veto: true`, and
   `safety_veto_method` matching the arm. The file name is `<arm>-<mix>-<seed>.json`, and the
   seed is in the intent.

Reported only (warnings): whether each B/D entry's `veto_diagnostics` agrees with its probe
counters (`screen.probe_identities_hold`): equal `decisions`, `kept_base = kept`,
`vetoes_applied` equal, `fallback_no_spacious = no_spacious`,
`vetoed_base_boost >= boost_landing_vetoes` and
`vetoes_speed_switched >= boost_to_normal_same_direction`.

## Compute cap and operations

- 276 episodes: 240 A/B, 24 C and 12 D.
- Expected time is about 1.7 to 2 h on CPU with 2 torch threads, assuming about 21 s per v2
  episode (the run-v3 mean). v5 adds at most one capped flood fill per boost direction per
  decision, and only when the choice depends on it (the same cost class as v2's own three
  fills).
- Cap: at most 4 h (14400 s) of wall time. `screen.py` refuses a `--deadline-utc` more than
  4 h after launch, and the run stops at that deadline. The harness will not start an
  episode with less than max(45 s, 2 × the mean episode time) remaining. A deadline stop
  gives `INCOMPLETE`.
- Sizes: outside a smoke, `screen.py` refuses (exit 2, before any world is played) any
  `--worlds-per-mix`/`--determinism-worlds` other than 40/8. D is fixed at 4 worlds per mix
  by the spec. Smokes never run D.
- Slot lock: the screen holds one shared CPU slot lock (`cpu-slot-{1,2}.lock` under
  `snake-dqn-artifacts/pqn-followup-20260909`, opened read-only, never created). It takes
  the lock before `--out` exists and holds it until `summary.json` is written.
- Power: the screen refuses to start on battery (macOS `pmset`).
- Output: the real run writes only under its own root,
  `snake-dqn-artifacts/apex-veto-v5-screen-20261001/run-v1`. Lane work and smokes stay
  outside `snake-dqn-artifacts`.
- Before GO: one smoke dry run (≤ 2 episodes × 500 frames, legacy path, smoke namespace)
  through `screen.py` with self-check and receipt, from the same source. Landing vetoes may
  not occur in a short smoke, so the landing path is pinned by a unit test: a live
  tiny-world rollout with every landing forced to fail must take landing vetoes, satisfy
  the probe identities and the B self-check, and replay identically.
- Test safety: every unit test that calls `screen.main`, `dev_screen.main` or another
  harness entry first replaces `tournament_eval.rollout` and `dev_screen.run_episode` with
  functions that raise (the v3 revision-4 runaway lesson).
- Recovery: one mechanical-defect recovery under `recovery-1/` with the same intent (same
  arms, seeds, metric and rule), as governance Tier 1 allows. Anything else needs a new
  namespace.

## Non-claims

- A `RECOMMEND_STRICT_GATE` is not promotion. v5 would still need a Tier-2 strict gate on
  fresh worlds with its own source-closure binding, plus a serving qualification of its own.
- A `NOT_ADVANCED` does not refute the trap-horizon finding; it says only that a two-cell
  landing check on boosts did not raise the mass integral detectably at this size.
- The released v2 configuration is unaffected. `src/evaluation/safety_veto.py` (sha256
  `1b62d15c...c428`), `safety_veto_v3.py` and `safety_veto_v4.py` stay byte-identical;
  tests assert all three. No dev_screen file changed.
