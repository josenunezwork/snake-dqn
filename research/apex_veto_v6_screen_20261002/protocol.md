# Tier-1 screen: Apex + v6 head-and-fallback veto vs Apex + v5 veto (pre-registration)

Label: **screen (non-authoritative)**. Governance: Tier 1 in
`docs/research/governance_tiers_2026-09-26.md`. Nothing here can change a default, the
champion, a deployment profile or the released Watch-hero veto. Written 2026-10-02, before
any episode of this screen ran (the only episodes allowed before GO are one plumbing smoke
on the smoke namespace `apex-veto-v6-screen-smoke-v1`, at most 2 episodes x 500 frames).

- `screen_id`: `apex-veto-v6-screen-v1`
- Harness: `research/apex_safety_20260926/dev_screen.py`, driven by `screen.py` beside this
  file (`ScreenSpec` `SPEC`). No dev_screen change was needed: the spec uses only existing
  opt-in fields (callable arm installers for A/B/C/D, `replay_worlds`, `max_wall_seconds`,
  slot lock, AC guard, Tier-1 intent fields). Output: `intent.json` (before any episode),
  `records/`, `events.jsonl`, `summary.json`, then `receipt.json` (written by `screen.py`).
- Owner: Apex safety lane. `intent.json` records the git SHA and dirty paths, this file's
  `protocol_sha256`, the owner, hypothesis, decision informed, primary metric, estimator,
  decision rule and compute cap (`compute_cap`: planned episodes, deadline, 4 h maximum),
  all before any episode runs.

## Question and the decision it informs

Does adding two layers on top of the released v5 boost-aware veto
(`free-space-veto/v5-boost-aware`, `src/evaluation/safety_veto_v5.py`; STRICT_PASS and
SERVING_PASS, the served Watch-hero configuration on local main `03abe8e`) raise the Apex
champion's H5000 mass integral? The candidate is `free-space-veto/v6-head-and-fallback`
(`src/evaluation/safety_veto_v6.py`). The decision informed: design a Tier-2 strict gate
for v6 against v5 (`RECOMMEND_STRICT_GATE`) or stop (`NOT_ADVANCED`; the served v5
configuration is unchanged either way).

### Motivation (disclosed): the v5 death census

This screen was designed after, and because of, the v5 death census
(`docs/research/death_census_v5_2026-10-02.md`, `research/trap_horizon_20261001/
diagnose_live_v5.py`, commit `e04da43`; results commit `c025dcf`). That development
diagnostic ran champion + v5 on 48 H5000 worlds of a **different** namespace,
`trap-horizon-v5-dev-v1`, which this screen excludes (see Worlds), and read (only) the death
causes of the v5 strict-gate final records (`apex-veto-v5-strict-final-v1`, also excluded).
Its ranked recommendations 1 and 2 are the two v6 layers:

1. **Head-ons.** 7.8% of v5 strict-final episodes ended in a head-on (12.3% in the
   scripted mix; 30 of the 33 scripted head-ons within roughly the first 500 frames, mean
   mass 0.6 against 183.4 for the other scripted episodes). In all 5 census head-ons v5
   `kept` a move into a cell that was empty before the move but reachable by an
   equal-or-larger opponent's head, and 2 to 5 masked-legal alternatives survived the frame
   with a static escape.
2. **No-spacious self deaths.** All 36 census self deaths were `no_spacious` at the fatal
   decision, where v5 leaves the base action unchanged; 10 point-of-no-return choices were
   boosts with landing counts of 1 to 44 cells (same direction at normal speed escaped in 7).
   The census expects mostly delay from this layer (most of those heroes were enclosed 20-60
   frames earlier), so its expected effect is smaller and less certain than layer 1's.

The hypothesis was formed on those worlds and records; none of them are reused here. The
census did not show that the hero survives longer or gains mass after either change; that
is what this screen asks, on fresh worlds, with the mass integral as the outcome. The
mechanism estimate (scripted mass +20, pooled about +8 if every early head-on became a
typical episode) is an upper-bound style guess, not a prediction.

## v6 rule (the arm-B configuration; both layers on, no other options)

Full specification, with code references, in the module docstring. Summary:

- **Base: exactly v5.** v2's cap/need and one-step spacious test, v5's two-cell landing
  check for boosts and v5's replacement rule, imported unchanged. v5's own counters are
  recomputed beside v6's (`veto_diagnostics.v5`).
- **Layer (a), opponent-head avoidance** (only when v5's outcome is `kept` or `vetoed`).
  An opponent head can reach next frame: its current cell, the first cell of each of its
  three relative moves and, if `length >= MIN_BOOST_LENGTH`, the second cell of each.
  An action is head-risky if any cell it traverses (both cells of a boost, from v5's exact
  one-frame movement model) is reachable by an opponent the hero would not beat. Head-on
  rule (`GameState.handle_collisions`): under mechanics v2 a snake whose logical length is
  at least 1.15x the other's survives, else both die (always mutual at v1). The hero counts
  as the winner only if its post-burn length is >= 1.15 x (opponent logical length + 1),
  conservatively allowing one pellet of opponent growth this frame. A head-risky v5 choice
  is replaced by the highest-Q masked-legal, v5-eligible, non-risky other action in the same
  speed mode, else the other mode; if there is none, v5's choice stands.
- **Layer (b), no-spacious fallback** (only when v5's outcome is `no_spacious`; it never
  overlaps layer (a), which needs a v5-eligible alternative). (i) A firing boost whose
  landing count is below its direction's normal-speed count minus 1 becomes that
  direction at normal speed when masked-legal. Both counts use one model (v2's static
  capped flood over the hero's post-move body and the other live snakes, from the new
  head: v5's `landing_count` for the boost, `normal_post_count` for normal speed). A boost
  uses one more cell than normal speed, so in a region the first cell does not split the
  landing is exactly normal - 1 and the boost is NOT switched; the switch fires only when
  the second cell cuts the hero off from part of what the first cell reaches (the census
  pattern). `fallback_landing_switches` therefore counts pocket boosts, not every
  no-spacious boost. (ii) v4's `EscapeSearch` (depth 8, node budget 2000 per decision,
  fixed child order), judged per action: a normal-speed action is v4's `escape(direction)`;
  a firing boost must escape along its forced path (first cell, then the next cell
  straight ahead at step 2 with the first cell on the path, and that second cell not next
  to another live snake's head, as v4 requires of a first cell). If the action escapes it
  is kept; otherwise the other masked-legal actions are tried in descending Q (ties:
  lowest index; boosts only if their landing passes (i)'s test) and the first that
  escapes is chosen. Budget exhaustion leaves (i)'s result unchanged
  (`fallback_budget_unknown`). These two model choices (like-for-like landing counts,
  boost forced path) were fixed after review and before any real screen episode.
- Deterministic: no randomness or timing in any choice.
- The probe (`probes.safety_veto`) is the descriptor plus exactly v2's seven counters. A
  layer-(a) replacement is a `vetoed` decision; every layer-(b) decision stays
  `no_spacious` (no v2-spacious action existed). The v6 counters (`head_risky_vetoes`,
  `head_risky_kept_no_alternative`, `head_risk_waived_hero_wins`,
  `fallback_landing_switches`, `fallback_escape_switches`, `fallback_budget_unknown`,
  search nodes, `action_differs_from_v5`, wall cost, nested v5 counters) are stored in each
  B/D entry's `veto_diagnostics`.

## Arms

| Arm | Hero | Veto |
|---|---|---|
| A | `champion_a5_freespace_20260621.pth` (sha256 `43d4e2c5...d747ac93`) | v5, the released configuration (`safety_veto_v5.py` sha256 `d86d084e...ec86c`), installed by `screen.install_v5` through `dev_screen.hero_veto_installer` after rollout's built-in install (whose vector61 guard still runs) |
| B | same checkpoint | v6, installed by `screen.install_v6` the same way |
| C | same as A | determinism control: A repeated on the first 8 worlds per mix, after all A/B pairs. Each C record must equal its A record exactly. |
| D | same as B | B replay control: B repeated on the first 4 worlds per mix, after all C episodes. Each D record must equal its B record exactly. |

Opponent pool, roster construction and checkpoint snapshots are the dev_screen ones (strict
balanced rosters).

## Worlds

- Profile `promotion-v2-watch-rect`, horizon 5000, digest `d396d3ed...0e8b`. Config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3...715aa5`).
- Mixes `frozen`, `scripted`, `mixed`. **40 worlds per mix**, the same seeds in every mix.
- Seeds: `uint32(sha256("apex-veto-v6-screen-v1|worlds|<i>")[:4])`, i = 0..39.
- Construction-time disjointness (fail closed, in code) against the first 1000 seeds of:
  - the task-aligned challenger namespaces, the strict pilot's observed seeds and seeds
    0..999 (dev_screen checks);
  - `apex-safety-screen-v1/worlds` (also the worlds the SIMD H5000 parity check replayed);
  - `apex-veto-strict-{dev,final,serving}-v{1,2,3}/worlds` and `apex-veto-strict-smoke-v1`;
  - `apex-veto-web-serving-v1` and `apex-veto-web-serving-smoke-v1` (`watch`, `play`,
    `parity`);
  - `apex-veto-v3-screen-v1`, `apex-veto-v3-screen-v2`, `apex-veto-v3-screen-smoke-v1`,
    `apex-veto-v4-screen-v1`, `apex-veto-v4-screen-smoke-v1`;
  - `trap-horizon-dev-v1` and `trap-horizon-dev-smoke-v1` (the v2 trap-horizon study);
  - every `apex-veto-v5-*` domain: `apex-veto-v5-screen-v1` and its smoke,
    `apex-veto-v5-strict-{dev,final,serving}-v1` and `apex-veto-v5-strict-smoke-v1`,
    `apex-veto-v5-web-serving-v1` and its smoke (`watch`, `play`, `parity`);
  - `trap-horizon-v5-dev-v1` and `trap-horizon-v5-dev-smoke-v1` (the motivating census);
  - this screen's smoke domain, `apex-veto-v6-screen-smoke-v1`.

  The strict pilot recipe and roster parity checks must run and pass.
- Smoke dry runs use `apex-veto-v6-screen-smoke-v1` only, never these 40 worlds.

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
  - per-mix means, wins and losses; survival fraction Δ; death causes per arm and mix
    (`reported.death_causes`; in particular `head_on`, which layer (a) targets, and `self`);
  - veto counters for A and B;
  - the v6 per-layer counters and cost (`reported.v6_layers`, arm B per mix and in total:
    head-risky decisions, head vetoes, kept-without-alternative, size-rule waivers, fallback
    decisions, landing and escape switches, budget-unknown decisions, search nodes,
    decisions differing from v5, mean wall seconds per decision and per fallback, mean B
    episode wall time against A's);
  - arm A's v5 landing counters (`reported.v5_landing_arm_A`);
  - the mix-stratified mean of means with a 90% Welch CI and clear-loser flags;
  - random-effects and crossed pools;
  - sizing for a later Tier-2 design.

  Layer attribution is descriptive only: the screen tests the two layers together, as
  shipped in v6, and cannot say which layer caused an effect.
  Wording: say "harm" or "no effect" only when the CI excludes the effect of interest;
  otherwise say "not established".

## Self-checks (gating) and their writers

`screen.py` runs these on every record after the run. They are tested in
`tests/test_safety_veto_v6.py`.

1. `probes.safety_veto` equals the arm's descriptor (A/C: v5's; B/D: v6's, including
   `head_avoidance: true`, `no_spacious_fallback: true`, depth 8 and budget 2000) plus
   exactly the seven v2 counters, with `decisions = kept_base + vetoes_applied +
   fallback_no_spacious`. Writer: `rollout` attaches `veto.record()`. Checked with
   `strict_promotion._validate_candidate_wrapper_probe`.
2. `record.seed` equals the planned world seed, `world_identity` equals the materialized
   roster's identity, and `evaluation_profile_digest` is the profile's. Writer:
   `rollout(profile=...)`.
3. `denominators.scored_frames` = 5000 and `counters.decisions` =
   `denominators.decision_frames`. Writers: `EvaluationMetricsAccumulator`, and
   `AISnake.update` → `veto.apply`.
4. Entry: schema `apex-veto-v6-screen/v1`, screen id, hero sha256, `safety_veto: true`, and
   `safety_veto_method` matching the arm. The file name is `<arm>-<mix>-<seed>.json`, and the
   seed is in the intent.

Reported only (warnings): B/D `veto_diagnostics` agree with the probe counters
(`screen.v6_identities_hold`: probe mirror, per-layer splits, and the link to the nested v5
counters); A/C `veto_diagnostics` are v5's and agree with their probe
(`v5screen.probe_identities_hold`).

## Compute cap and operations

- 276 episodes: 240 A/B, 24 C and 12 D.
- Expected time is about 2 to 2.5 h on CPU with 2 torch threads, assuming about 30 s per
  live v5 episode (the census simulation took 1,436 s for 48 episodes). v6 adds a few cell
  lookups per decision (and six one-frame move simulations only when an opponent head is
  within reach) and, on the ~1% `no_spacious` decisions, an escape search capped at 2000
  nodes.
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
  `snake-dqn-artifacts/apex-veto-v6-screen-20261002/run-v1`. Lane work and smokes stay
  outside `snake-dqn-artifacts`.
- Before GO: one smoke dry run (≤ 2 episodes × 500 frames, legacy path, smoke namespace)
  through `screen.py` with self-check and receipt, from the same source. Neither layer may
  trigger in a short smoke, so each layer's path is pinned by unit tests: analytic
  geometries checked against the live movement and collision code, and live tiny-world
  rollouts with each layer forced to trigger that must satisfy the probe identities and the
  B self-check, and replay identically.
- Test safety: every unit test that calls `screen.main`, `dev_screen.main` or another
  harness entry first replaces `tournament_eval.rollout` and `dev_screen.run_episode` with
  functions that raise (the v3 revision-4 runaway lesson).
- Recovery: one mechanical-defect recovery under `recovery-1/` with the same intent (same
  arms, seeds, metric and rule), as governance Tier 1 allows. Anything else needs a new
  namespace.

## Non-claims

- A `RECOMMEND_STRICT_GATE` is not promotion. v6 would still need a Tier-2 strict gate
  against v5 on fresh worlds with its own source-closure binding, plus a serving
  qualification of its own.
- A `NOT_ADVANCED` does not refute the census findings; it says only that these two layers,
  together, did not raise the mass integral detectably at this size. Giving up food races
  to greedy opponents (layer a) or delaying an enclosed death (layer b) may cost or gain
  little mass.
- The released configuration is unaffected. `src/evaluation/safety_veto.py` (v2, sha256
  `1b62d15c...c428`), `safety_veto_v3.py`, `safety_veto_v4.py` and `safety_veto_v5.py`
  (sha256 `d86d084e...ec86c`) stay byte-identical; tests assert all four. No dev_screen file
  changed.
