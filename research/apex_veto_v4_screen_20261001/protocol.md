# Tier-1 screen: Apex + v4 look-ahead veto vs Apex + v2 veto (pre-registration)

Label: **screen (non-authoritative)**. Governance: Tier 1 in
`docs/research/governance_tiers_2026-09-26.md`. Nothing here can change a default, the
champion, a deployment profile or the released Watch-hero veto. Written 2026-10-01, before
any screen episode or smoke of this screen ran.

Revision 1 (2026-10-01, before any screen episode; one 2-episode plumbing smoke on the
smoke namespace had run from the pre-fix commit 9259e29, with 0 v4 searches): the no-escape and
budget fallbacks now defer to the released v2 veto instead of keeping the base action
(review finding: v4 could keep a certain-death dead end that v2 vetoes when the v2
replacement's step-1 cell is next to another snake's head, or when the budget ran out
after the base was proven escape-less). B therefore never drops v2's protection in an
untriggered or fallback state, so a B loss cannot be explained by lost v2 protection.

- `screen_id`: `apex-veto-v4-screen-v1`
- Harness: `research/apex_safety_20260926/dev_screen.py`, driven by `screen.py` beside this
  file (`ScreenSpec` `SPEC`). Output: `intent.json` (before any episode), `records/`,
  `events.jsonl`, `summary.json`, then `receipt.json` (written by `screen.py`).
- Owner: Apex safety lane. `intent.json` records the git SHA and dirty paths, this file's
  `protocol_sha256`, the owner, hypothesis, decision informed, primary metric, estimator,
  decision rule and compute cap (`compute_cap`: planned episodes, deadline, 4 h maximum),
  all before any episode runs.

## Question and the decision it informs

Does replacing the released v2 free-space veto (`free-space-veto/v2-speed-preserving`,
STRICT_PASS in `apex-veto-strict-20260927` run-v3, receipt sha256 `18b655196cf1...d704c1`;
SERVING_PASS for the Watch hero) with the v4 look-ahead veto
(`free-space-veto/v4-lookahead`, `src/evaluation/safety_veto_v4.py`) raise the Apex
champion's H5000 mass integral? The decision informed: design a Tier-2 strict gate for v4
against v2 (`RECOMMEND_STRICT_GATE`) or stop (`NOT_ADVANCED`).

Motivation (from the v3 screen, `docs/experiments/apex_veto_serving_and_v3_2026-10-01`):
v3's tail-aware one-step count changed the decision in only 4 of 120 worlds. The remaining
self-collision deaths occur after the safe-and-spacious set is already empty; the trap is
entered several moves before any one-step reachability check can see it. v4 asks a
different question: is there a sequence of the snake's own next `depth` moves (default 8)
that stays passable and ends in a cell whose tail-aware reachable count is at least `need`?
A one-step count measures area, not traversability: a 1-wide dead end of `need + 3` cells
passes v2 and v3 and is fatal.

## v4 rule (defaults are the arm-B configuration)

Full specification in the module docstring; summary:

- **Trigger.** Search only when the base action's v2 one-step count is below
  `min(2 * need, cap)`; otherwise keep the base action (v4 = v2 there). The `cap` bound is
  a deliberate reading of "2x need": the count saturates at `cap`, so for snakes of length
  `>= cap / 2` the literal `2 * need` would trigger on every decision.
- **Escape** for a direction: depth-first search over the snake's own moves (3 relative
  turns per step, one cell per step) to depth 8. Own body releases time-indexed from the
  tail (v3's model, slack 1). Other snakes' bodies are static walls. At step 1 the cells
  4-adjacent to another live snake's head are blocked. Walls are blocked. Cells already on
  the search path stay blocked. The escape test `count >= need` is applied at the leaf
  (depth 8). Each node at step `k` must have a count of at least `need + 8 - k`; this is a
  bound prune that is exact for static occupancy. Applying `count >= need` at depth 1
  instead would make v4 accept everything v3 accepts, so it could never veto a move v3
  allows. That reading of "or any node" was therefore rejected.
- **Veto.** Keep the base action if its direction escapes. Otherwise choose the highest-Q
  masked-legal escaping action in the base action's speed mode, else in the other mode
  (v2's `veto_choice`). A boost is scored by its direction's escape: its first step is the
  search's first step.
- **v2 fallback.** If no masked-legal action escapes (`fallback_no_escape`), the decision
  is exactly v2's: `veto_choice` with v2's one-step `spacious_directions`. Untriggered
  decisions already equal v2's (the base count is at least `need`), so B is at least as
  protective as the released v2 everywhere except where the escape search itself keeps or
  vetoes.
- **Budget.** At most 4000 nodes per decision. If the budget runs out (`budget_exhausted`),
  the v2 fallback decides. Child order is fixed, so every result is deterministic.
- The probe (`probes.safety_veto`) is the descriptor plus exactly v2's seven counters,
  counting the action taken. The v4 counters are stored in each B/D entry's
  `veto_diagnostics`; they split each fallback by the v2 outcome (`fallback_v2_kept`,
  `fallback_v2_vetoes`, `fallback_v2_no_spacious`).

## Arms

| Arm | Hero | Veto |
|---|---|---|
| A | `champion_a5_freespace_20260621.pth` (sha256 `43d4e2c5...d747ac93`) | v2 via `rollout(hero_safety_veto=True)`, the released configuration |
| B | same checkpoint | v4, defaults: depth 8, node budget 4000, trigger factor 2, tail slack 1 |
| C | same as A | determinism control: A repeated on the first 8 worlds per mix, after all A/B pairs. Each C record must equal its A record exactly. |
| D | same as B | B replay control: B repeated on the first 4 worlds per mix, after all C episodes. Each D record must equal its B record exactly. |

Opponent pool, roster construction and checkpoint snapshots are the dev_screen ones (strict
balanced rosters).

## Worlds

- Profile `promotion-v2-watch-rect`, horizon 5000, digest `d396d3ed...0e8b`. Config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3...715aa5`).
- Mixes `frozen`, `scripted`, `mixed`. **40 worlds per mix**, the same seeds in every mix.
- Seeds: `uint32(sha256("apex-veto-v4-screen-v1|worlds|<i>")[:4])`, i = 0..39.
- Construction-time disjointness (fail closed, in code) against:
  - the task-aligned challenger namespaces, the strict pilot's observed seeds and seeds
    0..999 (dev_screen checks);
  - the first 1000 seeds of `apex-safety-screen-v1/worlds`;
  - `apex-veto-strict-{dev,final,serving}-v{1,2,3}/worlds` and `apex-veto-strict-smoke-v1`;
  - `apex-veto-web-serving-v1` and `apex-veto-web-serving-smoke-v1` (`watch`, `play`,
    `parity`);
  - all v3 screen banks: `apex-veto-v3-screen-v1` (consumed and quarantined),
    `apex-veto-v3-screen-v2` (the real v3 run) and `apex-veto-v3-screen-smoke-v1`;
  - this screen's smoke domain, `apex-veto-v4-screen-smoke-v1`.

  The strict pilot recipe and roster parity checks must run and pass.
- Smoke dry runs use `apex-veto-v4-screen-smoke-v1` only, never these 40 worlds.

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
  - per-mix means, wins and losses; survival fraction Δ; death causes;
  - veto counters for A and B;
  - the v4 counters and cost (`receipt.json` `reported.v4_cost`: searches, vetoes,
    budget exhaustions, no-escape fallbacks and their v2 split, nodes; mean wall seconds
    per decision and per search; mean B episode wall time against A's);
  - the mix-stratified mean of means with a 90% Welch CI and clear-loser flags;
  - random-effects and crossed pools;
  - sizing for a later Tier-2 design.

  Wording: say "harm" or "no effect" only when the CI excludes the effect of interest;
  otherwise say "not established".

## Self-checks (gating) and their writers

`screen.py` runs these on every record after the run. They are tested in
`tests/test_safety_veto_v4.py`.

1. `probes.safety_veto` equals the arm's descriptor plus exactly the seven v2 counters, with
   `decisions = kept_base + vetoes_applied + fallback_no_spacious`. Writer: `rollout`
   attaches `veto.record()`. Checked with `strict_promotion._validate_candidate_wrapper_probe`.
2. `record.seed` equals the planned world seed, `world_identity` equals the materialized
   roster's identity, and `evaluation_profile_digest` is the profile's. Writer:
   `rollout(profile=...)`.
3. `denominators.scored_frames` = 5000 and `counters.decisions` =
   `denominators.decision_frames`. Writers: `EvaluationMetricsAccumulator`, and
   `AISnake.update` → `veto.apply`.
4. Entry: schema `apex-veto-v4-screen/v1`, screen id, hero sha256, `safety_veto: true`, and
   `safety_veto_method` matching the arm. The file name is `<arm>-<mix>-<seed>.json`, and the
   seed is in the intent.

Reported only (warnings): whether each B/D entry's `veto_diagnostics` agrees with its probe
counters: equal `decisions`, `kept_base = kept_untriggered + kept_escape +
fallback_v2_kept`, `vetoes_applied = vetoes_applied (v4) + fallback_v2_vetoes` and
`fallback_no_spacious = fallback_v2_no_spacious` (`screen.probe_identities_hold`).

## Compute cap and operations

- 276 episodes: 240 A/B, 24 C and 12 D.
- Expected time is about 2 h on CPU with 2 torch threads. This assumes about 21 s per v2
  episode (the run-v3 mean) plus v4 search overhead. The search runs only on triggered
  decisions, and the unit-test geometries take about 1 ms per search.
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
  `snake-dqn-artifacts/apex-veto-v4-screen-20261001/run-v1`. Lane work and smokes stay
  outside `snake-dqn-artifacts`.
- Before GO: one smoke dry run (≤ 2 episodes × 500 frames, legacy path, smoke namespace)
  through `screen.py` with self-check and receipt, from the same source. The default
  trigger rarely fires in a short smoke, so the search path itself is pinned by a unit
  test: a live tiny-world rollout with the trigger forced on every decision must search,
  satisfy the probe identities and the B self-check, and replay identically.
- Test safety: every unit test that calls `screen.main`, `dev_screen.main` or another
  harness entry first replaces `tournament_eval.rollout` and `dev_screen.run_episode` with
  functions that raise. This exists because of the v3 revision-4 runaway test.
- Recovery: one mechanical-defect recovery under `recovery-1/` with the same intent (same
  arms, seeds, metric and rule), as governance Tier 1 allows. Anything else needs a new
  namespace.

## Non-claims

- A `RECOMMEND_STRICT_GATE` is not promotion. v4 would still need a Tier-2 strict gate on
  fresh worlds with its own source-closure binding, plus a serving qualification of its own.
- The released v2 configuration is unaffected. `src/evaluation/safety_veto.py` (sha256
  `1b62d15c...c428`) and `src/evaluation/safety_veto_v3.py` stay byte-identical; tests
  assert both. The dev_screen additions (`ScreenSpec.replay_worlds`, the `replay_control`
  summary block) are opt-in and default off. With the default `replay_worlds = 0`, the
  original screen and the v3 screen produce the same plans, intents, decision rule text and
  summaries as before.
