# Tier-1 screen: Apex + v3 tail-aware veto vs Apex + v2 veto (pre-registration)

Label: **screen (non-authoritative)**. Governance: Tier 1 in
`docs/research/governance_tiers_2026-09-26.md`. Nothing here can change a default, the
champion, or a deployment profile. Written 2026-10-01, before any screen episode ran.
Revised once after review, still before any screen episode: the growth-gap caveat and its
reported diagnostic, the refusal of non-pre-registered sizes, the 4 h wall-time cap, and
the Tier-1 intent fields below. Arms, worlds, metric and decision rule are unchanged.

- `screen_id`: `apex-veto-v3-screen-v1`
- Harness: `research/apex_safety_20260926/dev_screen.py`, driven by `screen.py` beside this
  file (`ScreenSpec` `SPEC`). Output: `intent.json` (before any episode), `records/`,
  `events.jsonl`, `summary.json`, then `receipt.json` (written by `screen.py`).
- Owner: Apex safety lane. The git SHA and dirty paths are recorded in `intent.json`, with
  this file's `protocol_sha256`, the owner, hypothesis, decision it informs, primary
  metric, estimator, decision rule and compute cap (`compute_cap`: planned episodes,
  deadline and the 4 h maximum), all written before any episode runs.

## Question and the decision it informs

Does replacing the v2 free-space veto (`free-space-veto/v2-speed-preserving`, STRICT_PASS in
`apex-veto-strict-20260927` run-v3, receipt sha256 `18b655196cf1...d704c1`) with the v3
tail-aware veto (`free-space-veto/v3-tail-aware`, `src/evaluation/safety_veto_v3.py`) raise
the Apex champion's H5000 mass integral? The decision informed: whether to design a Tier-2
strict gate for v3 against the v2 configuration (`RECOMMEND_STRICT_GATE`) or stop
(`NOT_ADVANCED`).

Motivating Tier-0 diagnostic (read-only, saved records; run before this protocol). Inputs:
the 705 `final-candidate-*.json` records of run-v3 (`output/final/records/`, sha256 of the
sorted per-file sha256 digests `00dbea3e...00ef4a1a`). Findings: 593 of 705 episodes end in
self-collision; **all 593** had at least one `fallback_no_spacious` decision (v2 found no
spacious move and kept the policy's action), against 31 of the other 112. Vetoes are rare
(0.09% of decisions); no-spacious fallbacks are 0.81%. v2 blocks the whole body, so a coiled
snake that could follow its own tail out is reported as trapped. v3 counts own-body cells as
passable once the search distance reaches the steps until they vacate. With one-step scoring
it is never stricter than v2: it can only keep a base move v2 would veto, or turn a v2
no-spacious fallback into an informed choice.

Known permissive approximation (the growth gap). v3's release model allows for one pellet
of growth (`tail_release_slack=1`). Every pellet eaten on the way delays every later tail
release by one more step (`Snake.grow` adds length, and `move()` skips the pop). The
profile has 300 initial food plus corpse and trail pellets, so a path that follows the tail
through dense food can be counted open while the real tail is still there. In that one
direction v3 is less conservative than its own model, on top of being more permissive than
v2 by design: "never stricter than v2" also means it can admit a pocket that closes. The
screen measures the net effect; the diagnostic below separates the two cases.

## Arms

| Arm | Hero | Veto |
|---|---|---|
| A | `champion_a5_freespace_20260621.pth` (sha256 `43d4e2c5...d747ac93`) | v2 via `rollout(hero_safety_veto=True)`, the passed configuration |
| B | same checkpoint | v3, default options: `tail_release_slack=1`, one-step boost scoring (`boost_two_step=False`), v2 cap/need and replacement rule. v2-comparison diagnostics on (they never change an action). |
| C | same as A | determinism control: A repeated on the first 8 worlds per mix, after all A/B pairs. Each C record must equal its A record exactly. |

The opt-in `boost_two_step` refinement is **not** under test. Opponent pool, roster
construction and checkpoint snapshots are the dev_screen ones (strict balanced rosters).

## Worlds

- Profile `promotion-v2-watch-rect`, horizon 5000, digest `d396d3ed...0e8b`. Config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3...715aa5`).
- Mixes `frozen`, `scripted`, `mixed`. **40 worlds per mix**, the same seeds in every mix.
- Seeds: `uint32(sha256("apex-veto-v3-screen-v1|worlds|<i>")[:4])`, i = 0..39.
- Construction-time disjointness (fail closed, in code; the governance registry is not
  implemented yet) against: the task-aligned challenger namespaces and the strict pilot's
  observed seeds, plus seeds 0..999 (dev_screen checks); the first 1000 seeds of
  `apex-safety-screen-v1/worlds`; `apex-veto-strict-{dev,final,serving}-v{1,2,3}/worlds` and
  `apex-veto-strict-smoke-v1/worlds`; `apex-veto-web-serving-v1` and its smoke domain
  (`watch`, `play`, `parity`); and this screen's smoke domain `apex-veto-v3-screen-smoke-v1`.
  The strict pilot recipe and roster parity checks must run and pass.
- Smoke dry runs use `apex-veto-v3-screen-smoke-v1` only, never these 40 worlds.

## Primary metric, estimator, decision rule

- Primary: paired Δ `mass_integral` (B − A) per world, per mix.
- Test: one-sided paired t test per mix (`paired_delta_test`), Holm across the three mixes at
  α = 0.05 (`holm_three_mix_superiority`, 2 of 3 required).
- Decision (computed by `dev_screen.summarize`; `screen.py` adds the self-check override):
  - `NON_PREREGISTERED_DESIGN` unless worlds per mix = 40 and determinism worlds = 8;
  - `INCOMPLETE` if any planned A/B pair or C episode is missing;
  - `INVALID_NONDETERMINISTIC` if any C record differs from its A record;
  - `RECOMMEND_STRICT_GATE` if Holm rejects H0 in at least 2 of 3 mixes;
  - `NOT_ADVANCED` otherwise;
  - `INVALID_SELF_CHECK_FAILED` (receipt only) if any record fails a gating self-check.
- Reported, not used for the decision: per-mix means, wins and losses; survival fraction Δ;
  death causes; veto counters for A and B; v3-vs-v2 diagnostics (`veto_diagnostics` per B
  entry); the mix-stratified mean of means with a 90% Welch CI and clear-loser flags
  (`screen_stats.stratified_mean_of_means`); random-effects and crossed pools; sizing for a
  later Tier-2 design. Wording: "harm" or "no effect" only when the CI excludes the effect
  of interest, otherwise "not established".

## Self-checks (gating) and their writers

Run by `screen.py` on every record after the run. Tested in `tests/test_safety_veto_v3.py`
against real run-v3 records (`fixtures/`) and live rollout output.

1. `probes.safety_veto` equals the arm's descriptor plus exactly the seven v2 counters, with
   `decisions = kept_base + vetoes_applied + fallback_no_spacious`. Writer: `rollout` attaches
   `veto.record()`. Checked with `strict_promotion._validate_candidate_wrapper_probe`.
2. `record.seed` equals the planned world seed. `world_identity` equals the materialized
   roster's identity. `evaluation_profile_digest` is the profile's. Writer: `rollout(profile=...)`.
3. `denominators.scored_frames` = 5000 and `counters.decisions` = `denominators.decision_frames`.
   Writers: `EvaluationMetricsAccumulator` and `AISnake.update` → `veto.apply`, once per frame
   the hero acts. This held in all 705 run-v3 candidate records.
4. Entry: schema `apex-veto-v3-screen/v1`, screen id, hero sha256, `safety_veto: true`, and
   `safety_veto_method` matching the arm. The file name is `<arm>-<mix>-<seed>.json`, and the
   seed is in the intent.

Reported only: whether B's `veto_diagnostics.decisions` equals its counters, and the
closing-pocket diagnostic (`receipt.json` `reported.v3_closing_pocket`, per mix). A B
decision is *tail-admitted* when its executed direction is spacious under v3 but not under
v2's static count. A B episode is `admitted_closing_pocket` when the hero died by
self-collision (`probes.death_cause == "self"`) with a tail-admitted decision in its last
50 decisions (the hero is terminal and decides once per frame alive, so about 50 frames).
It is `rescued` when it had tail-admitted decisions and no such death. This never gates
the decision.

## Compute cap and operations

- 264 episodes (240 A/B + 24 C). At about 21 s per v2 episode (run-v3 mean) plus v3 search
  overhead, expect about 1.6–2 h on CPU with 2 torch threads.
- Cap: at most 4 h (14400 s) of wall time. `screen.py` refuses a `--deadline-utc` more than
  4 h after launch, and the run stops at that deadline. The harness does not start an
  episode with less than max(45 s, 2 × the mean episode time) remaining. A deadline stop
  gives `INCOMPLETE`.
- Sizes: outside a smoke, `screen.py` refuses (exit 2, before any world is played) any
  `--worlds-per-mix`/`--determinism-worlds` other than 40/8. A smaller look at these worlds
  would otherwise be recorded and a later 40/8 run could still reach a decision after it.
- The screen holds one shared CPU slot lock (`cpu-slot-{1,2}.lock` under
  `snake-dqn-artifacts/pqn-followup-20260909`, opened read-only, never created). It takes the
  lock before `--out` exists and holds it until `summary.json`. It refuses to start on
  battery (macOS `pmset`). The real run writes only under its own root,
  `snake-dqn-artifacts/apex-veto-v3-screen-20261001/run-v1` (outside
  `ongoing-research-20260913`, which the harness refuses). Lane work and smokes stay
  outside `snake-dqn-artifacts`.
- Before GO: one smoke dry run (≤ 2 episodes × 500 frames, legacy path, smoke namespace)
  through `screen.py` with self-check and receipt, from the same source.
- Recovery: one mechanical-defect recovery under `recovery-1/` with the same intent (same
  arms, seeds, metric and rule), as governance Tier 1 allows. Anything else needs a new
  namespace.

## Non-claims

- A `RECOMMEND_STRICT_GATE` is not promotion. v3 would still need a Tier-2 strict gate on
  fresh worlds with its own source-closure binding. It would also need a serving receipt
  schema that admits a vector61 + wrapper candidate; `strict_promotion._serving` accepts only
  raster31v3 today.
- The v2 receipt is unaffected. `src/evaluation/safety_veto.py` is byte-identical (sha256
  `1b62d15c...c428`, asserted by a test). dev_screen's original screen (`DEFAULT_SPEC`)
  writes the same records, intent and summary as before.

## Revision 3 (pre-run, 2026-10-01; no episode had been run)

Two changes were made at the final verifier's request, before any real-world
episode:

1. **Output root.** The earlier text forbade any output under
   `snake-dqn-artifacts`, which contradicted the operator root. The real run now
   writes to `snake-dqn-artifacts/apex-veto-v3-screen-20261001/run-v1`.
2. **Wall cap.** Raised from 3 h to 4 h. The only full-size precedent, the
   original 264-episode screen, needed more than 3 h under host contention.

Arms, seeds, metric, sizes and the decision rule are unchanged.
