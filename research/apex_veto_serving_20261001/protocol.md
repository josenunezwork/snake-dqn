# Apex + free-space veto: web serving qualification (pre-registration)

Status: **pre-registered, written 2026-10-01 before any non-smoke serving episode.**
Revised once, after the dry-run smoke and before any non-smoke episode: the compute cap
went from 3600 s to 10800 s, on the measured per-frame cost. No criterion changed. The
smoke intent therefore records the pre-revision `protocol_sha256`.
Harness: `serving_run.py`; audit: `serving_audit.py` (standard library only); shared
record shapes: `schema.py`. Governance:
[`docs/research/governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md).

## Question and decision it informs

Can the real web backend serve the champion behind the strict-gated wrapper, with every
served episode bound to the gated identities, the wrapper demonstrably active, its counters
consistent with the frames on which it decided, behavior identical to the gate's rollout
where the two are comparable, and nothing changed when the flags are off? A PASS sets
`serving_path_qualified = true` for the run-v3 STRICT_PASS receipt
(`receipt.json` sha256 `18b655196cf1…`). It changes no default, flag or deployment.
Release stays a separate, explicit, user-approved action (see "Release step").

This run measures serving correctness, not skill. It makes no mass, survival or score claim.

## Identities (pinned)

| Item | Value |
|---|---|
| Checkpoint | `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93` |
| Wrapper | `free-space-veto/v2-speed-preserving`, `src/evaluation/safety_veto.py` sha256 `1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428`, descriptor = `FreeSpaceVeto().descriptor()` |
| Served config | `configs/mechanics_v2.yaml`, sha256 `c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911` (what `_config_for(61)` serves) |
| Parity config | `research/apex_safety_20260926/deployment.yaml`, sha256 `4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5`, profile `promotion-v2-watch-rect` |

## Serving hooks under test (opt-in, default off)

`web/backend/safety_veto_serving.py`, called once at the end of every
`GameSession._build_impl`:

- `SNAKE_SERVE_VETO_WATCH_HERO=1`: Watch builds wrap the slot-0 hero only (the measured
  configuration: one wrapped hero, unwrapped opponents sharing the same policy).
- `SNAKE_SERVE_VETO_PLAY_AI=1`: Play builds wrap every AI snake, never the human. This is
  **not** the measured configuration; see non-claims.
- Only `1/true/yes/on` enable a flag. Train mode, raster checkpoints and humans are never
  wrapped. A flag that cannot take effect leaves the build unwrapped and says why in
  `GameSession.safety_veto_state()["reason"]`.

## Episodes and worlds

Seeds: `uint32_be(sha256("apex-veto-web-serving-v1|<purpose>|<i>")[:4])` for purposes
`watch` (1), `play` (49) and `parity` (2). `serving_run.py` refuses to write the intent unless
they are unique and disjoint from the strict-veto dev/final/serving banks v1-v3, the first
1000 seeds of `apex-safety-screen-v1` and the task-aligned challenger namespaces. The
governance namespace registry is not implemented, so this is an in-code check only. Smoke
uses the separate domain `apex-veto-web-serving-smoke-v1`.

1. **Watch, as served (1 episode, 5000 frames).** `set_seed(seed)`, then
   `GameSession(checkpoint)` with the Watch flag set in the environment. The game is stepped
   with `step()` followed by `snapshot()`, the order of `Hub.engine_loop`. Legacy Watch is
   all-respawn, so the hero respawns as it does for users.
2. **Play (49 episodes).** One session is built with the Play flag and switched with a
   `set_mode` control. Each episode runs `set_seed(seed)` and a `new_game` control. A seeded
   stand-in then sends `human_input` controls (named directions, no boost, a wall- and
   body-avoiding wanderer with turn probability 0.08) until the human dies (`run_over`) or
   5000 frames have been stepped. Every control goes through `web.backend.app._apply_control`,
   the browser's WebSocket dispatch, with one fixed connection id.
3. **Parity probes (2, not served episodes).** Same seed on both sides, H5000:
   - Session side: `GameSession` on the parity config (the module's 61-D config path is
     pointed at `deployment.yaml` for the probe only), Watch flag on, hero made terminal
     (`auto_respawn=False`, the profile's `hero_terminal`), stepped as in 1.
   - Rollout side: `tournament_eval.rollout(champion, 5 x champion, 5000, seed,
     profile=promotion-v2-watch-rect, hero_safety_veto=True)`, which is the gate's serving
     stage roster. A `BehaviorProbes` subclass records each frame without changing it.
   - Per frame, both sides record the frame number, each snake's id, head, length, alive and
     boosting flags, and the food count.

Comparability. The served Watch differs from the gate profile in config (`initial_food`
250 vs 300, plus the `pqn` identity block) and in hero lifecycle (respawn vs terminal), so
served Watch episodes are not compared to rollouts. Play has a human and has no rollout
analogue. Parity therefore runs the session code under the profile's config and lifecycle.

Compute cap: 10800 s wall for the whole run. The smoke measured 31 ms per served frame
(`step()` plus `snapshot()`, 500 frames in 15.7 s on CPU), so a 5000-frame episode takes
about 160 s. A parity probe takes about 180 s (the session side plus a roughly 20 s
rollout). The worst case, with every Play episode reaching 5000 frames, is about
50 x 160 + 2 x 180 = 8400 s. Play episodes that end at the human's death are shorter.

## Receipts

`<out>/intent.json` is written before any episode. Then `default_check.json`,
`records/<kind>-<NNN>.json` (one per served episode), `parity/parity-<NNN>.json` and
`receipt.json`, which lists every file with its sha256. Files are create-only. Each episode
record (`schema.EPISODE_KEYS`) binds:

- `checkpoint.sha256`, hashed by the harness from the file bytes, and
  `checkpoint.session_sha256`, which the session computed while loading;
- `wrapper`: method, descriptor, `source_path` and `source_sha256`, as installed;
- `served`: mode, config path and sha256, obs_spec, policy class, epsilon, world values,
  snake ids, hero and human ids, and the flags read at build;
- `veto`: active, scope, `wrapped_snake_ids`, and per-snake and total counter deltas over
  the episode (`FreeSpaceVeto.counters`);
- `decision_frames`: per wrapped snake, the number of advanced frames on which the
  harness saw it decide (alive before the step, alive after it, or in that frame's
  `frame_death_causes`, from the respawn-then-act order of `GameState.update`).

## Pass criteria (all required; `serving_audit.py` decides)

Each rule names the writer of the fields it checks.

- **S1 Completeness.** `receipt.json` lists exactly the intent's counts: 1 `watch_hero`
  record, 49 `play_ai` records and 2 parity records. Every listed file exists and its sha256
  matches. Every record has `status` `complete`, `error` null and `intent_sha256` equal to the
  intent file's sha256. Each record's seed equals the seed re-derived from the recipe.
  Watch steps exactly 5000 frames and ends at `horizon`. Play ends at `human_death` with at
  most 5000 frames, or at `horizon` with exactly 5000. (Writer: `serving_run.py`.)
- **S2 Identity binding.** In every record, `checkpoint.sha256` and `session_sha256` both equal
  the pinned checkpoint sha. `wrapper.method` and `wrapper.descriptor` equal the intent's
  `wrapper_identity`, and `wrapper.source_sha256` equals the pinned wrapper sha. `served`
  shows obs_spec `vector61`, policy `ApexPolicy`, training false and epsilon 0. Its config
  is `mechanics_v2.yaml` with the pinned sha, and its mode is `watch` (Watch) or `play`
  (Play). (Writers: `web/backend/safety_veto_serving.py` for `wrapper` and `session_sha256`,
  `GameSession` for `served`.)
- **S3 Wrapper active, right scope.** `veto.active` is true. Watch: scope `watch_hero` and
  `wrapped_snake_ids == [hero_id]`. Play: scope `play_ai`, and `wrapped_snake_ids` is
  every snake id except `human_id`, so `num_snakes - 1` of them. Flags: Watch
  `{watch_hero: true, play_ai: false}`, Play `{watch_hero: false, play_ai: true}`.
- **S4 Counters consistent with decision frames.** For every wrapped snake in every
  record: `decisions == kept_base + vetoes_applied + fallback_no_spacious`. Also
  `vetoes_to_boost`, `vetoed_base_boost` and `vetoes_speed_switched` are each at most
  `vetoes_applied`, every counter is at least 0, `decisions` equals that snake's
  `decision_frames` entry, and `total` is the per-snake sum. Watch `decisions > 0`. Each
  Play episode that stepped at least one frame has total `decisions > 0`. (Writer:
  `FreeSpaceVeto.apply` through `AISnake.update`, greedy path only. Epsilon 0 (S2) is what
  makes every decision greedy.)
- **S5 Parity.** For each probe: `status` `complete`, `frames_compared == 5000`,
  `first_divergence_frame` null, `trace_sha256_equal` true and `veto_counters_equal`
  true. The rollout record is complete (`record_complete`). The session-side hero's
  `decisions` equals its `decision_frames`.
- **S6 No default change.** `default_check.json` was built with both flags removed from the
  environment. `ServingVetoFlags.from_env({})` is all-off. A default-flag session wraps no
  snake in Watch, or in Play after a `set_mode` control. `DEFAULT_CHECKPOINT` is still the
  champion's basename, and the served 61-D config is still `mechanics_v2.yaml` with the
  pinned sha.

Reported, not gated: veto rates, Play end reasons, run frames, scores and hero deaths.

Outcomes: `SERVING_PASS`, which means every rule passes and `serving_path_qualified =
true`; `SERVING_FAIL`, when a rule fails on complete evidence; and `INVALID`, when evidence
is missing or unreadable. Recovery follows the governance tiers. If the producer completed
and only an audit rule is defective, one audit-only re-audit of the unchanged evidence is
allowed. That fix must be unit-tested against these real record shapes. Any producer
defect needs a new intent under a new seed domain (`-v2`).

## Dry run before GO

Run `serving_run.py --smoke --out <scratch>` first: 1 Watch and 1 Play episode at 500
frames, and 1 parity probe at 200 frames. In smoke, the rollout's H5000 accumulator cannot
complete, so the audit accepts `record_complete` false only when `intent.smoke` is true.
Then run `serving_audit.py --root <scratch> --out <scratch>/audit`. The audit's unit tests
use records captured from that smoke and one run-v3 serving record
(`fixtures/`). The real run needs explicit user authorization, and its `--out` is a fresh
directory.

## Release step and rollback (not part of this run)

- **Release.** After a `SERVING_PASS`, and with an explicit user approval that cites the
  run-v3 receipt and this audit, start the server with `SNAKE_SERVE_VETO_WATCH_HERO=1`, for
  example `SNAKE_SERVE_VETO_WATCH_HERO=1 ./venv/bin/python web/serve.py`. No code, config or
  champion file changes.
- **Rollback.** Restart without the variable, or with `=0`. Every build reads the flags, and
  with the flag off no snake is wrapped (S6). The champion file is never touched, so the
  incumbent is what is served.
- `SNAKE_SERVE_VETO_PLAY_AI` stays off at release. Wrapping every Play AI snake has no strict
  evidence.

## Non-claims

- No skill claim. The strict gate measured one wrapped hero against unwrapped opponents.
  The Play flag wraps 5 of 6 snakes, and its effect on the human's experience is untested.
- The served Watch hero respawns, while the gate's hero is terminal. The wrapper decision
  rule is the same, but the gate's H5000 mass numbers describe the terminal-hero profile
  only.
- `set_hero` in Watch moves the inspector, not the wrapper. The wrapper stays on the slot-0
  snake that the build wrapped.
- No change to `src/evaluation/strict_promotion.py`. Its `_serving` still accepts only
  raster31v3 receipts. This run's schema `apex-veto-web-serving/v1` is audited by
  `serving_audit.py` alone.
