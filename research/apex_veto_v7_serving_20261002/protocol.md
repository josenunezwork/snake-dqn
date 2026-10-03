# Apex + v7 space-preference veto: web serving qualification (pre-registration)

Status: **pre-registered, written 2026-10-02 before any non-smoke serving episode.**
This lane copies the v5 web serving lane
(`research/apex_veto_v5_serving_20261001/`, SERVING_PASS at
`snake-dqn-artifacts/apex-veto-v5-serving-20261002/run-v1`) with the v7 identities swapped in:
the same episode counts, horizons, stepping order, dispatch path, stand-in, parity design and
pass criteria S1-S6. Differences are listed under "Changes from the v5 lane".
The sha256 of this file is pinned in code (`serving_run.PROTOCOL_SHA256` and
`serving_audit.PROTOCOL_SHA256`), not here. Any later edit must re-pin both, and a non-smoke
intent that names another sha cannot pass.
Harness: `serving_run.py`; audit: `serving_audit.py` (standard library only); shared record
shapes: `schema.py`. The harness imports the v2 lane's (`research/apex_veto_serving_20261001/`)
stepping, stand-in, dispatch, decision-counting and rollout-tracing helpers unchanged, as the
v5 lane did; the audit loads the v2 audit's evidence loader and counter-partition rule and the
v5 audit's v5-diagnostics rule by file path. The intent records the sha256 of every harness
file it ran, including the imported v2 and v5 ones.
Governance: [`docs/research/governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md).

## Question and decision it informs

Can the real web backend serve the champion behind the v7 wrapper (lambda 4.0) on the Watch
hero, selected by `SNAKE_SERVE_VETO_VARIANT=v7`, with every served episode bound to the v7
strict-gated identities, the wrapper demonstrably active on the hero alone, its counters and
v7 (and nested v5) diagnostics consistent with the frames on which it decided, behavior
identical to the v7 gate's rollout where the two are comparable, Play left unwrapped, and the
released v5 default and rollbacks intact? A PASS sets `serving_path_qualified = true` for the
v7 STRICT_PASS receipt (`apex-veto-v7-strict-20261002/run-v1/output/receipt.json`, sha256
`86ee36791e33…`; intent `3e39e0699ac1…`; audit report `9cfc7dbb7eaa…`), whose own serving
stage was the rollout harness, not this web path (closeout `serving_path_qualified=false`).
It changes no default, flag or deployment. Release stays a separate, explicit, user-approved
action.

This run measures serving correctness, not skill. It makes no mass, survival or score claim.

## Identities (pinned)

| Item | Value |
|---|---|
| Checkpoint | `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93` |
| Wrapper | `free-space-veto/v7-space-preference(lambda=4.0)`, descriptor = `SpacePreferenceVeto(4.0).descriptor()` |
| Wrapper sources | `src/evaluation/safety_veto_v7.py` `56ff7009…e2980`, `safety_veto_v5.py` `d86d084e…ec86c`, `safety_veto.py` `1b62d15c…c428`, `safety_veto_v3.py` `ed3a6d86…b5be1` (the v7 strict intent's candidate `source_sha256s`) |
| Served config | `configs/mechanics_v2.yaml`, sha256 `c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911` |
| Parity config | `research/apex_safety_20260926/deployment.yaml`, sha256 `4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5`, profile `promotion-v2-watch-rect` |

## Serving hook under test

`web/backend/safety_veto_serving.py`, called once at the end of every
`GameSession._build_impl`:

- `SNAKE_SERVE_VETO_WATCH_HERO` stays the master switch (unset = on, the released default;
  `0/false/no/off` = off). `SNAKE_SERVE_VETO_VARIANT` chooses the Watch hero's wrapper: unset or
  blank = `VARIANT_RELEASED_DEFAULT` (`v5` until a v7 release step), `v2`, `v5` or `v7`. Any
  other value falls back to `VARIANT_RELEASED_DEFAULT` and names the bad value in `reason` and
  the log line. Unit-tested; not part of this run.
- `v7` installs `install_space_preference_veto(snake, 4.0)` on the slot-0 Watch hero only, and
  only when the served checkpoint hashes to the pinned champion sha, all four source files hash
  to the pinned shas and the installed method is `free-space-veto/v7-space-preference(lambda=4.0)`
  (fail closed; otherwise unwrapped with a `reason`).
- The variant never affects Play. `SNAKE_SERVE_VETO_PLAY_AI` keeps its v2 semantics (opt-in,
  v2 wrapper) and is unset here, so Play is served unwrapped. Play AI wrapping is not part of
  the release.
- When a flag is requested, each build logs one INFO line starting `safety-veto-serving:` with
  the released v2 fields first in their released order, then `variant`, `variant_requested`,
  `wrapper_sources_match`, `wrapper_method`; `reason` is last.

The release environment every served build in this run reads is
`SNAKE_SERVE_VETO_VARIANT=v7` with the other two variables unset.

## Episodes and worlds

Seeds: `uint32_be(sha256("apex-veto-v7-web-serving-v1|<purpose>|<i>")[:4])` for purposes
`watch` (25), `play` (25) and `parity` (2). `serving_run.py` refuses to write the intent unless
they are unique and disjoint from: everything the v5 serving lane checked (which includes the
v2 serving lane's report); the first 1000 seeds of every domain/purpose in the v7 strict gate's
`EARLIER_DOMAINS` (including both web serving lanes and their smokes, the v3-v7 screens and
smokes, the v7 DEV sweep and the trap-horizon diagnostics); the first 1000 seeds of the v7
strict dev/final/serving banks and their smoke; the first 1000 seeds of the in-flight v8
domains (`apex-veto-v8-dev-v1`, `apex-veto-v8-dev-smoke-v1`, `apex-veto-v8-screen-v1`,
`apex-veto-v8-screen-smoke-v1`); and seeds 0..999. In-code check only (no namespace registry).
Smoke uses `apex-veto-v7-web-serving-smoke-v1`.

Counts: 25 Watch + 25 Play (50 served episodes) + 2 parity probes, the v5 lane's split
(accepted there under standing authority because only the Watch hero is wrapped by a release).
This lane was commissioned as "exactly as done for v5", so the split is the commissioned one.

1. **Watch, as served (25 episodes, 5000 frames each).** `set_seed(seed)`, then
   `GameSession(checkpoint)` under the release environment, stepped with `step()` then
   `snapshot()` (the order of `Hub.engine_loop`). Legacy Watch is all-respawn.
2. **Play (25 episodes).** One session built under the release environment and switched with a
   `set_mode` control. Each episode: `set_seed(seed)`, a `new_game` control, then the v2 lane's
   seeded stand-in sends `human_input` controls until the human dies (`run_over`) or 5000 frames.
   Every control goes through `web.backend.app._apply_control`; a control that sets
   `session.last_error` fails its episode.
3. **Parity probes (2, not served episodes, H5000).** Session side: `GameSession` on the parity
   config, release environment, hero made terminal (`auto_respawn=False`), stepped as in 1.
   Rollout side: `tournament_eval.rollout(champion, 5 x champion, 5000, seed,
   profile=promotion-v2-watch-rect, hero_safety_veto=True)` inside
   `dev_screen.hero_veto_installer(strict_run.install_candidate_veto)`, the v7 strict gate's
   candidate install (`install_space_preference_veto(hero, 4.0)`). Per frame both sides record
   frame number, each snake's id, head, length, alive and boosting flags, and the food count.

Comparability is as in the v2 and v5 lanes: served Watch differs from the gate profile in
config and hero lifecycle, so only parity runs the session code under the profile's config and
lifecycle.

Compute cap: 10800 s wall for the whole run (the v2/v5 lanes' cap). The v5 real run took
1512 s; v7 adds a capped area count on re-rank decisions, so the run is expected to take well
under an hour on one CPU slot.

## Receipts

`<out>/intent.json` is written before any episode, then `default_check.json`,
`records/<kind>-<NNN>.json` (kinds `watch_hero`, `play`), `parity/parity-<NNN>.json` and
`receipt.json`, which lists every file with its sha256. Files are create-only. Each episode
record (`schema.EPISODE_KEYS`) binds the env the build read, the checkpoint sha (file and
session), the wrapper identity as installed (method, descriptor, source path, source sha256 and
the four `source_sha256s`), the served identity, the veto block (`schema.VETO_KEYS`: variant,
scope, reason, flags, match flags, wrapped ids, per-snake and total counter deltas, per-snake and
total integer v7 diagnostics with their nested integer v5 diagnostics, and wall-clock timings),
the ids of snakes carrying any veto, and per wrapped snake the harness-observed decision frames.

## Pass criteria (all required; `serving_audit.py` decides)

- **S1 Completeness.** Outside a smoke: `intent.protocol_sha256` equals the pinned sha,
  `intent.git.dirty` is false and `intent.seed_report.disjoint` is true. `receipt.json` lists
  exactly 25 `watch_hero`, 25 `play` and 2 parity records plus `default_check.json`, every hash
  matches, every record is `complete` with `error` null, bound to the intent sha and schema, with
  the recipe seed. `intent.release_env` is the release environment. Watch steps exactly 5000
  frames (`horizon`); Play ends at `human_death` within 5000 frames or at `horizon` with 5000.
- **S2 Identity binding.** Intent and every Watch record carry exactly the pinned v7 wrapper
  identity (method, descriptor, source path, source sha256, all four `source_sha256s`); the
  intent's expected `v7_source_sha256s` are the pins and its expected v7 strict receipt sha is
  `86ee3679…`. Every record: checkpoint sha (file and session) is the champion, `env` is the
  release environment, `served` shows its mode, obs_spec `vector61`, `ApexPolicy`, training
  false, epsilon 0, `mechanics_v2.yaml` with the pinned sha and flags
  `{watch_hero: true, play_ai: false}`. Play records carry no wrapper identity.
- **S3 Scope and activity.** Every record: flags as above, `variant_requested` `v7`. Watch:
  active, scope `watch_hero`, reason null, variant `v7`, both match flags true, hero is slot 0,
  `wrapped_snake_ids == [hero_id]` and the hero is the only snake carrying a veto. Play:
  inactive, scope and variant null, reason `no serving veto flag applies to play mode`, nothing
  wrapped and no snake carrying a veto. Outside a smoke, **activity**: the Watch records' summed
  `veto.total.vetoes_applied` is greater than 0 (any decision whose executed action differs
  from the policy's masked argmax: a v7 re-rank change of a v5-kept base, or a v5 veto, possibly
  re-ranked). The split (`rerank_changes`, nested v5 `boost_landing_vetoes`, `vetoes_applied`)
  is reported, not gated.
- **S4 Counters.** Watch, per wrapped snake: the v2 counter partition and bounds; the v7
  identities (`decisions == kept + vetoes_applied + no_spacious`, each mirroring its v2-shaped
  counter; `v5.decisions == decisions`; `kept == v5.kept - rerank_changes_of_v5_kept`;
  `vetoes_applied == v5.vetoes_applied + rerank_changes_of_v5_kept`;
  `no_spacious == v5.no_spacious`; `rerank_decisions == rerank_pruned + rerank_scored`;
  `rerank_changes <= rerank_scored`; `rerank_changes_of_v5_kept`, `rerank_changes_boost_mode` and
  `rerank_changes_tail_release_driven` at most `rerank_changes`;
  `static_area_evaluations == 2 * rerank_changes`; `area_cap_hits <= area_evaluations`); the
  nested v5 diagnostics' own identities (the v5 lane's rule without its probe mirrors, since v7
  changes outcomes after v5); `decisions` equals the decision frames; totals are per-snake sums
  (`area_cap_max`: the per-snake maximum); decisions > 0. Play: no counters, diagnostics or
  decision frames (total 0).
- **S5 Parity.** Each probe: complete, env is the release environment, 5000 frames compared, no
  divergence, equal trace sha256, equal v2-shaped counters AND equal integer v7 diagnostics
  (nested v5 included), session variant `v7` and both sides' wrapper method
  `free-space-veto/v7-space-preference(lambda=4.0)`, session config the parity config, only snake
  0 wrapped, complete rollout record (smoke exempt), counters and diagnostics valid on both
  sides, session decisions equal decision frames. Reported, not gated: label
  `rerank-exercised` (a v7 re-rank changed an action), `exercised` (some replacement ran) or
  `non-exercising`.
- **S6 Released default and rollbacks intact.** `default_check.json`, built from real sessions:
  an empty environment reads flags `{watch_hero: true, play_ai: false}` and variant `v5`; with an
  empty environment and with `SNAKE_SERVE_VETO_VARIANT=v5`, Watch wraps only the hero with
  `free-space-veto/v5-boost-aware` and Play wraps nothing; with `SNAKE_SERVE_VETO_VARIANT=v2`,
  Watch wraps only the hero with `free-space-veto/v2-speed-preserving` and Play wraps nothing;
  with `SNAKE_SERVE_VETO_WATCH_HERO=0` and `SNAKE_SERVE_VETO_VARIANT=v7` nothing is wrapped in
  Watch or Play; `DEFAULT_CHECKPOINT` is the champion and the served config is the pinned one.
  (So this run must happen before a release step flips the default variant to v7.)

Reported, not gated: veto rates, Watch activity split, v7 mean apply cost, Play end reasons,
run frames, hero deaths, the parity label, and the current tree's v7 source shas.

Outcomes: `SERVING_PASS` (every rule passes, `serving_path_qualified = true`), `SERVING_FAIL`
(a rule fails on complete evidence), `INVALID` (evidence missing or unreadable). Recovery
follows the governance tiers: one audit-only re-audit of unchanged evidence for an audit defect
(unit-tested against real record shapes); any producer defect needs a new seed domain (`-v2`).
Failures are preserved and never relabeled.

## Changes from the v5 lane

- Wrapper: v7 (lambda 4.0), four bound sources instead of three; the gated method string
  carries the lambda, so a different lambda cannot match.
- Diagnostics: the v7 counter set (`SpacePreferenceCounters.to_dict()` integer keys) with the
  v5 counters nested under `v5`; S4/S5 check v7 identities plus the nested v5 shape rules.
- S6 checks the currently released v5 default (and the v2 and off rollbacks) instead of v2.
- Seed freshness adds the v7 strict gate's `EARLIER_DOMAINS` and banks and the v8 domains.
- Execution holds one shared CPU slot lock (`pqn-followup-20260909/cpu-slot-{1,2}.lock`) for the
  whole run and requires AC power at launch; these are launch conditions, not criteria.

## Dry run before GO

`serving_run.py --smoke --out <scratch>`: 1 Watch and 1 Play episode at 500 frames, 1 parity
probe at 200 frames (the rollout's H5000 accumulator cannot complete, so `record_complete`
false is accepted only when `intent.smoke`). Then `serving_audit.py --root <scratch> --out
<scratch>/audit`. The real run needs a clean tree, the pinned protocol and a fresh `--out`.

## Release step and rollback (not part of this run)

- **Release.** After a `SERVING_PASS`, with explicit user approval citing the v7 strict receipt
  and this audit: set `VARIANT_RELEASED_DEFAULT = VARIANT_V7` in
  `web/backend/safety_veto_serving.py` (and update its tests), then restart the server. Verify
  on stderr: `safety-veto-serving: active=True scope=watch_hero`, `wrapped_ids=[<hero id>]`,
  `strict_checkpoint_match=True`, `wrapper_source_sha256=56ff7009…e2980`, `variant=v7
  variant_requested=v7 wrapper_sources_match=True`,
  `wrapper_method=free-space-veto/v7-space-preference(lambda=4.0)`, `reason=None`.
- **Rollback to v5.** Restart with `SNAKE_SERVE_VETO_VARIANT=v5` (no code change).
- **Rollback to v2.** Restart with `SNAKE_SERVE_VETO_VARIANT=v2`.
- **Rollback to off.** Restart with `SNAKE_SERVE_VETO_WATCH_HERO=0`: no snake is wrapped.
- `SNAKE_SERVE_VETO_PLAY_AI` stays off. It has no strict evidence for any variant but v2.

## Non-claims

- No skill claim. The v7 strict gate measured one terminal wrapped hero against unwrapped
  opponents; served Watch respawns. Play under a wrapped AI is untested for v7 and not released.
- `set_hero` in Watch moves the inspector, not the wrapper.
- No change to `src/evaluation/strict_promotion.py`; schema `apex-veto-v7-web-serving/v1` is
  audited by `serving_audit.py` alone.
