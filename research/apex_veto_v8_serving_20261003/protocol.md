# Apex + v8 space-and-head veto: web serving qualification (pre-registration)

Status: **pre-registered, written 2026-10-03 before any non-smoke serving episode.**
This lane copies the v7 web serving lane
(`research/apex_veto_v7_serving_20261002/`, SERVING_PASS at
`snake-dqn-artifacts/apex-veto-v7-serving-20261002/run-v1`) with the v8 identities swapped in:
the same episode counts, horizons, stepping order, dispatch path, stand-in, parity design and
pass criteria S1-S6. Differences are listed under "Changes from the v7 lane".
The sha256 of this file is pinned in code (`serving_run.PROTOCOL_SHA256` and
`serving_audit.PROTOCOL_SHA256`), not here. Any later edit must re-pin both, and a non-smoke
intent that names another sha cannot pass.
Harness: `serving_run.py`; audit: `serving_audit.py` (standard library only); shared record
shapes: `schema.py`. The harness imports the v2 lane's (`research/apex_veto_serving_20261001/`)
stepping, stand-in, dispatch, decision-counting and rollout-tracing helpers unchanged, as the
v5 and v7 lanes did; the audit loads the v2 audit's evidence loader and counter-partition rule
and the v7 audit's v7-diagnostics rule (which applies the v5 lane's v5 rule to its nested v5
block) by file path. The intent records the sha256 of every harness file it ran, including the
imported v2, v5 and v7 ones and the v8 strict `spec.py` whose candidate installer the parity
probe uses.
Governance: [`docs/research/governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md).

## Question and decision it informs

Can the real web backend serve the champion behind the v8 wrapper (lambda 8.0, head layer on,
no diagnostic reference lambda) on the Watch hero, selected by `SNAKE_SERVE_VETO_VARIANT=v8`,
with every served episode bound to the v8 strict-gated identities, the wrapper demonstrably
active on the hero alone, its counters and v8 diagnostics (v7 and v5 nested, v7's own probe
counters) consistent with the frames on which it decided, behavior identical to the v8 gate's
rollout where the two are comparable, Play left unwrapped, and the released v7 default and
rollbacks intact? A PASS sets `serving_path_qualified = true` for the v8 STRICT_PASS receipt
(`apex-veto-v8-strict-20261003/run-v1/output/receipt.json`, sha256 `29b1f7f6f095…`; intent
`ca4aa9707445…`; audit report `72410ec57530…`), whose strict package had no web serving stage.
It changes no default, flag or deployment. Release stays a separate, explicit, user-approved
action.

This run measures serving correctness, not skill. It makes no mass, survival or score claim.

## Identities (pinned)

| Item | Value |
|---|---|
| Checkpoint | `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93` |
| Wrapper | `free-space-veto/v8-space-and-head(lambda=8.0)`, descriptor = `SpaceAndHeadVeto(8.0).descriptor()` (`head_avoidance: true`) = the v8 strict intent's `arms.candidate.descriptor` |
| Wrapper sources | the v8 strict intent's `arms.candidate.source_sha256s`: `safety_veto_v8.py` `faf3695f…ac05`, `safety_veto_v7.py` `56ff7009…e2980`, `safety_veto_v6.py` `a6117cb9…57d5`, `safety_veto_v5.py` `d86d084e…ec86c`, `safety_veto_v4.py` `3f0881af…4578`, `safety_veto_v3.py` `ed3a6d86…b5be1`, `safety_veto.py` `1b62d15c…c428` |
| Served config | `configs/mechanics_v2.yaml`, sha256 `c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911` |
| Parity config | `research/apex_safety_20260926/deployment.yaml`, sha256 `4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5`, profile `promotion-v2-watch-rect` |

## Serving hook under test

`web/backend/safety_veto_serving.py`, called once at the end of every
`GameSession._build_impl`:

- `SNAKE_SERVE_VETO_WATCH_HERO` stays the master switch (unset = on, the released default;
  `0/false/no/off` = off). `SNAKE_SERVE_VETO_VARIANT` chooses the Watch hero's wrapper: unset or
  blank = `VARIANT_RELEASED_DEFAULT` (`v7` until a v8 release step), `v2`, `v5`, `v7` or `v8`.
  Any other value falls back to `VARIANT_RELEASED_DEFAULT` and names the bad value in `reason`
  and the log line. Unit-tested; not part of this run.
- `v8` installs `install_space_and_head_veto(snake, 8.0)` (the v8 strict candidate's install:
  head layer on, `reference_lambda=None`) on the slot-0 Watch hero only, and only when the served
  checkpoint hashes to the pinned champion sha, all seven source files hash to the pinned shas,
  the installed method is `free-space-veto/v8-space-and-head(lambda=8.0)`, the descriptor has
  `head_avoidance: true`, and the installed veto has the head layer on and no reference lambda
  (fail closed; otherwise unwrapped with a `reason`).
- The variant never affects Play. `SNAKE_SERVE_VETO_PLAY_AI` keeps its v2 semantics (opt-in,
  v2 wrapper) and is unset here, so Play is served unwrapped. Play AI wrapping is not part of
  the release.
- When a flag is requested, each build logs one INFO line starting `safety-veto-serving:` with
  the released v2 fields first in their released order, then `variant`, `variant_requested`,
  `wrapper_sources_match`, `wrapper_method`; `reason` is last.

The release environment every served build in this run reads is
`SNAKE_SERVE_VETO_VARIANT=v8` with the other two variables unset.

## Episodes and worlds

Seeds: `uint32_be(sha256("apex-veto-v8-web-serving-v1|<purpose>|<i>")[:4])` for purposes
`watch` (25), `play` (25) and `parity` (2). `serving_run.py` refuses to write the intent unless
they are unique and disjoint from: everything the v7 serving lane checked (which includes the
v5 and v2 serving lanes' reports, the v7 strict gate's `EARLIER_DOMAINS` and banks and the v8
DEV/screen domains); the v8 strict gate's `spec.excluded_seeds()` (first 1000 seeds of every
domain/purpose in its `EARLIER_DOMAINS`, including the v7 web serving lane and its smoke, the
challenger namespaces, the strict pilot's observed seeds, seeds 0..999 and the seeds saved in
every earlier v7/v8 artifact); the first 1000 seeds of the v8 strict banks
(`apex-veto-v8-strict-dev-v1`, `apex-veto-v8-strict-final-v1`); and every seed in the closed v8
strict run's intent `banks` and `output/rosters.json`. In-code check only (no namespace
registry). Smoke uses `apex-veto-v8-web-serving-smoke-v1`.

Counts: 25 Watch + 25 Play (50 served episodes) + 2 parity probes, the v5/v7 lanes' split.
This lane was commissioned as "mirroring exactly how v7 was qualified", so the split is the
commissioned one.

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
   `dev_screen.hero_veto_installer(spec.install_candidate)`, the v8 strict gate's candidate
   install (`install_space_and_head_veto(hero, 8.0)`). Per frame both sides record frame
   number, each snake's id, head, length, alive and boosting flags, and the food count.

Comparability is as in the v2, v5 and v7 lanes: served Watch differs from the gate profile in
config and hero lifecycle, so only parity runs the session code under the profile's config and
lifecycle.

Compute cap: 10800 s wall for the whole run (the earlier lanes' cap). The v7 real run took
1650 s; v8 adds a head check per decision, so the run is expected to take well under an hour
on one CPU slot.

## Receipts

`<out>/intent.json` is written before any episode, then `default_check.json`,
`records/<kind>-<NNN>.json` (kinds `watch_hero`, `play`), `parity/parity-<NNN>.json` and
`receipt.json`, which lists every file with its sha256. Files are create-only. Each episode
record (`schema.EPISODE_KEYS`) binds the env the build read, the checkpoint sha (file and
session), the wrapper identity as installed (method, descriptor, source path, source sha256 and
the seven `source_sha256s`), the served identity, the veto block (`schema.VETO_KEYS`: variant,
scope, reason, flags, match flags, wrapped ids, per-snake and total counter deltas, per-snake and
total integer v8 diagnostics with `head_avoidance`, `reference_lambda`, v7's integer diagnostics
(v5 nested) and v7's probe counters, and wall-clock timings), the ids of snakes carrying any
veto, and per wrapped snake the harness-observed decision frames.

## Pass criteria (all required; `serving_audit.py` decides)

- **S1 Completeness.** Outside a smoke: `intent.protocol_sha256` equals the pinned sha,
  `intent.git.dirty` is false and `intent.seed_report.disjoint` is true. `receipt.json` lists
  exactly 25 `watch_hero`, 25 `play` and 2 parity records plus `default_check.json`, every hash
  matches, every record is `complete` with `error` null, bound to the intent sha and schema, with
  the recipe seed. `intent.release_env` is the release environment. Watch steps exactly 5000
  frames (`horizon`); Play ends at `human_death` within 5000 frames or at `horizon` with 5000.
- **S2 Identity binding.** Intent and every Watch record carry exactly the pinned v8 wrapper
  identity (method, descriptor incl. `head_avoidance: true`, source path, source sha256, all
  seven `source_sha256s`); the intent's expected `v8_source_sha256s` are the pins, its expected
  v8 strict receipt, intent and audit-report shas are `29b1f7f6…`, `ca4aa970…` and `72410ec5…`,
  and it names head layer on and no reference lambda. Every record: checkpoint sha (file and
  session) is the champion, `env` is the release environment, `served` shows its mode, obs_spec
  `vector61`, `ApexPolicy`, training false, epsilon 0, `mechanics_v2.yaml` with the pinned sha and
  flags `{watch_hero: true, play_ai: false}`. Play records carry no wrapper identity.
- **S3 Scope and activity.** Every record: flags as above, `variant_requested` `v8`. Watch:
  active, scope `watch_hero`, reason null, variant `v8`, both match flags true, hero is slot 0,
  `wrapped_snake_ids == [hero_id]` and the hero is the only snake carrying a veto. Play:
  inactive, scope and variant null, reason `no serving veto flag applies to play mode`, nothing
  wrapped and no snake carrying a veto. Outside a smoke, **activity**: the Watch records' summed
  `veto.total.vetoes_applied` is greater than 0. The split (v8 head vetoes, v7 re-rank changes,
  v5 vetoes and boost-landing vetoes) is reported, not gated.
- **S4 Counters.** Watch, per wrapped snake: the v2 counter partition and bounds; the v8
  identities (`decisions == kept + vetoes_applied + no_spacious`, each mirroring its v2-shaped
  counter; `v7.decisions == decisions`; `kept == v7.kept - head_vetoes_of_v7_kept`;
  `vetoes_applied == v7.vetoes_applied + head_vetoes_of_v7_kept`; `no_spacious == v7.no_spacious`;
  `head_checks == decisions - no_spacious`; `head_risky_decisions <= head_checks`;
  `head_risky_decisions == head_risky_vetoes + head_risky_kept_no_alternative`;
  `action_differs_from_v7 == head_risky_vetoes`; `head_vetoes_of_v7_kept`,
  `head_vetoes_of_v7_rerank`, `head_vetoes_speed_switched` and
  `head_vetoes_space_differs_from_highest_q` at most `head_risky_vetoes`;
  `head_risk_waived_hero_wins <= head_checks`); the served-candidate identity
  (`head_avoidance` true, `reference_lambda` null, `reference_decisions ==
  action_differs_from_reference == 0`); v7's probe counters satisfy the v2 partition; the nested
  v7 diagnostics satisfy the v7 lane's S4 rule including its probe mirrors against v7's probe
  counters (v8 runs v7 unchanged before its head layer), with v5's nested shape rule;
  `decisions` equals the decision frames; totals are per-snake sums (v7 `area_cap_max`: the
  per-snake maximum); decisions > 0. Play: no counters, diagnostics or decision frames (total 0).
  Every rule was checked against all 561 candidate final records of the v8 strict run before
  this file was pinned (0 failures).
- **S5 Parity.** Each probe: complete, env is the release environment, 5000 frames compared, no
  divergence, equal trace sha256, equal v2-shaped counters AND equal integer v8 diagnostics
  (v7, v5 and v7 probe counters included), session variant `v8` and both sides' wrapper method
  `free-space-veto/v8-space-and-head(lambda=8.0)`, session config the parity config, only snake
  0 wrapped, complete rollout record (smoke exempt), counters and diagnostics valid on both
  sides, session decisions equal decision frames. Reported, not gated: label `head-exercised`
  (a v8 head veto replaced an action), `rerank-exercised` (a v7 re-rank changed an action),
  `exercised` (some replacement ran) or `non-exercising`.
- **S6 Released default and rollbacks intact.** `default_check.json`, built from real sessions:
  an empty environment reads flags `{watch_hero: true, play_ai: false}` and variant `v7`; with an
  empty environment and with `SNAKE_SERVE_VETO_VARIANT=v7`, Watch wraps only the hero with
  `free-space-veto/v7-space-preference(lambda=4.0)` and Play wraps nothing; with
  `SNAKE_SERVE_VETO_VARIANT=v5` (resp. `v2`), Watch wraps only the hero with
  `free-space-veto/v5-boost-aware` (resp. `free-space-veto/v2-speed-preserving`) and Play wraps
  nothing; with `SNAKE_SERVE_VETO_WATCH_HERO=0` and `SNAKE_SERVE_VETO_VARIANT=v8` nothing is
  wrapped in Watch or Play; `DEFAULT_CHECKPOINT` is the champion and the served config is the
  pinned one. (So this run must happen before a release step flips the default variant to v8.)

Reported, not gated: veto rates, Watch activity split, v8 mean apply cost, Play end reasons,
run frames, hero deaths, the parity label, and the current tree's v8 source shas.

Outcomes: `SERVING_PASS` (every rule passes, `serving_path_qualified = true`), `SERVING_FAIL`
(a rule fails on complete evidence), `INVALID` (evidence missing or unreadable). Recovery
follows the governance tiers: one audit-only re-audit of unchanged evidence for an audit defect
(unit-tested against real record shapes); any producer defect needs a new seed domain (`-v2`).
Failures are preserved and never relabeled.

## Changes from the v7 lane

- Wrapper: v8 (lambda 8.0, head layer on, no reference lambda), seven bound sources instead of
  four; the gated method string carries the lambda, so a different lambda cannot match.
- Diagnostics: the v8 counter set (`SpaceAndHeadCounters.to_dict()` integer keys) plus
  `head_avoidance`/`reference_lambda`, with v7's integer diagnostics (v5 nested) under `v7` and
  v7's probe counters under `v7_probe_counters`; S4/S5 check v8 identities plus the v7 lane's
  rule on the nested block.
- S6 checks the currently released v7 default (and the v5, v2 and off rollbacks) instead of v5.
- Seed freshness adds the v8 strict gate's exclusion set, banks and observed seeds.
- Parity uses the v8 strict `spec.install_candidate`.
- Execution holds one shared CPU slot lock (`pqn-followup-20260909/cpu-slot-{1,2}.lock`) for the
  whole run and requires AC power at launch; these are launch conditions, not criteria.

## Dry run before GO

`serving_run.py --smoke --out <scratch>`: 1 Watch and 1 Play episode at 500 frames, 1 parity
probe at 200 frames (the rollout's H5000 accumulator cannot complete, so `record_complete`
false is accepted only when `intent.smoke`). Then `serving_audit.py --root <scratch> --out
<scratch>/audit`. The real run needs a clean tree, the pinned protocol and a fresh `--out`.

## Release step and rollback (not part of this run)

- **Release.** After a `SERVING_PASS`, with explicit user approval citing the v8 strict receipt
  and this audit: set `VARIANT_RELEASED_DEFAULT = VARIANT_V8` in
  `web/backend/safety_veto_serving.py` (and update its tests), then restart the server. Verify
  on stderr: `safety-veto-serving: active=True scope=watch_hero`, `wrapped_ids=[<hero id>]`,
  `strict_checkpoint_match=True`, `wrapper_source_sha256=faf3695f…ac05`, `variant=v8
  variant_requested=v8 wrapper_sources_match=True`,
  `wrapper_method=free-space-veto/v8-space-and-head(lambda=8.0)`, `reason=None`.
- **Rollback to v7.** Restart with `SNAKE_SERVE_VETO_VARIANT=v7` (no code change); `v5`/`v2`
  likewise.
- **Rollback to off.** Restart with `SNAKE_SERVE_VETO_WATCH_HERO=0`: no snake is wrapped.
- `SNAKE_SERVE_VETO_PLAY_AI` stays off. It has no strict evidence for any variant but v2.

## Non-claims

- No skill claim. The v8 strict gate measured one terminal wrapped hero against unwrapped
  opponents; served Watch respawns. Play under a wrapped AI is untested for v8 and not released.
- `set_hero` in Watch moves the inspector, not the wrapper.
- The SIMD engine path is not part of this lane.
- No change to `src/evaluation/strict_promotion.py`; schema `apex-veto-v8-web-serving/v1` is
  audited by `serving_audit.py` alone.
