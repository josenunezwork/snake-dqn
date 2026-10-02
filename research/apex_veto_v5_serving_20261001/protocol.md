# Apex + v5 boost-aware veto: web serving qualification (pre-registration)

Status: **pre-registered, written 2026-10-01 before any non-smoke serving episode.**
Revised once after the dry-run smoke (compute paragraph only, now quoting the smoke's measured
timings; no criterion changed), so the smoke intent records the pre-revision `protocol_sha256`.
The sha256 of this file is pinned in code (`serving_run.PROTOCOL_SHA256` and
`serving_audit.PROTOCOL_SHA256`), not here. Any later edit must re-pin both, and a non-smoke
intent that names another sha cannot pass.
Harness: `serving_run.py`; audit: `serving_audit.py` (standard library only); shared record
shapes: `schema.py`. The v2 serving lane (`research/apex_veto_serving_20261001/`, SERVING_PASS
at `snake-dqn-artifacts/apex-veto-serving-20261001/run-v1`) is reused: the harness imports its
stepping, stand-in, dispatch, decision-counting and rollout-tracing helpers unchanged, and the
audit loads its evidence loader and counter-partition rule by file path. The intent records the
sha256 of every harness file it ran, including the imported v2 ones.
Governance: [`docs/research/governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md).

## Question and decision it informs

Can the real web backend serve the champion behind the v5 wrapper on the Watch hero, selected
by `SNAKE_SERVE_VETO_VARIANT=v5`, with every served episode bound to the v5 strict-gated
identities, the wrapper demonstrably active on the hero alone, its counters and v5 diagnostics
consistent with the frames on which it decided, behavior identical to the v5 gate's rollout
where the two are comparable, Play left unwrapped, and the released v2 default and rollbacks
intact? A PASS sets `serving_path_qualified = true` for the v5 STRICT_PASS receipt
(`apex-veto-v5-strict-20261001/run-v1/output/receipt.json`, sha256 `cd843edfcfaf…`). It changes
no default, flag or deployment. Release stays a separate, explicit, user-approved action.

This run measures serving correctness, not skill. It makes no mass, survival or score claim.

## Identities (pinned)

| Item | Value |
|---|---|
| Checkpoint | `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93` |
| Wrapper | `free-space-veto/v5-boost-aware`, descriptor = `BoostAwareFreeSpaceVeto().descriptor()` |
| Wrapper sources | `src/evaluation/safety_veto_v5.py` `d86d084e…ec86c`, `src/evaluation/safety_veto.py` `1b62d15c…c428`, `src/evaluation/safety_veto_v3.py` `ed3a6d86…b5be1` (the v5 strict intent's candidate `source_sha256s`) |
| Served config | `configs/mechanics_v2.yaml`, sha256 `c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911` |
| Parity config | `research/apex_safety_20260926/deployment.yaml`, sha256 `4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5`, profile `promotion-v2-watch-rect` |

## Serving hook under test

`web/backend/safety_veto_serving.py`, called once at the end of every
`GameSession._build_impl`:

- `SNAKE_SERVE_VETO_WATCH_HERO` stays the master switch (unset = on, the released default;
  `0/false/no/off` = off). `SNAKE_SERVE_VETO_VARIANT` chooses the Watch hero's wrapper: unset or
  blank = `VARIANT_RELEASED_DEFAULT` (`v2` until the release step), `v2`, or `v5`. Any other
  value installs nothing on the Watch hero and says why.
- `v5` installs `install_boost_aware_veto` on the slot-0 Watch hero only, and only when the
  served checkpoint hashes to the pinned champion sha and all three source files hash to the
  pinned shas (fail closed; otherwise unwrapped with a `reason`).
- The variant never affects Play. `SNAKE_SERVE_VETO_PLAY_AI` keeps its v2 semantics (opt-in,
  v2 wrapper) and is unset here, so Play is served unwrapped. Play AI wrapping is not part of
  the release.
- When a flag is requested, each build logs one INFO line starting `safety-veto-serving:` with
  `active`, `variant`, `variant_requested`, `scope`, `wrapped_ids`, `strict_checkpoint_match`,
  `wrapper_sources_match`, `wrapper_method`, the wrapper source sha256 and `reason`.

The release environment every served build in this run reads is
`SNAKE_SERVE_VETO_VARIANT=v5` with the other two variables unset.

## Episodes and worlds

Seeds: `uint32_be(sha256("apex-veto-v5-web-serving-v1|<purpose>|<i>")[:4])` for purposes
`watch` (25), `play` (25) and `parity` (2). `serving_run.py` refuses to write the intent unless
they are unique and disjoint from: everything the v2 serving run checked (strict v1-v3 banks,
`apex-safety-screen-v1[:1000]`, the task-aligned challenger namespaces, the v3 screen); the
first 1000 seeds of every domain/purpose in the v5 strict gate's `EARLIER_DOMAINS` (including
`apex-veto-web-serving-v1` and its smoke, the v3/v4/v5 screens and smokes and the trap-horizon
diagnostic); the first 1000 seeds of the v5 strict dev/final/serving banks and their smoke; and
seeds 0..999. In-code check only (no namespace registry). Smoke uses
`apex-veto-v5-web-serving-smoke-v1`.

Counts: 25 Watch + 25 Play (50 served episodes) + 2 parity probes. The v2 serving lane moved
from 1 Watch + 49 Play to this split after review because Watch is the scope a release turns
on; the same holds for v5, and Play is unwrapped here.

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
   `dev_screen.hero_veto_installer(install_boost_aware_veto)`, the v5 strict gate's candidate
   install. Per frame both sides record frame number, each snake's id, head, length, alive and
   boosting flags, and the food count.

Comparability is as in the v2 lane: served Watch differs from the gate profile in config and
hero lifecycle, so only parity runs the session code under the profile's config and lifecycle.

Compute cap: 10800 s wall for the whole run (the v2 lane's cap, which assumed about 160 s per
5000-frame episode). This lane's dry-run smoke measured 2.5 s per 500-frame Watch episode (v5
mean apply cost 0.11 ms per hero decision), 2.3 s per 500-frame Play episode and 1.7 s for a
200-frame parity probe, 7.1 s in all, so the real run is expected to take well under an hour.

## Receipts

`<out>/intent.json` is written before any episode, then `default_check.json`,
`records/<kind>-<NNN>.json` (kinds `watch_hero`, `play`), `parity/parity-<NNN>.json` and
`receipt.json`, which lists every file with its sha256. Files are create-only. Each episode
record (`schema.EPISODE_KEYS`) binds the env the build read, the checkpoint sha (file and
session), the wrapper identity as installed (method, descriptor, source path, source sha256 and
the three `source_sha256s`), the served identity, the veto block (`schema.VETO_KEYS`: variant,
scope, reason, flags, match flags, wrapped ids, per-snake and total counter deltas, per-snake and
total integer v5 diagnostics, and wall-clock timings), the ids of snakes carrying any veto, and
per wrapped snake the harness-observed decision frames.

## Pass criteria (all required; `serving_audit.py` decides)

- **S1 Completeness.** Outside a smoke: `intent.protocol_sha256` equals the pinned sha,
  `intent.git.dirty` is false and `intent.seed_report.disjoint` is true. `receipt.json` lists
  exactly 25 `watch_hero`, 25 `play` and 2 parity records plus `default_check.json`, every hash
  matches, every record is `complete` with `error` null, bound to the intent sha and schema, with
  the recipe seed. `intent.release_env` is the release environment. Watch steps exactly 5000
  frames (`horizon`); Play ends at `human_death` within 5000 frames or at `horizon` with 5000.
- **S2 Identity binding.** Intent and every Watch record carry exactly the pinned v5 wrapper
  identity (method, descriptor, source path, source sha256, all three `source_sha256s`); the
  intent's expected `v5_source_sha256s` are the pins. Every record: checkpoint sha (file and
  session) is the champion, `env` is the release environment, `served` shows its mode, obs_spec
  `vector61`, `ApexPolicy`, training false, epsilon 0, `mechanics_v2.yaml` with the pinned sha
  and flags `{watch_hero: true, play_ai: false}`. Play records carry no wrapper identity.
- **S3 Scope and activity.** Every record: flags as above, `variant_requested` `v5`. Watch:
  active, scope `watch_hero`, reason null, variant `v5`, both match flags true, hero is slot 0,
  `wrapped_snake_ids == [hero_id]` and the hero is the only snake carrying a veto. Play:
  inactive, scope and variant null, reason `no serving veto flag applies to play mode`, nothing
  wrapped and no snake carrying a veto. Outside a smoke, **activity**: the Watch records' summed
  `veto.total.vetoes_applied` is greater than 0. What counts: any decision whose executed action
  differs from the policy's masked argmax, from either branch: a v5 boost-landing veto
  (`diagnostics.boost_landing_vetoes`) or a v2-path veto (`vetoes_applied -
  boost_landing_vetoes`). The split, `landing_checks` and `action_differs_from_v2` are reported,
  not gated: a 25-episode Watch sample need not contain a failing boost landing.
- **S4 Counters.** Watch, per wrapped snake: the v2 counter partition and bounds; the v5
  identities (`decisions == kept + vetoes_applied + no_spacious`, each equal to its v2-shaped
  counter; `base_landing_failed == boost_landing_vetoes + boost_landing_no_eligible`;
  `boost_landing_vetoes == boost_to_normal_same_direction + boost_landing_v2_rule`;
  `boost_landing_vetoes <= vetoed_base_boost`; `landing_failures <= landing_checks`;
  `boost_base_decisions` and `action_differs_from_v2` at most `decisions`); `decisions` equals
  the decision frames; totals are per-snake sums; decisions > 0. Play: no counters,
  diagnostics or decision frames (total 0).
- **S5 Parity.** Each probe: complete, env is the release environment, 5000 frames compared, no
  divergence, equal trace sha256, equal v2-shaped counters AND equal integer v5 diagnostics,
  session variant `v5` and both sides' wrapper method `free-space-veto/v5-boost-aware`, session
  config the parity config, only snake 0 wrapped, complete rollout record (smoke exempt),
  counters and diagnostics valid on both sides, session decisions equal decision frames.
  Reported, not gated: label `landing-exercised`, `exercised` or `non-exercising`.
- **S6 Released default and rollbacks intact.** `default_check.json`, built from real sessions:
  an empty environment reads flags `{watch_hero: true, play_ai: false}` and variant `v2`; with
  an empty environment and with `SNAKE_SERVE_VETO_VARIANT=v2`, Watch wraps only the hero with
  `free-space-veto/v2-speed-preserving` and Play wraps nothing; with
  `SNAKE_SERVE_VETO_WATCH_HERO=0` and `SNAKE_SERVE_VETO_VARIANT=v5` nothing is wrapped in Watch
  or Play; `DEFAULT_CHECKPOINT` is the champion and the served config is the pinned one. (So this
  run must happen before the release step flips the default variant.)

Reported, not gated: veto rates, Watch activity split, Play end reasons, run frames, hero
deaths, the parity label, and the current tree's v5 source shas.

Outcomes: `SERVING_PASS` (every rule passes, `serving_path_qualified = true`), `SERVING_FAIL`
(a rule fails on complete evidence), `INVALID` (evidence missing or unreadable). Recovery
follows the governance tiers: one audit-only re-audit of unchanged evidence for an audit defect
(unit-tested against real record shapes); any producer defect needs a new seed domain (`-v2`).

## Dry run before GO

`serving_run.py --smoke --out <scratch>`: 1 Watch and 1 Play episode at 500 frames, 1 parity
probe at 200 frames (the rollout's H5000 accumulator cannot complete, so `record_complete`
false is accepted only when `intent.smoke`). Then `serving_audit.py --root <scratch> --out
<scratch>/audit`. The real run needs explicit user authorization and a fresh `--out`.

## Release step and rollback (not part of this run)

- **Release.** After a `SERVING_PASS`, with explicit user approval citing the v5 strict receipt
  and this audit: set `VARIANT_RELEASED_DEFAULT = VARIANT_V5` in
  `web/backend/safety_veto_serving.py` (and update its tests), then restart the server. Verify
  on stderr: `safety-veto-serving: active=True variant=v5 variant_requested=v5
  scope=watch_hero`, `wrapped_ids=[<hero id>]`, `strict_checkpoint_match=True
  wrapper_sources_match=True`, `wrapper_method=free-space-veto/v5-boost-aware`, wrapper source
  sha256 `d86d084e…ec86c`.
- **Rollback to v2.** Restart with `SNAKE_SERVE_VETO_VARIANT=v2` (no code change): the log line
  shows `variant=v2` and `wrapper_method=free-space-veto/v2-speed-preserving`.
- **Rollback to off.** Restart with `SNAKE_SERVE_VETO_WATCH_HERO=0`: no snake is wrapped and no
  line is logged.
- `SNAKE_SERVE_VETO_PLAY_AI` stays off. It has no strict evidence for either variant.

## Non-claims

- No skill claim. The v5 strict gate measured one terminal wrapped hero against unwrapped
  opponents; served Watch respawns. Play under a wrapped AI is untested for v5 and not released.
- `set_hero` in Watch moves the inspector, not the wrapper.
- No change to `src/evaluation/strict_promotion.py`; schema `apex-veto-v5-web-serving/v1` is
  audited by `serving_audit.py` alone.
