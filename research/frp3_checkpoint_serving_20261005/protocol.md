# FRP-v3 s12 checkpoint swap + released v8 veto: web serving qualification (pre-registration)

Status: **pre-registered, written 2026-10-05 before any non-smoke serving episode.** It may be
run only after (1) the swap's strict gate (frp3-s12 + v8 vs the champion + v8) closes
`STRICT_PASS` and (2) the owner records that receipt in the served-checkpoint pins with
`fill_strict_pin.py`. Until then the registry refuses `frp3-s12` and a real run refuses to
start.

This lane mirrors the v8 web serving lane (`research/apex_veto_v8_serving_20261003/`,
SERVING_PASS) with the checkpoint swapped and the veto held fixed: the same episode counts,
horizons, stepping order, dispatch path, stand-in, counter/diagnostics rules and criteria
S1-S6. Differences are listed under "Changes from the v8 lane".
The sha256 of this file is pinned in code (`serving_run.PROTOCOL_SHA256` and
`serving_audit.PROTOCOL_SHA256`), not here. Any later edit must re-pin both, and a non-smoke
intent that names another sha cannot pass.
Harness: `serving_run.py`; audit: `serving_audit.py` (standard library only); shared record
shapes: `schema.py`; owner pin tool: `fill_strict_pin.py`. The harness imports the v2 lane's
stepping, stand-in, dispatch, decision-counting, served-identity and parity-config helpers and
the v8 lane's veto block, diagnostics and seed-report helpers unchanged; the audit loads the v8
audit (which loads the v2 and v7 audits) by file path for its evidence loader, v8 identity,
counter and diagnostics rules and its S4 criterion. The intent records the sha256 of every
harness file it ran, the registry and pins file, the veto serving hook, the session and the
three SIMD engine files the parity probe runs.
Governance: [`docs/research/governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md).

## Question and decision it informs

Can the real web backend, built exactly as `web/backend/app.py` builds it (`GameSession()`
with no explicit checkpoint), serve the FRP-v3 arm-M3 seed-12 checkpoint at 60000 learner
updates on Watch when an operator sets `SNAKE_SERVE_CHECKPOINT=frp3-s12`, with the released
v8 wrapper (lambda 8.0, head layer on, no reference lambda) on the Watch hero, every served
episode bound to the pinned checkpoint bytes and to the swap's strict receipt, the wrapper
active on the hero alone with counters and v8 diagnostics consistent with the frames on which
it decided, behavior identical to a SIMD rollout of the same identity, Play left unwrapped,
and the released defaults (champion + v8), the fail-closed refusals and the rollbacks intact?
A PASS sets `serving_path_qualified = true` for the swap. It changes no default, flag or
deployment. Release stays a separate, explicit, user-approved action.

This run measures serving correctness, not skill. It makes no mass, survival or score claim.

## Identities (pinned)

| Item | Value |
|---|---|
| Checkpoint | registry name `frp3-s12`; FRP-v3 `train/arm-M3/seed-12/checkpoints/apex_mark_u60000.pth`, sha256 `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`; served from `saved_snakes/frp3_m3_s12_u60000_20261005.pth` when present, else the artifact path |
| Strict receipt | `web/backend/served_checkpoint_pins.json` `checkpoints["frp3-s12"].strict_receipt`: verdict `STRICT_PASS`, absolute `root`, and path + sha256 of the strict receipt, intent and audit report (shipped as `null`; filled by the owner after the gate) |
| Wrapper | the released v8: `free-space-veto/v8-space-and-head(lambda=8.0)`, descriptor `SpaceAndHeadVeto(8.0).descriptor()`, the seven v8 source sha256s of the v8 strict receipt (`29b1f7f6…`), unchanged |
| Served config | `configs/mechanics_v2.yaml`, sha256 `c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911` |
| Parity config | `research/apex_safety_20260926/deployment.yaml`, sha256 `4146baa3…15aa5`, profile `promotion-v2-watch-rect`, digest `d396d3ed…ea0e8b` |

## Serving hooks under test

- `web/backend/served_checkpoint.py` (new), called by `GameSession.__init__` when no explicit
  checkpoint is given. `SNAKE_SERVE_CHECKPOINT` unset/blank = `CHECKPOINT_RELEASED_DEFAULT`
  (`champion`, served exactly as before: `DEFAULT_CHECKPOINT`, untrained weights if absent);
  `champion`; `frp3-s12`; anything else = the released default with a `reason`. `frp3-s12` is
  served only when its strict pin is filled and valid (schema, checkpoint sha, v8 variant and
  method, `STRICT_PASS`, absolute root, three relative paths with 64-hex shas) and the file it
  finds hashes to the pinned sha; otherwise the released default is served with a `reason`
  (fail closed, never raised). One INFO line `served-checkpoint: name=… requested=… sha256=…
  path=… reason=…` per resolution when the variable is set, a request is refused or a
  non-champion checkpoint is served.
- `web/backend/safety_veto_serving.py`: unchanged v8 pins. `checkpoint_match` under `v8` also
  accepts a registry checkpoint whose filled pin names v8; every other variant keeps its
  champion-only binding.
- The variant never affects Play. `SNAKE_SERVE_VETO_PLAY_AI` is unset, so Play is served
  unwrapped, with the frp3-s12 opponents.

The release environment every served build in this run reads is
`SNAKE_SERVE_CHECKPOINT=frp3-s12` with the three veto variables unset (Watch hero on by the
released default, variant `v8` by the released default, Play AI off).

## Episodes and worlds

Seeds: `uint32_be(sha256("frp3-checkpoint-web-serving-v1|<purpose>|<i>")[:4])` for purposes
`watch` (25), `play` (25) and `parity` (2). `serving_run.py` refuses to write the intent unless
they are unique and disjoint from: everything the v8 web lane checked (its report contains the
v7/v5/v2 lanes', the v8 strict gate's exclusions, banks and observed seeds); the first 1000 seeds
of every web purpose of the v8 lane's own domains; FRP-v3's written training exclusion file
(`frp-v3-20261005/namespaces/training_exclusions.json`, sha256 `4b0f3377…1edf`: every seed its
preflight found in every artifact root plus its own banks); seeds 0..999; and (real run only;
required) every integer under a seed-named key or `banks` in the swap's strict intent and in
every `rosters.json` under its root, plus the first 1000 `worlds` seeds of every domain or
namespace string its intent names. In-code check only. Smoke uses
`frp3-checkpoint-web-serving-smoke-v1`.

Counts: 25 Watch + 25 Play (50 served episodes) + 2 parity probes, the v8 lane's split.

1. **Watch, as served (25 episodes, 5000 frames each).** `set_seed(seed)`, then `GameSession()`
   under the release environment, stepped with `step()` then `snapshot()`. Legacy Watch is
   all-respawn.
2. **Play (25 episodes).** One `GameSession()` built under the release environment and switched
   with a `set_mode` control. Each episode: `set_seed(seed)`, a `new_game` control (under the
   release environment, since it rebuilds), then the v2 lane's seeded stand-in sends
   `human_input` controls until the human dies or 5000 frames. Every control goes through
   `web.backend.app._apply_control`; a control that sets `session.last_error` fails its episode.
3. **Parity probes (2, not served episodes, H5000).** Session side: `GameSession()` on the
   parity config, release environment, hero made terminal (`auto_respawn=False`), stepped as
   in 1. SIMD side: `run_simd_eval(frp3, 5 x frp3, 5000, [seed], profile=promotion-v2-watch-rect,
   vector61=True, hero_safety_veto="v8", hero_safety_veto_lambda=8.0,
   vector61_forward="rowwise")` on the parity config. Per frame both sides record, in cell
   space, the frame number, per slot alive and (alive only) head cell and logical length, the
   food count and the sha256 of the sorted food cells; the SIMD trace is read from the BatchSim
   after each transition (observation only, through the engine's post-frame snapshot seam).

Compute: one shared CPU slot (pool 3, the Tier-1/dev default) held for the whole run,
`OMP_NUM_THREADS=1`, torch 1 thread. A real run refuses to start without AC power and an open
lid, and before every episode waits while either fails (at most 1 h, then that episode records
an error). Thread caps (`OMP`/`MKL`/`OPENBLAS`/`VECLIB` = 1) are set before any import. Wall
time is reported in the receipt, not gated (the v8 lane took well under an hour).

## Receipts

`<out>/intent.json` is written before any episode, then `default_check.json`,
`records/<kind>-<NNN>.json`, `parity/parity-<NNN>.json` and `receipt.json`, which lists every
file with its sha256. Files are create-only. Each episode record carries the v8 lane's fields
plus `served_checkpoint` (the registry's resolution: requested, name, path, sha256, reason,
pin problems). The intent carries the strict pin it served under.

## Pass criteria (all required; `serving_audit.py` decides)

- **S1 Completeness.** The v8 lane's S1 with this lane's design: outside a smoke
  `intent.protocol_sha256` is the pinned sha, `intent.git.dirty` is false,
  `intent.seed_report.disjoint` is true and the strict run's seeds were checked; exactly 25
  `watch_hero`, 25 `play` and 2 parity records plus `default_check.json`, hashes match, every
  record complete, bound, recipe seed; `intent.release_env` is the release environment; Watch
  steps 5000 frames; Play ends at `human_death` within 5000 frames or at `horizon` with 5000.
- **S2 Identity binding.** Outside a smoke, the intent's strict pin equals the repo pins
  file's filled `frp3-s12` entry; the strict receipt, intent and audit report it names exist
  and hash to the pinned shas; the strict intent names the checkpoint sha and the v8 method;
  the receipt or its audit report says `STRICT_PASS`. Intent and every Watch record carry
  exactly the pinned v8 wrapper identity. Every record: checkpoint sha (file and session) is
  frp3-s12's; `served_checkpoint` requested and resolved `frp3-s12` with the pinned sha, no
  reason and no pin problems, from the saved-snakes filename or the artifact path; `env` is
  the release environment; `served` shows its mode, `vector61`, `ApexPolicy`, training false,
  epsilon 0, `mechanics_v2.yaml` with the pinned sha and flags `{watch_hero: true, play_ai:
  false}`. Play carries no wrapper identity.
- **S3 Scope and activity.** The v8 lane's S3: `variant_requested` `v8`; Watch active, scope
  `watch_hero`, reason null, variant `v8`, both match flags true (the checkpoint match now
  comes through the swap's pin), hero slot 0 and the only snake carrying a veto; Play inactive
  with reason `no serving veto flag applies to play mode`. Outside a smoke the Watch records'
  summed `veto.total.vetoes_applied` is greater than 0.
- **S4 Counters.** The v8 lane's S4, unchanged (v2 partition, v8 identities, nested v7/v5
  rules, served-candidate identity, decisions == decision frames, totals).
- **S5 Parity.** Each probe: complete, release environment, 5000 frames compared, no
  divergence, equal trace sha256, equal v2-shaped counters AND equal integer v8 diagnostics
  (v7, v5, v7 probe counters included); session checkpoint and `served_checkpoint` are
  frp3-s12's, session variant `v8`, both sides' method the v8 method, SIMD provenance exactly
  `{engine: simd, policy: Vector61SimdPolicy, forward: rowwise, bit_exact_forward: true,
  hero_safety_veto: true, safety_veto_method: <v8 method>}`, SIMD profile digest the pinned
  one at scored horizon 5000 (smoke exempt), session config the parity config, only snake 0
  wrapped, counters and diagnostics valid on both sides, session decisions equal decision
  frames. The parity label (head/rerank/exercised/non-exercising) is reported, not gated.
- **S6 Released defaults, refusals and rollbacks intact.** `default_check.json`, from real
  `GameSession()` builds: an empty environment reads flags `{watch_hero: true, play_ai:
  false}`, variant `v8` and checkpoint default `champion`; `DEFAULT_CHECKPOINT` is the
  champion and the served config the pinned one; and per case (Watch, then Play after
  `set_mode`):
  `empty_env` and `SNAKE_SERVE_CHECKPOINT=champion` serve the champion with the v8 hero
  wrapper; `SNAKE_SERVE_CHECKPOINT=frp3-s99` (unknown) and `frp3-s12` under an unfilled pin
  (in-process) are refused with a `reason` and serve the champion with the v8 hero wrapper;
  `frp3-s12` with `SNAKE_SERVE_VETO_VARIANT=v7` serves frp3-s12 with Watch unwrapped (no v7
  evidence for it); `frp3-s12` with `SNAKE_SERVE_VETO_WATCH_HERO=0` serves frp3-s12 unwrapped;
  Play is unwrapped in every case. (So this run must happen before a release step flips the
  checkpoint default.)

Reported, not gated: the v8 lane's reported block (veto rates, Watch activity split, mean
apply cost, Play end reasons, hero deaths, parity label, current-tree v8 source shas) and the
served checkpoint paths.

Outcomes: `SERVING_PASS` (every rule passes, `serving_path_qualified = true`), `SERVING_FAIL`
(a rule fails on complete evidence), `INVALID` (evidence missing or unreadable). Recovery
follows the governance tiers: one audit-only re-audit of unchanged evidence for an audit
defect; any producer defect needs a new seed domain (`-v2`). Failures are preserved and never
relabeled.

## Changes from the v8 lane

- The checkpoint is chosen by the new registry from `SNAKE_SERVE_CHECKPOINT` (a fourth env key)
  instead of an explicit `GameSession(checkpoint=…)`; records add `served_checkpoint`.
- S2 binds the swap's strict receipt (through the pins file) instead of the v8 strict receipt;
  the v8 wrapper identity is unchanged.
- Parity compares the session with a SIMD rollout of the same identity (cell-space traces)
  instead of the live `tournament_eval.rollout`.
- S6 checks the released champion + v8 defaults, the fail-closed refusals and the frp3-s12
  rollbacks instead of the v7/v5/v2 variant rollbacks.
- Seed freshness adds the v8 lane's own domains, FRP-v3's training exclusions and the swap's
  strict run.
- The harness holds the CPU slot and checks AC power and the lid itself.

## Dry run before GO

`serving_run.py --smoke --out <scratch>`: 1 Watch and 1 Play episode at 500 frames, 1 parity
probe at 200 frames (the SIMD profile's `scored_horizon` shortened to 200, dynamics
unchanged), served under an in-process placeholder pin that the intent records. Then
`serving_audit.py --root <scratch> --out <scratch>/audit`. A smoke never qualifies. The real
run needs a clean tree, the pinned protocol, the filled pin, the champion copied into the
checkout's `saved_snakes/` and a fresh `--out`.

## Release step and rollback (not part of this run)

- **Release.** After a `SERVING_PASS`, with explicit user approval citing the swap's strict
  receipt and this audit: copy the checkpoint to `saved_snakes/frp3_m3_s12_u60000_20261005.pth`,
  set `CHECKPOINT_RELEASED_DEFAULT = NAME_FRP3_S12` in `web/backend/served_checkpoint.py` (and
  update its tests), then restart the server. Verify on stderr: `served-checkpoint:
  name=frp3-s12 requested=None sha256=eec144bf…3723 path=…/saved_snakes/frp3_m3_s12_u60000_20261005.pth
  reason=None` and `safety-veto-serving: active=True scope=watch_hero mode=watch
  wrapped_ids=[<hero id>] … checkpoint_sha256=eec144bf…3723 strict_checkpoint_match=True …
  variant=v8 variant_requested=v8 wrapper_sources_match=True
  wrapper_method=free-space-veto/v8-space-and-head(lambda=8.0) reason=None`.
- **Rollback to the champion.** Restart with `SNAKE_SERVE_CHECKPOINT=champion` (no code change).
- **Veto rollbacks with frp3-s12 served.** `SNAKE_SERVE_VETO_WATCH_HERO=0` serves it
  unwrapped; `SNAKE_SERVE_VETO_VARIANT=v7/v5/v2` also leaves its hero unwrapped (those
  receipts bind only the champion). To roll back the veto, roll back the checkpoint too.

## Non-claims

- No skill claim (that is the strict gate's job). Served Watch respawns; the strict gate
  measures one terminal wrapped hero against unwrapped opponents.
- The checkpoint's Q-values are about 4x smaller than the champion's (FRP-v3 trained after a
  0.25 head and reward rescale). v8's space score normalizes Q by its candidate range, so the
  lambda trade-off is scale-free; the UI's fixed confidence thresholds (steering-wheel hub,
  narrator) are not, which this lane neither tests nor changes.
- Train mode from a frp3-s12 session is not part of this lane.
- No change to `src/evaluation/strict_promotion.py`.
