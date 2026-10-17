DECISION: RELEASE

# Advice: release B8 seed 3 as the served Watch hero, after an extended Mac serving qualification (2026-10-17)

Advisory decision (cloud advisor), delegated by the owner ("ask Fable, not me"). Not gate
evidence. Inputs: `t7_phase_r_result_2026-10-16.md`, `frp3_serving_result_and_release_2026-10-06.md`,
`research/frp3_checkpoint_serving_20261005/protocol.md`, `web/backend/served_checkpoint.py`,
`served_checkpoint_pins.json`, `safety_veto_serving.py`, `ego2s_policy.py`, `ego_live_bv8.py`,
`session.py`, `redesign_scope_2026-10-07.md` section 18.1, STATUS.md.

## Why release

The campaign's completion bar was Tier 1 AND Tier 2, fixed before gameplay. Both passed on a
fresh, audited, one-look bank: mass +138.7 (LB90 +129.5) on 5/5 seeds; scripted alive-frame
survival at parity with the served agent (-0.007, SE 0.012); frozen and mixed better by about
+0.045; absolute episode survival above 0.50 on every mix; Tier 2 -0.016 (SE 0.023) against
the champion. The candidate rule was fixed in advance and selected seed 3 cleanly. The
pre-registration said: after STRICT_PASS, Mac serving qualification exactly as for frp3-s12,
then release. Holding now would be changing the rule after the result, which is the one
thing this project's governance exists to prevent. Open research (LOCAL30, T10) cannot alter
this candidate and is not a reason to wait.

The release claim is precise and should be recorded as such: **more mass at equal survival
against the served agent, with no veto.** It is not "safer". Watch viewers will see the hero
die late in the game about as often as frp3-s12 + v8 did.

## What is new at serving, and what the qualification must therefore add

frp3-s12 was a checkpoint swap inside the same policy class and wrapper. B8 is a different
policy class (`Ego2sServingPolicy`, forward-only), a different observation (ego2s-bv8 with 30
v8 scalars computed live by `LiveV8Source`), and no wrapper. The frp3 protocol's S1-S6 carry
over as structure; these checks are additional and each must be a pass criterion:

1. **Identity of the live observation.** The P2 no-veto record parity (grid vs live, advisory
   mask) already validates for all five finals (9/9 H5000, 3/3 H10000). Re-run it for seed 3's
   exact pinned bytes from the release build, plus the lockstep `test_ego_live_bv8.py` and
   `test_ego_live_identity.py`. The parity probe (S4 analogue) compares a served session to
   `run_simd_eval` with `ego2s_mask="advisory"`, rowwise forward, no veto, on the gate profile.
2. **The veto is absent and says so.** With the released default env (Watch hero flag on by
   default), `install_serving_vetoes` must return inactive with the reason "applies only to a
   vector61 Apex policy", the stderr line must show `active=False`, and `v8_gated_checkpoint_sha256s`
   must not contain B8's sha. The pin's `veto_variant` must be `none`. Both guards, not one.
3. **Registry, pin and fail-closed fallback.** Add a `b8-s3` registry entry with the T7
   checkpoint sha, artifact path and a gitignored copy under `saved_snakes/`. The frp3 pin
   tool is hard-coded to frp3-s12 and the strict-lane receipt shape; write a sibling tool that
   binds T7's `decision.json`, isolated audit and independent review hashes, refuses unless the
   audit says PASS with STRICT_PASS recomputed, and refuses to overwrite. When the B8 pin or
   bytes do not verify, the fallback must be **frp3-s12 + v8** if its pin verifies, else the
   champion, with a reason. That is a change to `resolve_served_checkpoint` and needs its own
   S6-style test matrix: empty env, `champion`, `frp3-s12`, `b8-s3`, unknown name, unfilled
   pin, corrupted bytes, and `SNAKE_SERVE_VETO_VARIANT=v7` with no name (still champion + v7).
4. **Latency at the served roster.** ego2s-b measured 5.4 ms p50 and 7.0 ms p95 per frame for
   12 driven snakes; ego2s-bv8 adds a v8 flood per driven snake, about 0.6 ms each on the frp3
   wrapper's measurement. Measure the whole `GameState.update` at 6 and 12 snakes, Watch and
   Play, with every AI snake served by B8. Pass: p95 under the frame budget at the default
   12 fps (83 ms) with margin, and record the highest speed setting the server sustains
   without tick overrun, so the UI's speed slider ceiling can be documented or clamped.
5. **Self-play roster disclosure.** In Watch every snake is B8 and in Play every opponent is
   B8; the gate measured B8 against scripted, frozen and mixed rosters, never against itself.
   frp3-s12 was released under the same gap. Record the Watch hero's deaths per 1,000 decisions
   and length distribution over the 25 Watch episodes as the baseline for monitoring, and
   disclose the gap in the release record. It is not a pass criterion.
6. **Play mode.** 25 Play episodes with the stand-in human: no `last_error`, opponents
   unwrapped, the human never featurized or driven, `play_opponents` adjustable, and the
   scored-run path (`/api/players`) still records. Opponents are now about 35% heavier at
   H5000; the leaderboard's "best game" rows are no longer comparable with pre-release rows.
   If the store does not record the served checkpoint name per scored run, add it in this
   release; it is a one-field change and keeps the leaderboard honest.
7. **UI.** The inspector groups for ego2s-bv8 exist; confirm the narrator and steering-wheel
   hub render with no veto counters and that the relative-margin confidence cutoffs give a
   sane split on 1,500 B8 Watch frames (report the split; recalibrate only if it collapses to
   one bucket, as it would have for frp3 under the old absolute cutoffs).
8. **Train mode** is refused for ego2s students and drops to Watch with a message; keep that
   behaviour and state in the release record that Train is not qualified, as for frp3.

Governance: write the extended protocol as a new sibling file, pin its sha in the harness and
audit before any non-smoke episode, seed-disjoint from every bank in the T7 freshness audit.
Do not edit the frp3 protocol.

## Release mechanics

- A checkpoint swap only: `CHECKPOINT_RELEASED_DEFAULT = "b8-s3"`, pin-gated, fallback order
  frp3-s12 + v8 then champion. No veto flag changes. No obs-spec or engine default changes.
- Rollback is one environment variable and a restart: `SNAKE_SERVE_CHECKPOINT=frp3-s12`
  restores frp3-s12 + v8 exactly as released on 2026-10-06; `=champion` restores champion + v8.
  Both must be exercised in the qualification run, not just unit-tested.
- Expected default stderr: `served-checkpoint: name=b8-s3 ... sha256=3e649783...` and
  `safety-veto-serving: active=False ... reason=applies only to a vector61 Apex policy`.
- Record the release in STATUS.md and a release document mirroring the frp3 one, with the
  claim stated as above and the disclosures: trained with v8 labels, served without a veto,
  Tier 2 margin 0.034 above threshold at SE 0.023, self-play roster ungated.

## Monitoring and rollback triggers after release

Over the first week, or the first 100,000 Watch hero decisions, whichever is later:

- **Roll back** if any served session logs a fallback reason or an exception in the ego2s
  path, if the Watch hero's deaths per 1,000 decisions exceed twice the qualification-run
  baseline (frp3-s12's own serving run was 0.14 per 1,000), or if p95 frame time at the
  default speed exceeds the qualification measurement by more than 50%.
- **Review, do not roll back** on complaints that the snake dies late. That rate is at parity
  with the incumbent by the gate; it is the known open problem, and T10 and LOCAL30 are the
  work on it.
- One owner check-in at seven days with the counters, then normal operation.

## Conditions in one line

Release B8 seed 3 only after the extended qualification above passes on the Mac with its
protocol pinned beforehand, the fallback chain and both rollbacks exercised live, and the
release record carrying the exact claim and disclosures. If any numbered check fails, HOLD
on that check alone, fix the serving code, and re-run; the gate result itself does not need
to be revisited.
