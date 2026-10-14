# Candidate 6102: prospective serving bridge and strict challenge

Date: 2026-09-24. Status: **PLAN ONLY — NOT ADMITTED**.

This source review describes a possible individual challenge of candidate
2026096102 against the incumbent. It authorizes no numerical work, checkpoint
access, new experiment window, or release. The strict consumer accepts one
candidate and one incumbent; there is no requirement that all three research
lineages pass. Completed research comparisons support selecting this candidate,
but do not establish strict promotion authority.

## Identities and actual implementation blockers

The completed study declares candidate 6102's source checkpoint SHA-256 as
`8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f`.
Its source path is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-cz-integration-20260924/admission-v2/run/train_2026096102/output/checkpoint_rollout_31005.pth`.
This is inert prior metadata, not a checkpoint identity freshly verified by this
review. A faithful derived serving artifact, if necessary, would have its own
identity and explicit relationship to these source bytes.

The incumbent path is `saved_snakes/champion_a5_freespace_20260621.pth`, as named
by `src/scripts/tournament_eval.py:116` and `web/backend/session.py:54`. No
incumbent payload was opened or hashed. Strict execution obtains the incumbent
digest from the request and E0, not from this default filename, so the eventual
request must explicitly authenticate the intended champion.

The serving review identified concrete compatibility blockers: the custom
task-aligned checkpoint lacks stock observation, model-head, effective-world and
runtime contracts. Its inference recipe restricts tactical inputs to channels
0–5, scalar inputs to indices 0, 1, 12–14, zeros strategic input, and uses Q scale
0.1. Its actual mixed Watch training uses `training=False`, terminal heroes and
opponent respawning. That provenance is incompatible with the stock strict
candidate validator's required `pqn_train` source-runtime declaration. These
findings were supplied by the serving review; this operations review did not
inspect the checkpoint.

Consequently, changing metadata to claim stock PQN training would falsify the
source contract. The missing work is a versioned, faithful serving bridge and
compatible validation path that preserve the restricted inference recipe,
weights and actual lifecycle/provenance. Its implementation, bounded
qualification, deployment-world treatment, source closure and artifact identities
must be settled before admission. Current source validation opens candidate
bytes and checks those contracts, rather than accepting a declaration alone
(`src/evaluation/strict_promotion.py:484`, `:1056`).

## Required artifact chain

1. **E0 and source freeze.** Snapshot candidate, incumbent, declared checkpoint
   pool and configuration. E0 requires exact role identities, content hashes and
   a clean committed source closure, conservatively including all Python source
   below `src` and `web/backend`. Ordered roles are incumbent, pool, candidate.
   Bridge changes therefore precede this freeze
   (`strict_promotion.py:946`; `src/evaluation/artifacts.py:74`, `:200`).
2. **Profile, roster and namespace declaration.** Use the live engine and
   `promotion-v2-watch-rect`, with scored and observation horizons 5000 and
   learning disabled. Declare six nonempty, internally unique, pairwise disjoint
   uint32 seed lists: training, development, shakedown, pilot, final and serving.
   Pilot requires at least two seeds; serving requires exactly 50. Canonical
   rosters cover frozen, scripted and mixed opponents. Frozen slots rotate the
   checkpoint pool; scripted slots use greedy-food; mixed slots alternate the
   checkpoint pool and random-safe, with deterministic per-slot balancing
   (`strict_promotion.py:343`, `:391`, `:652`).
3. **Development calibration.** Run the incumbent on all development seeds in
   all three mixes. Positive reference means determine the NI margin and MDEs
   from predeclared positive fractions. Nonempty behavioral bands use reference
   means plus declared offsets. Required unavailable metrics fail validation
   (`strict_promotion.py:1228`).
4. **Paired pilot.** Produce candidate/incumbent pairs on every pilot seed and
   mix, with matched world/roster identities. Sizing uses paired-delta variation,
   planning alpha .05/3 and an 80% marginal-power approximation, not a joint-power
   guarantee. Final N is at least 40 and at least the maximum pilot-derived
   requirement across the three mixes (`strict_promotion.py:1128`;
   `src/scripts/eval_stats.py:581`).
5. **Serving bundle.** Execute exactly one Watch and 49 Play episodes on their
   frozen serving seeds. Watch completes external H5000; Play completes at
   observed human terminal or external H5000. Receipts must demonstrate actual
   seeded execution, layouts, lifecycle, participants, human input, dispatch and
   candidate contracts. Short operability spot checks do not substitute
   (`strict_promotion.py:548`, `:1500`; `src/evaluation/serving_episode.py`).
6. **Frozen final request and challenge.** The request binds the exact E0,
   calibration, pilot and serving bytes before final worlds. Execute candidate
   and incumbent across three mixes and N final seeds. The consumer requires
   complete 2 × 3 × N coverage and recomputes the final decision from bound
   records (`strict_promotion.py:1957`, `:1986`, `:2197`, `:2337`).

Strict success requires at least two superior mixes under one-sided Holm testing
at family alpha .05, scripted noninferiority with its lower bound strictly above
minus the calibrated margin, and all declared behavioral bands passing. The
combined descriptive interval is not a decision gate. Even a successful receipt
sets `release_action_authorized: false`
(`src/scripts/eval_stats.py:476`, `:530`, `:622`).

## Work that must be bounded before admission

Let D be the development-world count, P the pilot-world count and N the final
count. The evidence schedule contains **3D + 6P + 6N + 50 episodes**, before any
bridge qualification or shakedown. Source minima are D ≥ 1, P ≥ 2 and N ≥ 40,
so the smallest structurally possible schedule contains **305 episodes**. Its
full-horizon allowance is **1,525,000 world frames**. Final worlds alone account
for 240 episodes and a 1,200,000-frame allowance at N = 40; serving accounts for
50 episodes and at most 250,000 frames. These are episode-count minima and
H5000 frame ceilings, not claims that every episode must execute its full
horizon or measured runtime. They exclude qualification/shakedown work.

N has no finite upper ceiling in the current pilot rule. Before any new
admission, choose D, P, NI/MDE fractions, implemented behavioral metrics and
offsets, checkpoint pool, deployment world/roster width, and a maximum N with an
explicit fit-stop rule. A pilot that exceeds the ceiling must not cause
post-observation relaxation of the MDE or extension of an old window.

Also resolve actual wall-clock, inference, checkpoint-load/read, memory, CPU,
audit and handoff bounds. World frames do not equal model-forward calls or hero
transitions. The stock strict CLI freezes, builds a receipt, then validates the
receipt again, repeatedly opening bound inputs, checking checkpoint bytes and
deserializing candidate metadata (`src/scripts/tournament_eval.py:1051`). Those
physical operations and the bridge's own qualification need explicit accounting.
None of the counts in this plan is authorization to execute them.

## Reuse limits and admission decision

Completed training and audit receipts, declared checkpoint identities and
authenticated exclusion-ledger identities may remain prior provenance. The
completed H3000 research comparisons used source parents and task profiles, not
this incumbent, live H5000 profile and strict roster/artifact chain. They do not
replace strict calibration, pilot, serving or held-final evidence. The old
uint64 training-world ledger also cannot be pasted into uint32 strict seed
fields or silently truncated. Define the meaning of the declared training
namespace, preserve the historical ledger as supplementary provenance, and
establish the new exclusion policy explicitly.

Admission is blocked pending the faithful bridge and validator implementation,
bounded qualification design, complete identities/world/namespace decisions and
a prospective resource/time budget covering the complete chain. No existing
window is reopened. This review read source and documentation only, ran no
tests or numerical code, opened no checkpoint/raw-game/training-log artifacts,
and reran no existing audit.
