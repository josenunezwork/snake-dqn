# REL — Qualify an exact candidate for deployment

**Status:** PLANNED. **Earliest wave:** 10. **Role:** verifier. **Kind:** experiment.

**Dependencies:** X0, G0, E2, S1

Evaluate one selected immutable candidate against the pinned incumbent in the deployed task.

Read [decisions and wave plan](../README.md), [agent runbook](../agent_runbook.md), and [dispatch.json](../dispatch.json) before starting. You are not alone: edit only this package's claimed files and inherit accepted dependency commits.

## Exclusive write ownership

- `docs/experiments/rl_candidate_promotion_2026-09-06/` — new

## Implementation / execution steps

1. Before opening final worlds, freeze candidate/incumbent/opponent hashes, source/profile, roster rotations, pilot-derived n, statistical hypotheses and absolute noninferiority margin. At least 40 fresh worlds are required; honor any larger powered n.
2. Use the live engine for cross-stack comparison, the 5000-frame deployment profile, all required mixes and balanced per-world rosters. Keep observation normalization 5000 explicit.
3. Run50 separate serving-path episodes from the same snapshot/profile for compatibility, including human-plus-AI dispatch. They are not extra independent training seeds or a substitute for the paired effect estimate.
4. Emit strict readiness/statistical/decision receipts. No champion file, default or remote deployment is changed by this experiment. Any later promotion action must use this exact passing receipt and preserve rollback assets.
5. Bind readiness to the exact evaluation/serving source: a pre-L1 PQN qualification can use its G0-qualified source; every post-L1 source requires G1 even when the candidate is PQN. A PPO candidate also requires L1 and X2. Any later material source change invalidates affected receipts and requires requalification before final scoring.

## Acceptance evidence

- All E2 readiness, powered seed, superiority, noninferiority and calibrated behavioral requirements pass on untouched final data.
- If PPO is selected, L1, X2 and G1 receipts for the exact post-integration source are mandatory. Otherwise record that branch as not applicable with the selected algorithm/head identity; do not silently waive it.
- A failed or inconclusive gate leaves the candidate experimental and retains the full result.
- A fixture with a PQN candidate and post-L1 evaluator source cannot qualify using only old G0/E2 receipts; it requires a matching G1 receipt. A pre-L1 PQN path remains independent of optional PPO work.

Use the exact frozen commands from the execution protocol once the dependency entrypoints exist. Do not invent an already-available CLI flag or run a long command during implementation.

## Conditional prerequisites

[
  {
    "when": "selected_candidate.algorithm == ppo",
    "requires": [
      "L1",
      "X2",
      "G1"
    ]
  },
  {
    "when": "The source SHA used by REL includes L1 or later changes to the shared PPO-integrated runtime/evaluator, regardless of the selected candidate algorithm.",
    "requires": [
      "G1"
    ]
  }
]

## Compatibility and rollback

Keep A5 and its prior deployment available. Never substitute another checkpoint after passing.

## Handoff

Write command/result logs, source/config/checkpoint hashes, changed files, commit SHA, unresolved risks and interface changes to `<EVIDENCE_ROOT>/packages/rel/`. The root resolves EVIDENCE_ROOT to an absolute durable path outside disposable worktrees before dispatch. Obtain a root-issued test/compute slot before running commands. Return absolute artifact paths and hashes; root verifies the durable manifest before accepting the dependency or cleaning a worktree. A passing narrow test does not waive an upstream gate or promote a model.

Audit source: [expanded state/system review](../../../state_and_system_review_2026-09-06.md).
