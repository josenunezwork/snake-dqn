# T34 shadow measurements

Owner-authorized measurement implementation of
`docs/research/specs/2026-10-19_t34_shadow_measurements.md`, bound to
`docs/research/final_run_20261019/DESIGN_V2_SHADOW.md`.
The spec and design remain unchanged. This package is a development diagnostic;
it cannot authorize efficacy, training, a strict gate, or a serving change.

## Execution choices fixed before measurement

- Mac CPU only. Each numerical process uses one thread, with at most two
  measurement/test processes at once. M3 always uses exactly one thread.
- Each measurement invocation has a wall cap no greater than 1,800 seconds.
  M4 has a 600-second cap. Completed units have create-only receipts; resume
  validates source/input identity and skips completed units. A capped or failed
  invocation remains recorded and is never reported as a completed measurement.
- No tmux, live service, other T34 worktree, checkpoint, or serving-pin changes.
  The primary checkout is used only for its existing interpreter/checkpoints.
- Branch commits are pushed during implementation. The completed branch is
  merged with current `origin/main` and published to main only after the required
  non-slow suite and focused verification.
- Full state identity includes RNG streams, ordered bodies/food, lifecycle
  counters, immutable checkpoint identities, and mutable opponent runtime state.
  There is no direct live-to-Grid loader; M1 uses seed/action replay.
- Path B LAD is bisection followed by a radius-three stride-one neighbourhood. A
  survival predicate is not mathematically monotone: this is a measured search
  heuristic and does not certify the global last avoidable decision. Ties in
  descending B8 Q order use the lowest action index.
- The owner-side pre-data addendum and Design Amendment 1 (main `105debee`)
  additionally require the teacher rule: stride-ten probes followed by stride-one
  refinement within +/-5 of the midpoint of every adjacent sampled survive-to-die
  transition. The midpoint convention covers the complete ten-decision gap.
  Both rules are compared with exact stride-one LAD truth, obtained by scanning
  backward from doom until the first surviving alternative. Shared exact-state
  outcomes are memoized across rules; report each rule's charged work separately
  from actual union work, including the truth measurement overhead. Teacher GO
  requires >=90% intervention recovery. Neither heuristic proves global LAD.
- Fresh LAD expansions use the T3 continuation engine. A saved outcome is reused
  only on its exact original pre-action trajectory; cached future rows after a
  historical intervention cannot be queried on the abandoned greedy trajectory.
- The preserved T3 producer hashes differ in the rollout helper and optimized
  simulator/anchor modules. Historical query outcomes therefore require a fresh
  replay validation before reuse. Those validation row-steps are charged to the
  actual run. Fresh outcomes (including terminal mass) are memoized across the
  two rules and exact truth; there is no unqualified historical cache shortcut.
- M2 reconstructs T3 histories to measure search and label coverage. It does not
  execute or score a new assisted policy, or produce paired efficacy estimates.
  New per-action mass fields describe continuation labels only.
- M3 reachability is a timing projection using the empirical T1 late-death
  distribution (49 deaths, missing LADs remain failures), D=300, and measured
  catch-up/search percentiles. It is neither observed online reachability nor
  efficacy. Timing includes host contention and is not a quiet-host benchmark.

## Historical denominator correction

The spec calls the T3 reference “54/64 interventions.” The immutable raw ceiling
receipts and `docs/research/t3_label_bank_2026-10-15.md` identify 54/64 as episode
survival. There are 75 action-change events across 22 of the 64 worlds. Report
all three counts explicitly; use the actual action-change denominator for
intervention recovery. Changes without a horizon-surviving alternative are
reported separately, because v1 maximized alive steps even when every action
died. Do not relabel 54 survivors as 54 interventions.

The approximate v1 budget is 400,000 row-steps/world. The actual preserved dense
ceiling used 55,002,897 / 64 = 859,420.265625 branch row-steps/world, plus ordinary
episode stepping. Report both comparators, with the registered >=5x bar applied
to the spec's 400,000 reference, not the larger historical amount.

Raw receipts and logs live under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/`.
`RESULTS.md` is the final, evidence-linked account of GO/NO-GO, commands, caps,
coverage, limitations, and artifact SHA-256 identities.
