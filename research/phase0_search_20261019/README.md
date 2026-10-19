# Phase 0 search teacher

Implementation of `docs/research/specs/2026-10-19_phase0_search_teacher.md`.
The pre-registration is a draft. No selection episodes have been launched by
this implementation task. All measured data are disjoint preflight evidence.

`teacher.py` takes the world constructor from T3's captured strict evaluator
and retains its terminal-hero GridBatchSim, Watch preparation order and carried
Vector61 opponent state. Search evaluates every legal first action. The bare B8
outer action retains rowwise advisory-mask inference and no veto. Rollout batch
forwards are part of the search policy recipe; their Q values are not claimed
bit-identical to rowwise inference.

The score is an undiscounted sum of realized post-transition logical mass for H
frames, zero after death, plus `leaf_scale * max_legal Q_B8` at the next prepared
leaf decision. Default leaf scale is **1 mass-frame per Q unit**. This is an
explicit heuristic conversion, not a calibrated estimate of future mass; B8 Q
was trained against a different reward. Reported Phase 0 endpoints always use
realized H5000 mass with no leaf estimate in the outcome.

`bank.py` allocates all purposes under separate keys and collision-audits every
world before gameplay. Selection has 96 scripted / 32 frozen / 32 mixed worlds;
calibration, timing, parity and smoke worlds never enter the decision estimator.
`analyze.py` applies the complete-bank one-look decision, equal weights for each
mix, paired world-level uncertainty, and STOP before GO if clauses overlap.

`runner.py` binds a create-only plan and immutable per-world paired partials,
verifies resumes and merges in bank order. Local execution uses one math thread
per process, at most six workers, AC/open lid and thermal checks. A scientific
verdict requires the complete selection bank; missing or invalid evidence cannot
be silently omitted. Run outputs belong outside Git.

`parity.py` compares a reserved replay subset and records float32 tie evidence.
Same-platform serial/parallel identity is exact. Cross-platform differences must
be explained at identical pre-states; a small Q tie does not justify ignoring
arbitrary later trajectory changes. Actual CPU-pod parity and throughput remain
unverified until separately authorized operation.

## Preflight measurements and operation

Pending final preflight receipts. Exact commands, selected parameters, runtime
projections and limitations will be added after local verification.
