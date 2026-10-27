# F5 CLOSED — no fit, no numerical look (owner decision 2026-10-26)

Owner decision (verbatim, 2026-10-26 ~18:05 PT): "actually Stop and redirect: close F5 on the B, B-2 and B-3 evidence, with no training run, and move the effort to running search in served play and to GPU training."
(This supersedes the owner's message minutes earlier: "Keep finishing F5 as registered; it is cheap now and gives a clean registered answer.")

Disposition: F5 (exploratory W-vs-matched-control residual fit; registration F5_REGISTRATION_ADOPTED_20261025.md, selection W) is CLOSED before any fit. No C or W fit ran; no F5 evaluation world was played; the sole stage-look claim/token was never consumed and is hereby retired unused. F4 remains STOP_FUNDING. Nothing here is a scientific result about W; it is a stop decision on development evidence.

Basis (development-sandbox evidence only, never gate evidence; docs/research/governance_a100_sandbox_20261026/ADOPTED.md):
- Sandbox B (60 configs), B-2 (48 configs × 5 τ = 240 points) and B-3 (432 points incl. kNN ceiling, per-action geometry inputs and a decoupled survival veto) on frozen-B8 features, same fresh split a100-dev/sandbox-b/split-v1.
- All collapse onto one retention-vs-ordinary-flip-rate curve: best retention 0.587 (B-2, ~93% ordinary flips); at ≤1% ordinary flips only ~4.5% rescue retention (B-3); kNN ceiling on frozen phi ~0.15–0.17. 0/432 B-3 points met the pre-declared bar (retention ≥0.70, flip ≤1%, harm ≤1).
- Argon analyses: big-lever/exp/argon-20261026/sandbox-b-results-01, sandbox-b2-results-01.

State at closure: V6 engineering seal/package built (main a5bd00ea) but not reviewed or dispatched; step-1 qualification on V5 PASSED (published bc22e072, verified 4d64f1a2); long timing run PASSED operationally (published 93743a9a); reducer HOLD dispositions preserved. All F5 evidence and rulings remain immutable.

Redirect: effort moves to (1) served search on the owner's Mac and (2) GPU-resident training (sandbox A line).
