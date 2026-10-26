# Standing A100 Development Sandbox — ADOPTED by owner 2026-10-26

Owner decision (verbatim, 2026-10-26 ~15:00 PT): "Adopt Fable's Standing A100 Development Sandbox rule as written and start job A (resident collector+learner feasibility) and job B (successor head/loss sweep on a fresh salt split), without touching F5."

Adopted text: section 2 of FABLE_ON_ARGON_VS_ASTRA_20261026.md (copied alongside; sha256 8ab55d631d8655d36a2ca51b5e5418efb95ebb1e5758f70c2f68a4f7bd07f18d). Inputs to the decision: Argon review (big-lever/exp/argon-20261026/project-review-a100-02/answer.md), Astra fact-check (big-lever/exp/astra-20261024/ASTRA_X_ARGON_A100_PROJECT_20261026.md).

## 2. Governance: adopt this rule text

**Standing A100 Development Sandbox (one-time ratification; no per-job judge review).**

1. *Scope.* A job is a sandbox job if it (a) runs through the qualified supervisor and `tools/ops/dispatch.sh`, (b) reads only inputs on the **Sandbox Input Registry** (initially: closed F0 training rows D256/D1024 excluding F4 fold-0, the 29 census parents, R30 validation parents, and every strict/probe/serving/F5-evaluation bank; and worlds seeded from `uint64(sha256("a100-dev/" + study + "/" + idx))`), (c) writes only checkpoints, engineering timings, or development labels tagged `simulator-D`, and (d) has a finite endpoint (steps or hours) and a 2× runaway timeout.
2. *Process.* Sandbox jobs need an automated preflight (tests + manifest schema + registry check) and a committed 10-line manifest (inputs, salt, endpoint, timeout, excluded studies). **No pre-registration, judge review, timing canary or ratification per job.** They may be chained from a committed ready queue.
3. *Authority.* Sandbox outputs are Tier-1 development evidence at most: never gate evidence, never a label source for a registered study without that study's own premise ruling, never a change to F5 or any running registered study.
4. *Crossing.* At most **one** frozen checkpoint SHA per development line crosses into Mac evaluation (a pre-declared Holm family of ≤2 is allowed if declared before the sweep). Crossing is where the judge appears: freeze weights, commit the registration, fresh salt-derived paired bank, publish-before-read, one look. Mac strict gate, serving qualification and owner release are unchanged.
5. *Isolation.* Sandbox jobs never share an account assignment with a running registered job's manifest and never edit the qualified transport. While F5's timing qualification runs, allocate only if the active F5 contract permits coexistence (3-assignment cap), else wait.
6. *Seeds.* All new studies use domain-separated hash seeds plus an incremental canonical exclusion registry. Retroactive historical collision audits are required only for confirmation banks and are a registry intersection, not an archaeology project.

That is Argon's two zones with Astra's four guardrails (input registry, excluded studies, finite endpoint, one-candidate crossing). It drops Astra's "defer sweeps until F5 terminal" and "no auto-chaining"; both cost calendar time without protecting a look.

