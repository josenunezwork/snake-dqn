# T5b risk and fuse-layer evidence (2026-10-16)

Retrospective diagnostics on reused banks only; never gate evidence or GO. The raw run outputs stay
at `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5b-risk-and-fuse-20261016/runs`.

**Cited by:** [t5b_risk_and_fuse_2026-10-16.md](../t5b_risk_and_fuse_2026-10-16.md).

**What it supports:**

1. The modeled survival-guard failure risk of **38.81%**, above the advisor's 25% flag. Tier 2
   still fails (**FAIL**).
2. Every B8 fuse+Q probe stays below 0.85 AUC (GBM 0.7630-0.7693). Under the advisor's rule, that
   reads as **perception**.

| File | Contents |
| --- | --- |
| `risk.json` | Risk calculations: per-checkpoint and per-clause forecasts, combined risk, sensitivity, Tier 2. |
| `risk-inputs.json` | Inventory of every inspected input bank, with SHA-256 and accepted observation counts. |
| `risk-independent-audit.json` | Independent risk audit: 1,221 raw sources, 1,984 canonical rows; `PASS`. |
| `fuse.json` | Fuse-layer separability arms, dataset, method and provenance (`t5b-fuse-evidence/v1`). |
| `fuse-independent-audit.json` | Independent audit of the saved fuse artifacts, with the world bootstrap re-run; `PASS`. |
| `artifact-manifest.json` | Implementation commits, raw artifact and baseline input hashes, audit statuses, session exit codes. |
| `verification.json` | Root verification: pytest, style and evidence checks (`t5b-root-verification/v1`). |
| `environment.json` | Versions, CPU device, 5 parallel fuse processes x 1 thread; no training or new gameplay. |
