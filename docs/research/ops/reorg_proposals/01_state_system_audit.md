# Reorg proposal 01 — `docs/experiments/state_system_audit_2026-09-06/`

Scope: 57 files (evidence bundle for the 2026-09-06 state and system audit).
Proposals only. No files were moved, renamed, deleted or edited.

## Findings

- **A README already exists**
  ([README.md](../../../experiments/state_system_audit_2026-09-06/README.md), added in commit `6986fd35`).
  It already says what the audit was: pinned source `c1cf7e8`, base main `950d583`, no product-source edits, no training.
  It also lists each lane in an evidence-scope table and gives caveats and a reproduction recipe.
  It was **not overwritten** (org30 rules allow only link fixes to existing markdown).
- **The README files are hashed evidence.** `artifact_manifest.json` lists both `README.md` (line 6) and
  `evaluation/README.md` (line 108) with SHA-256 values. `completion_receipt.json` also pins
  `docs/state_and_system_review_2026-09-06.md` and `docs/rl_review_2026-09-06.md`. Editing any of them, even to add a
  banner, invalidates recorded hashes. Treat the folder as frozen.
- **Links:** every relative link in the three markdown files resolves. The links are `README.md` →
  `../../state_and_system_review_2026-09-06.md` and `../../pqn_sampler_results_2026-09-06.md`.
  `evaluation/README.md` and `vector/actor_vs_full_obs.md` have no relative links. No fixes were needed.
- **Status (from other docs; the bundle itself states none):** [docs/README.md](../../../README.md) labels the
  parent reviews "Historical audits". The audit fed the
  [RL repair plan](../../../plans/rl_repair_2026-09-06/README.md) and the
  [repair execution](../../../experiments/rl_repair_execution_2026-09-08.md); see the
  [research INDEX](../../INDEX.md) row for 2026-09-06–08. No document names a specific "superseded-by" file.

## Reading order (for an index entry, since the README cannot be edited)

1. [state_and_system_review_2026-09-06.md](../../../state_and_system_review_2026-09-06.md), "Decision" section.
   It says to keep vector61 Apex frozen, repair versioned contracts, then run small single-axis comparisons.
2. Bundle [README.md](../../../experiments/state_system_audit_2026-09-06/README.md): the evidence-scope table and caveats.
3. `completion_receipt.json` and `archive_replay_receipt.json`: integrity checks (100 source files, 0 mismatches,
   archive replay PASS).
4. `verification/major_claims_probe.json`: independent corroboration of four major claims.
5. Per-lane receipts (`vector/`, `raster/`, `serving/`, `config/`, `pqn/`, `apex/`, `dynamics/`, `evaluation/`,
   `instrumentation/`).
6. `prior_audit/` last. It holds older evidence recorded at different source hashes.

## Proposals (owner/main to decide)

1. **Add the reading order and a "historical" status to an index, not to the bundle.** The index could be
   `docs/research/INDEX.md` or `docs/README.md`. Adding them there leaves the hashed README untouched.
2. **Stale run path in `evaluation/README.md` (note only).** It runs scripts at `runs/state_system_audit_20260906/...`
   from the `snake-dqn-pqn-coverage` worktree, which differs from the top-level README's staging recipe.
   Because the file is hashed, document the difference in an index entry instead of editing it.
3. **Do not move the folder** into `docs/research/` or an archive tree. More than 30 docs link to it by path
   (`docs/plans/rl_repair_2026-09-06/**`, `docs/research/INDEX.md`, `docs/research_findings_2026-09-24.md`,
   and both 09-06 reviews), and the receipts embed original paths.
4. **Optional future grouping:** if September audits are ever gathered under `docs/audits/2026-09/`, move the two root
   reviews and this bundle together and update every inbound link in one commit. Record that move in a new receipt
   rather than editing `completion_receipt.json`, whose `report_sha256` keys are paths.
