# Reorg proposals: scope 06 (evidence folders + panel_2026-10-15)

Proposals only; none of these were carried out.

1. **Give the M3-C folder a dated name.** Every other evidence folder is named
   `<study>_<date>_evidence/`. Rename `m3c_m3d_evidence/` to `m3c_2026-10-09_evidence/`, since it
   holds only M3-C receipts and no M3-D ones. Then update its links in
   `m3c_m3d_learnings_2026-10-12.md` and `ops/ARTIFACTS_INVENTORY_2026-10-20.md`.
2. **Keep the panel material in one place.** `panel_review_summary_2026-10-14.md` and
   `panel_mac_answers_2026-10-14.md` describe `panel_2026-10-15/` but sit outside it. Either move
   them into the folder or rename the folder to `panel_review_2026-10-14_15/`. Then update the links
   in INDEX.md and in the t1a, t2 and t3 specs.
3. **Group the loose T4a JSON receipts.** Four T4a receipts sit at the top level of
   `docs/research/`: `t4a_counter_matching_2026-10-16.json`, `t4a_cpu_smoke_2026-10-16.json`,
   `t4a_default_trace_2026-10-16.json` and `t4a_probe_launch_2026-10-16.json`. Move them into a
   `t4a_screen_2026-10-16_evidence/` folder.
4. **List T5 and T8 in the artifacts inventory.** `ops/ARTIFACTS_INVENTORY_2026-10-20.md` mentions
   the m3c, t10, t4a and t5b evidence folders, but not `t5_separability_2026-10-16_evidence/` or
   `t8_probes_2026-10-17_evidence/`.
5. **Consider moving the large T4a summary out of git.**
   `t4a_posthoc_2026-10-16_evidence/posthoc-summary.json` is ~2.9 MB, far larger than any other
   evidence file. It could live with the external NPZ captures, with only a SHA-256 pointer kept in
   the repo.
6. **Make the evidence README a convention (AGENTS.md).** Every new `*_evidence/` folder should ship
   a README that names the doc citing it, the verdict the evidence supports, and a file table, like
   `t3_diagnostics_2026-10-16_evidence/README.md` and the READMEs added in this scope.
