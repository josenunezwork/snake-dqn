# Reorg proposals: loose files in docs/research/ (scope 07, 2026-10-20)

These are proposals only. Nothing was moved or renamed. Every move needs a check that no ratified study, prepare
step or hash binding refers to the current path. For example, the earlier index says the RunPod amendment is
hash-bound by template v3 `prepare`.

1. **Move the 37 T4 JSON receipts into a per-study folder**, for example `t4_2026-10-15_receipts/`. Use
   `git mv` and grep for path references first. These are `t4*_*.json`, `t6_*.json` and the two `q0_*.json`.
   They make up about a third of the loose files and are machine evidence, not reading material.
2. **Move T4 tooling and implementation notes** (`t4_*_tooling`, `t4a_implementation`, `t4c_implementation`,
   `t4c_data`, `t6_preparation`, `t11_implementation`) next to their receipts. Only result files would stay loose.
3. **Group the governance amendments** under `governance/`. This is blocked by the RULES freeze on
   `docs/research/governance_*` and by hash binding, so it needs owner approval.
4. **Group the task-aligned 6102 files from 2026-09** under `archive_2026-09_task_aligned/`. All are drafts that
   were never admitted.
5. **Fix stale header labels with sibling errata, not edits.** `governance_amendment_sequential_gates_2026-10-02.md`
   and `governance_amendment_paired_bands_2026-10-03.md` say "proposed amendment" in the header, but the earlier
   index records them as ratified. Check where each one says it was ratified before you add an erratum.
6. **Update GLOSSARY.md**, which stops at 2026-10-08. Add T-series terms such as LAD, fuse, RETURN_C, TWO_SIDED_T
   and Watch roster.
7. **Fix one naming inconsistency.** The `q0_*_20261018.json` files use a compact date, while every other file
   uses `YYYY-MM-DD`.
