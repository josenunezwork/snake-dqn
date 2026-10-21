# Reorg proposals 19: legacy research packages (2026-10-20)

Source catalog: [`docs/research/catalog/legacy_packages.md`](../../catalog/legacy_packages.md). Proposals only;
nothing was moved. All of `research/**` is hash-pinned by ratified studies, so every move below needs an owner
decision and a check that no live receipt hashes the old path.

1. **Promote the 7 REUSABLE-LIVE dependencies out of dated study folders (top proposal).** T34/T35 import
   `research.apex_safety_20260926.dev_screen`, `research.b8_death_census_20261014.census`,
   `research.apex_veto_serving_20261001.serving_run`, `research.compute.thermal_guard`,
   `research.perf_sim_20261007.with_slots`, `research/survival_band_v2_20261006/survival_pools_20261006.json`
   and `research/fixed_search.py`. Proposed: after T34 closes, copy (not move) these into a stable
   `research/common/` (or `src/research_common/`) package and point new studies there; leave originals in place
   for old receipts. Avoids new work depending on a 2026-09-26 "development screen" folder.
2. **Do not delete DEAD packages.** Tests (`tests/test_apex_veto_*`, `test_safety_veto_v3..v7`,
   `test_death_census_v8`, `test_m7_*`, etc.) and later packages' hashed source lists reference them. A future
   option: archive DEAD packages plus their tests into an `archive/` tree together, in one commit, after
   checking no open study hashes them.
3. **Group the Apex veto line.** 17 `apex_veto_*` folders plus `apex_safety`, `trap_horizon`, `death_census_v8`,
   `serving_time_search`, `simd_parity_*` could live under `research/legacy/apex_veto/` (path-change caveat as above).
4. **Group perf tooling** (`perf_*`, `actor_perf_20261017`) under `research/perf/`; rename
   `actor_perf_20261017` to `perf_actor_kernels_20261017` for prefix consistency.
5. **Record missing verdicts.** `perf_actor/learner/eval/opp/profile/ram/compile/podbench`, `actor_perf`,
   `simd_parity_v7v8`, `serving_time_search`, `ensemble`, `planner_feasibility` have no explicit verdict line
   found in their own files; add one-line outcome notes in `research/INDEX.md`.
6. **Stale comments in web/backend.** `web/backend/safety_veto_serving.py` comments still name the v7/v8 lanes;
   the served default is now B8 s3 without a veto (2026-10-17). Worth a comment refresh in a web scope.
