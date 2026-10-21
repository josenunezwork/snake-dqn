# Reorg proposals — scope 12: redesign M3 / M3-C (2026-10-20)

Proposals only; nothing here was applied. Catalog: [../../catalog/redesign_m3_m3c.md](../../catalog/redesign_m3_m3c.md).

1. **Promote the live M3-C trainer out of a dated study package (top proposal).**
   `research/redesign_m3c_20261008/{train_m3c,learner_m3c,obs_pack,demo_store,league_pool,
   league_random_safe,*_snapshot,*_checkpoint,compile_m3c,q0b_timing,profile_m3c,task_collection}.py`
   are the shared training stack for S1 / T-series work, ~70 test files and `s1_mature` import or
   hash them. Proposal: a future, non-ratified-time move to a stable home (e.g. `src/train_ego/`)
   with a thin re-export shim left in the old path so ratified study hashes and imports keep
   working. Must not happen while T34 / S1 studies pin these paths.
2. **Add a README.md stub to each package** pointing to the catalog and saying
   "closed study; trainer modules live". Requires an owner exception to the research/** freeze.
3. **Record the M3-C final result next to the package**: the outcome is only in
   `docs/research/m3c_m3d_learnings_2026-10-12.md` / `m3c_m3d_evidence/`. Add a ledger pointer
   from `INDEX.md` 2026-10-08 rows to the catalog (doc-only edit).
4. **Ledger gap**: `results/diag/noveto_summary.json` (M3-B no-veto probes) has no INDEX row; add one.
5. **Dead code, keep in place**: `train_m3.py`, v1 `gpu_run.py`/`gpu_agent.py`, `slotrun.py`,
   `m3b_*`, `bench_*`, `_DRAFT.md` files — no deletion (ratified evidence); just mark "dead" in
   the catalog (done).
6. **Fix INDEX evidence links** that use branch-prefixed pseudo-paths (`m3c:...`, `redesign-scope:...`)
   to plain relative links now that the files are on main.
