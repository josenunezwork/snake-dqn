# Reorg proposals — scope 13 (redesign packages other than M2 / M3 / M3-C)

Proposals only; nothing below was applied. Catalog: [../../catalog/redesign_other.md](../../catalog/redesign_other.md).

1. **Fix `research/INDEX.md` row for `redesign_m4_20261013`** (top proposal). It says "verdict not
   recorded here"; the verdict is **PARTIAL**, recorded in `docs/research/m4_phase_r_result_2026-10-13.md`.
   Add that link and the verdict. Doc-only, no package bytes touched.
2. **Promote `research/redesign_scope_20261007/grid_h5000_identity.py` helpers to a shared module.**
   `env_digest`, `strip_timing`, `canonical`, `_context` are imported ~95 times by tests and later
   packages (t33, t34, q0q1, t12, s1). A ratified-hash-safe route: add a new `src/simd_env/identity.py`
   (or `research/common/`) that re-exports them, migrate new code to it, and leave the original file
   untouched (ratified studies hash it).
3. **Mark `redesign_m6_20261014` and `redesign_t4_20261015` as dead plan-only packages** in INDEX.md
   (T4 is already UNUSED; M6 is HISTORICAL but its plan was never used for the delivered Mac run).
   Optionally move both plan JSONs into a single `research/archive/pod_plans/` folder once no
   ratified study hashes their paths (check first; do not move now).
4. **Fold `redesign_m4a_20261012` + `redesign_m4b_20261012` under one "M4 training inputs" entry**
   in INDEX.md, cross-linked to `docs/research/m4a_design_2026-10-12.md`; no file moves
   (`widen_bv8` is imported by tests).
5. **Add a short `README.md` to each in-scope package pointing to its result doc** — only if the
   ratification/hash rules allow new files in non-ratified packages (scope, m4a, m4b, m6, t4); never
   in `redesign_m4_20261013` or `redesign_t7_phase_r_20261016`.
