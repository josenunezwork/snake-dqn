# Reorg proposals — scope 11: `research/redesign_m2_20261008/`

Catalog: [../../catalog/redesign_m2_20261008.md](../../catalog/redesign_m2_20261008.md). Proposals only; nothing moved.

1. **Fix `research/INDEX.md` row** (top): it says "Verdict not recorded in package". Replace with
   "M2 FAIL (LB −43.7, `results/ni_check/verdict.json`); M2b PASS (LB +34.8, `results_m2b/ni_check/verdict.json`)"
   and link the catalog. Doc-only, safe.
2. **Do NOT move or rename the package or any code file.** `ni_spec.py`, `dev_probe.py`, `gen_data.py`,
   `ni_spec_m2b.py` are imported by ~80 sites in later studies, and all 12 code files are sha-pinned in
   ratified manifests (t34 REGISTRATION_ROUND0/VERIFICATION_A6*, t12/t12b/t15 intents, t7 plan). A move breaks both.
3. **Longer term (new code, not a move):** promote the shared seed/namespace helpers in `ni_spec.py` and the probe
   harness in `dev_probe.py` into a `src/` eval module with a thin re-export left in place, so new studies stop
   importing from a dated research package. Requires a fresh registration for any study that re-pins.
4. **Superseded preliminary results** (`results_m2b/prelim_live_pre_b84d4de/`,
   `results_m2b/bench_g1_ego2s_b_pre_unionfind.jsonl`): keep (owner rule against deletion; possibly hash-pinned via
   results), but label them in a package README if one is ever added.
5. **Add a short `README.md` in the package** (needs owner OK since `research/**` is frozen): point at the two
   prereg files, the two verdicts, and the catalog.
