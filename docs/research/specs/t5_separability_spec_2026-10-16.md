# Spec: T5 input features, separability test before any training (2026-10-16)

Implementer: Codex. Read AGENTS.md, `docs/research/advice/2026-10-16_inputs_design.md` (binding design:
features section 1, test section 2), `src/simd_env/v8_nb.py` (the 30 v8 per-action features and the
tail-aware flood), `src/simd_env/ego_raster_b.py` (the ego2s-b scalars, which ALREADY include
uncapped tail-aware region sizes for the three next cells: region_frac_* / region_log_*; note this
in the doc and make sure the "60 scalars" baseline contains them), the T4e return bank
(`t4e-return-bank-20261016`, per-action branch outcomes), the T4b snapshot store
(`research/t4_snapshot_restart_20261016/`, `t4b-snapshot-restart-20261016/full`) and T1's LAD rows
(`t1-commitment-20261015/rows.jsonl`).

1. Features in plain NumPy (no numba yet), per action (6 actions) x 4 = 24: area_log (uncapped
   post-move tail-aware reachable area, log2(min(a,4096)+1)/12), boost3_area_log (3 boosted frames
   straight along the action's direction, own body advanced, boost cost applied, opponents static,
   0 on any collision, else uncapped area from the landing cell), core_area_log (flood treating
   cells with <= 2 free 4-neighbours as impassable), tail_reachable (flood touches an own-tail cell
   with TTL <= 1). Unit tests on hand-built grids for each feature.
2. Restore the return bank's states (train decisive + outside + holdout) from their worlds (snapshot
   store or deterministic replay; prove a few restored states match the bank's stored observations
   bit for bit) and compute the 24 features plus the existing 60 scalars.
3. Separability, grouped folds by world: gradient boosting (sklearn HistGradientBoosting if present,
   else LightGBM/XGBoost if installed, else a documented alternative) on (a) per-(state, action)
   "branch died within 300" and (b) per-state "boost-only escape" with: 60 scalars; 60 + 24; 24
   alone. AUC with 2000-sample bootstrap CIs by world. Also logistic regression for robustness.
4. T1 check: on the 46 late LADs, the share where the best surviving action has higher area_log or
   boost3_area_log than the chosen action (bar: >= 70%).
5. Verdict per the advisor's bar (60+24 beats 60 by >= 0.05 on per-action death AND 24-alone >= 0.80),
   reported verbatim with the numbers. No training, no numba port.
Doc `docs/research/t5_separability_2026-10-16.md`. CPU only, up to 8 single-thread processes (the
Mac is otherwise idle; never MPS). Never touch tmux sessions or processes you did not start.
Git: branch t5-20261016 in /Users/josenunez/Projects/ml/snake-dqn-t5; commit often; after each commit
`git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t5-20261016`.
Final message: AUC table with CIs, T1 share, verdict, sha.
