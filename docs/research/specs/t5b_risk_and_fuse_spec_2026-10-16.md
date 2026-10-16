# Spec: Tier 1 risk estimate for B8 + fuse-layer per-action probe (2026-10-16, diagnostics only)

Read AGENTS.md, `docs/research/advice/2026-10-16_strategy.md` (sections 1 and 3, binding),
`docs/research/specs/guard_clause_amendment_2026-10-15.md` (Tier 1 clauses),
`docs/research/t5_separability_2026-10-16.md` and your T5 code.

Part 1, Tier 1 risk estimate (diagnostic, NEVER gate evidence; do not re-score anything as GO): pool
every no-veto probe the five B8 finals already have (census round 52 runs, M4 Phase R look-2 per-world
records for M4-selected/B8 and the incumbent, and any later B8 rows), and estimate per Tier 1 clause:
c2 scripted alive-frame vs served agent (pooled; threshold -0.03), frozen/mixed same, per-seed floor
(-0.15 vs served agent, 96 worlds per seed: failure probability per seed from the observed mean and
SE), absolute floor (pooled episode survival >= 0.50 per mix: failure probability by noise at 480
worlds), and the combined probability that B8 fails at least one clause. State clearly which numbers
reuse already-seen banks. Also report the Tier 2 status (champion comparison) from existing data.
Part 2, fuse-layer probe: on the restored return bank states (same folds as T5), for each B8 final
evaluate the 256-d fuse features (post-ReLU, the T1C hook) and the six Q values; fit GBM / logistic
regression on the per-action label "branch died within 300" with (a) fuse + Q, (b) fuse + Q + the 60
scalars + the 24 T5 features. AUC with world-bootstrap CIs, vs T5's 0.726 / 0.748. Advisor's reading:
>= 0.85 means the raster carries the information (learning failure); otherwise perception.
Doc `docs/research/t5b_risk_and_fuse_2026-10-16.md`. CPU only (MPS allowed for the forward pass if
faster; the Mac is otherwise idle), up to 8 processes; never touch tmux sessions or processes you did
not start. Git: branch t5-20261016; commit often; after each commit `git fetch origin && git merge
origin/main`, `git push origin HEAD:main`, `git push origin t5-20261016`.
Final message: per-clause risk table + combined risk, fuse probe AUC table + reading, sha.
