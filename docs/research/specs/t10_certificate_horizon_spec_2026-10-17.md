# Spec: T10 certificate audit + horizon-stratified AUC (2026-10-17, diagnostics)

Read AGENTS.md and `docs/research/advice/2026-10-17_after_t9.md` (steps 1 and 2, binding), the T9 report and code
(time-aware tail reachability with TTL release, race-128 area), the T3 dense ceiling artifacts
(`t3-rollout-labels-20261015/dense`, per-world shadow trajectories) and the B8 census series
(`b8-death-census-20261014/runs`), and the T5/T8 bank + fitting code.

Step 1, certificate audit: for the dense-ceiling (lookahead) trajectories and the B8 census trajectories on the same
64 worlds (B8_s0, B8_s3; replay as needed), per decision at L >= 800 compute (a) time-aware own-tail reachability
(flood with TTL release reaches an own-tail cell before it would be needed) and (b) race-128 area (cells reached
before any opponent within 128 steps). Report: fraction of L >= 800 decisions holding (a), and holding (b) above
thresholds L/2 and L, under each policy; for each B8 death and each lookahead death, decisions before death at which
(a) was last true; distribution of certificate-loss duration before death; survival conditional on holding (a) at
L >= 800 for k consecutive decisions. Verdict line: does the lookahead hold the certificate far more often (state the
numbers), or does it also play mostly without it?
Step 2, horizon-stratified AUC: on the T5 bank (same folds, bootstrap), relabel per-action outcomes as death within
30, 60, 100, 200, 300 transitions; refit GBM for the 60 scalars, 60+24, the T8 arrival-race features (when T8 has
pushed them; else skip that column and say so) and, if available, the T5b fuse features. Plot/table AUC by horizon per
input set. Verdict: horizon ceiling (all decay alike to ~0.77) vs input-dependent decay.
Doc `docs/research/t10_certificate_horizon_2026-10-17.md`; tests; flake8. CPU only, at most 2 processes; NEVER touch
tmux t7-phase-r or processes you did not start. Git: branch t9-20261017; commit often; after each commit
`git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t9-20261017`.
Final message: both verdicts with numbers, sha.
