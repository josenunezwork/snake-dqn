# Spec: T13b six-snake roster measurement + roster rule + monitoring erratum (2026-10-17)

Read AGENTS.md and `docs/research/advice/2026-10-17_after_t13.md` (binding: steps 1-3 and the decision rule).
Implementer: Codex. Reuse your T13 harness.
1. Write the decision rule into `docs/research/t13b_six_snake_roster_2026-10-17.md` BEFORE running (commit + push first):
   at six snakes, keep all-B8 if the all-B8 hero's deaths per 1,000 are within 0.10 of the mixed roster's AND its p95
   length is within 100 cells of the FRP3 hero's; otherwise the mixed roster (b8-s3 hero, frp3-s12 + v8 opponents)
   becomes the planned Watch default, subject to a separate roster qualification (not in this job).
2. Run the same four rosters at the default six snakes (configs/mechanics_v2.yaml num_snakes 6), 25 H5000 episodes
   each, fresh paired seeds disjoint from T7/T12/T13 banks, plus all-B8 at eight snakes (25 episodes). Same table and
   paired bootstrap CIs as T13. Apply the rule verbatim and state the outcome.
3. Monitoring erratum (do not edit the release record; add an erratum section/file): the 0.848 trigger is the six-snake
   Watch row; add a Watch12 row with baseline 0.768 and trigger 1.536 (and Watch8 from step 2 with 2x its baseline);
   update `scripts/monitor_b8_release.py` to apply triggers per driven roster size; tests.
Serving defaults unchanged. CPU, at most 6 processes. Never touch tmux sessions or processes you did not start.
Git: branch t13-20261017; commit often; after each commit `git fetch origin && git merge origin/main`,
`git push origin HEAD:main`, `git push origin t13-20261017`. Final message: six-snake table, rule outcome, erratum, sha.
