# Spec: T6 RESTART screen probes + analysis (2026-10-17)

Read AGENTS.md and the binding `docs/research/specs/t6_prereg_2026-10-16.md` (and its preparation report's
note that the seed audit must be refreshed before evaluation). Reuse your T4 probe/analysis code.
Runs (finished): `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t6-mac-20261016/RESTART_s{10..14}`.
Ignore `smoke_200k_s10`. CTRL = M6 CTRL finals; incumbent baseline.
1. Refresh the seed audit for rounds 72/73 (exclude the 319 snapshot source worlds and every previous bank,
   including the running T7 Phase R namespace). Probes: 96/32/32 per checkpoint, CPU rowwise, at most 2
   single-thread processes while tmux `t7-phase-r` is alive, up to 6 after. CTRL + incumbent first. tmux `t6-probe`.
2. Decision per the prereg (GO / KILL / EXTEND at round 73). Readouts per the prereg (training deaths/1k at
   L >= 800 restarted vs fresh from telemetry; held-out decisive Q-gap on the T4e holdout at 8M vs B8;
   shared-survivor length delta at 5000; boost curve).
3. Doc `docs/research/t6_screen_result_2026-10-17.md`; tests for new code; flake8.
Never touch tmux t7-phase-r or any process you did not start. Git: branch t4p-20261016; commit often; after each
commit `git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t4p-20261016`.
Poll until the doc is pushed. Final message: decision, primary delta/SE, mass per mix, readouts, sha.
