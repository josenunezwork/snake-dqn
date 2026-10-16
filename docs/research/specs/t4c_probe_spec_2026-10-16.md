# Spec: T4c screen probes + analysis (2026-10-16)

Read AGENTS.md and the binding pre-registration `docs/research/specs/t4c_prereg_2026-10-16.md`
(incl. Deviation 1). Reuse your T4a probe/analysis code in `research/t3_screen_20261015/` (extend).
Runs (launched by Claude; NEVER touch tmux t4c-lane0/t4c-lane1 or files in
/Users/josenunez/Projects/ml/snake-dqn-t4c): `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4c-mac-20261016/TWO_SIDED_T_s{10..14}`
(status `lane{0,1}-status.txt`). IGNORE `ABORTED_cpu_TWO_SIDED_T_s10` entirely.
1. Round 66 (re-verify the seed audit), 96/32/32 worlds, CPU rowwise probe path, at most 2
   single-thread processes while lanes live, 4 after. CTRL finals + incumbent first. tmux `t4c-probe`.
2. Decision exactly per the pre-registration (GO / KILL / EXTEND at round 67). Readouts per the
   prereg: per-family bank agree at 8M (retained decisive, in-window counter, outside counter);
   (c) consequential-change rate vs the drift baseline using the T4c-drift tooling; shared-survivor
   length delta at 5000; then state which branch of the declared fork applies, verbatim.
3. Context: TWO_SIDED (T4a) and the T3 arms alongside.
Doc `docs/research/t4c_screen_result_2026-10-16.md`; tests for new code; flake8.
Git: branch t4p-20261016; commit often; after each commit `git fetch origin && git merge origin/main`,
`git push origin HEAD:main`, `git push origin t4p-20261016`. Poll until the doc is pushed.
Final message: decision, primary delta/SE, mass per mix, readouts, fork branch, sha.
