# Spec: RETURN screen probes + readouts (2026-10-16)

Read AGENTS.md, the binding `docs/research/specs/t4e_prereg_2026-10-16.md` and
`docs/research/specs/t4e_prereg_amendment1_2026-10-16.md`. Reuse your T4a/T4c probe + analysis code.
Runs (launched by Claude; NEVER touch tmux t4e-lane0/t4e-lane1 or files in
/Users/josenunez/Projects/ml/snake-dqn-t4d): `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4e-mac-20261016/RETURN_s{10..14}`.
1. Round 68 (extension 69; re-verify the audits incl. the bank-world exclusion), 96/32/32, CPU rowwise,
   at most 2 single-thread processes while lanes live (4 after). CTRL + incumbent first. tmux `t4e-probe`.
2. Decision exactly per the prereg (GO / KILL / EXTEND).
3. Readouts per Amendment 1: held-out decisive Q-gap median (fatal minus best surviving) at 8M vs B8
   and vs CTRL; training-set gap; held-out outside-state gap change; (c) consequential change vs the
   0.99% drift baseline; max|Q| and return_to_td_grad_ratio curves per 100k; shared-survivor length
   delta at 5000. State the applicable fork branch verbatim.
4. Doc `docs/research/t4e_screen_result_2026-10-16.md`, tests for new code, flake8.
Git: branch t4p-20261016; commit often; after each commit `git fetch origin && git merge origin/main`,
`git push origin HEAD:main`, `git push origin t4p-20261016`. Poll until the doc is pushed.
Final message: decision, primary delta/SE, mass per mix, all readouts, fork branch, sha.
