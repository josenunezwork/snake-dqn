# Spec: T4a screen probes + analysis (2026-10-16)

Read AGENTS.md, the binding pre-registration `docs/research/specs/t4a_prereg_2026-10-16.md`, and
reuse `research/t3_screen_20261015/` (extend it; do not fork blindly).
Runs (launched by Claude; NEVER touch tmux t4a-lane0/t4a-lane1, and NEVER modify files in the
worktree /Users/josenunez/Projects/ml/snake-dqn-t4a — its lanes import that code per seed):
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4a-mac-20261016/TWO_SIDED_s{10..14}`,
status in `lane0-status.txt` / `lane1-status.txt`. CTRL = M6 CTRL finals; incumbent baseline.

1. Round 65 (as pre-registered; seed audit already in the prereg; re-verify), 96 scripted / 32
   frozen / 32 mixed per checkpoint, the T2/T3 probe path, CPU rowwise, at most 2 single-thread
   processes while t4a lanes live, up to 4 after. Probe CTRL finals and the incumbent first, then
   each TWO_SIDED final as it lands. Watcher in tmux `t4a-probe`.
2. Analysis exactly per the pre-registration (GO / KILL / EXTEND with the one-time extension), plus
   the pre-registered readout: `bank_agree_decisive` and `bank_agree_counter` at 8M from each run's
   log.jsonl, with the verdict text the prereg defines. Secondaries as in T3 (alive-frame, mass vs
   incumbent per mix, death lengths, greedy boost curve).
3. Also report T3's final arms next to TWO_SIDED (from the T3 analysis on main) for context only.
4. Doc `docs/research/t4a_screen_result_2026-10-16.md`; tests for new analysis code; flake8.
Git: branch t4prep-20261015; commit often; after each commit `git fetch origin && git merge
origin/main`, `git push origin HEAD:main`, `git push origin t4prep-20261015`.
Keep polling (~10 min) until the doc is pushed. Final message: decision with primary delta/SE, mass
per mix, per-side bank_agree readout verdict, commit sha.
