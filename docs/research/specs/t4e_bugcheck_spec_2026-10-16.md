# Spec: RETURN halt bug check (2026-10-16)

Read AGENTS.md, the amendment's "Deviation 1" in `docs/research/specs/t4e_prereg_amendment1_2026-10-16.md`,
the return tooling doc and code. Halted runs: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4e-mac-20261016/HALTED_boost_RETURN_s1{0,1}`
(log.jsonl, ckpt if any). Do the checks below, report evidence, fix any genuine bug with a regression
test, but do NOT relaunch training.
1. Action identity: the action index stored per branch in the bank vs the trainer's action space
   (which indices are boosts, relative vs absolute directions, the mask). Prove with a replayed state:
   apply bank action k in the sim and confirm the branch's first frame matches the recorded one.
2. Target statistics in the bank: per action class (boost vs non-boost) and per family (decisive,
   outside), the distribution of targets y (with the bootstrap evaluated by B8 and by the halted
   checkpoints), survival rate, and mean y gap boost minus non-boost. Is boost systematically higher?
3. Target vs trainer reward: re-check gamma, kappa, Phi, death value and boost burn/trail effects
   between the collector's S/G and the trainer's per-step reward on a boost branch specifically.
4. The trainer loss: confirm Q(s0, a) is indexed with the bank's action, the bootstrap uses the mask
   of s_T, return rows' observations are the right states (obs/action alignment), and W/delta/frac
   are applied as logged.
5. Q-scale: why max|Q| stays ~25 (CTRL 110-140): compare the return targets' range with CTRL Q on the
   same states.
Doc `docs/research/t4e_bugcheck_2026-10-16.md` with each check's verdict. CPU only for your work
(MPS allowed only for short evals if needed), at most 4 processes; never touch tmux sessions or
processes you did not start. Git: branch t4d-20261016; commit often; after each commit `git fetch
origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t4d-20261016`.
Final message: per-check verdict, any bug fixed (+ test), sha.
