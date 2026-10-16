# Spec: T4 step 2, return-regression arm: bank collection + pre-registration (2026-10-16)

Implementer: Codex. Read AGENTS.md, `docs/research/t4c_screen_result_2026-10-16.md` (T4c = KILL; the
declared fork selected "return regression next, with outside-bank branches rolled out too"), your
tooling doc `docs/research/t4_return_regression_tooling_2026-10-16.md`, and the advisor notes
`docs/research/advice/2026-10-16_t4_levers.md` (b) and `2026-10-16_t4_tighten.md` section 3.

1. Pre-registration FIRST (before collecting): `docs/research/specs/t4e_prereg_2026-10-16.md`.
   Arm RETURN: the M6 CTRL recipe exactly (on-policy v8 anchor, `--label-frac 1`, lambda 0.1
   throughout, `--td-huber-delta 1.0`, no margin bank) + `--return-bank BANK --return-frac F
   --return-weight 1.0`; pick F so return rows are about 128 per update like the anchor batch, and
   say why. 5 seeds 10-14 from B8_s0..4 finals, 8M, `--device mps`. CTRL = M6 CTRL finals. Fresh
   round 68 (seed audit; 69 for the extension), 96/32/32 worlds, same GO/KILL/EXTEND rule.
   Readouts: Q(fatal) - Q(best surviving) on held-out decisive branches before/after; (c)
   consequential change vs the drift baseline; shared-survivor length delta at 5000; return-loss
   curve. Post-result fork (write it now): GO -> 20M retrain + Phase R under the two-tier guard;
   survival moves but mass fails -> tune return weight once; nothing moves -> inputs next (narrowest
   gap to the outside, k-frame boost-escape feasibility), per the advisor.
2. Collect the return bank on a fresh collection namespace (seed audit): B8 played (5 finals),
   75% scripted / 25% mixed + frozen as before; death-window decisive states AND matched
   outside-bank states, all legal actions rolled out D=300 with outcomes per the tooling. Targets:
   >= 8,000 decisive states and an equal number of outside states. Use up to 12 single-thread CPU
   workers (the Mac GPU is idle; nothing else is training), checking AC power. Report wall time.
   Hold out 10% of decisive states (by world) for the Q-gap readout.
3. Launch scripts (do NOT start): `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4e-mac-20261016/lane{0,1}.sh`
   with `--device mps`, NO single-thread caps on the training process beyond what T4a used, running
   from this worktree; `--check` mode must print the full command.
Tests for anything new; flake8. Never touch tmux sessions or processes you did not start.
Git: branch t4d-20261016; commit often; after each commit `git fetch origin && git merge origin/main`,
`git push origin HEAD:main`, `git push origin t4d-20261016`.
Final message: prereg path, bank stats (states, branches, wall time), launch script paths, sha.
