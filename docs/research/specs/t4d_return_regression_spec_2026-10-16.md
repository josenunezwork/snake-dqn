# Spec: T4 step 2 tooling, return regression on rolled-out branches (build only, 2026-10-16)

Implementer: Codex. Read AGENTS.md, then `docs/research/advice/2026-10-16_t4_levers.md` section (b)
(the target formula, binding) and `docs/research/advice/2026-10-16_t4_tighten.md` section 3 (the
fork: return regression with outside-bank branches rolled out too), the T4 prep tooling doc
(`--record-outcomes`, `--label-policy`), the death-window collector `research/t3_rollout_labels_20261015/`,
the expert-iteration driver `research/t4_expert_iteration_20261015/`, and the T4a/T4c bank code in
`train_m3c.py` / `learner_m3c.py`. Build tools only; do NOT launch training or a long collection.

1. Collector: per rolled-out branch record T, died, S = sum_{t<T} gamma^t Phi(L_{t+1}) alive_{t+1}
   (raw, before kappa(1-gamma)), L0, L_T, and for survivors the packed terminal observation + mask
   (so the trainer can bootstrap). Also roll out outside-bank states (L >= 700, >= 300 decisions from
   any death, matched like the T4c outside bank) for ALL legal actions, so ordinary states get
   magnitude targets. Default-off flags; deterministic; tests vs a hand-computed branch.
2. Trainer (default-off, default path byte-identical; prove with the existing traces/tests):
   `--return-bank PATH --return-frac F --return-weight W`: an extra loss term, Huber (delta as the
   TD loss) on Q(s0, a) - y_a for every recorded action, with
   y = kappa(1-gamma) S - Phi(L0) + gamma^T Phi(L_T) [survivors] - 3 gamma^(T-1) [died]
   + gamma^T V(s_T) [survivors], V(s_T) = Q_target(s_T, argmax_a Q_online(s_T, a)) under the mask
   (double-DQN bootstrap). Verify the formula against the trainer's actual reward assembly (mass term
   + PBRS gamma*Phi(L') - Phi(L), Phi = L/10, Phi(dead) = 0, death -3) with a unit test that
   rolls a short branch in the sim and compares the discounted sum of trainer rewards to y.
3. A 2-world collection smoke (CPU) and a tiny CPU training smoke with the real format.
4. Doc `docs/research/t4_return_regression_tooling_2026-10-16.md`: formula, fields, timing per 1,000
   branches, the exact collection + arm commands (launch scripts must use `--device mps`; do not run).
CPU only for your own work (never MPS), at most 3 single-thread processes; never touch tmux sessions
(t4c-lane*, t4c-probe, git-sync) or processes you did not start.
Git: branch t4d-20261016; commit often; after each commit `git fetch origin && git merge origin/main`,
`git push origin HEAD:main`, `git push origin t4d-20261016`.
Final message: what was built, formula test result, timing, commands, sha.
