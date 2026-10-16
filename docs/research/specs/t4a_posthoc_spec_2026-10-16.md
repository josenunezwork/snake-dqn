# Spec: boost reconciliation (T3) + post-hoc spillover/growth diagnostics (T4a) (2026-10-16)

Read AGENTS.md, `docs/research/advice/2026-10-16_t3_diag_note.md`, the T4a pre-registration's
"Post-hoc diagnostics" section, and your T3 diagnostics tooling (`docs/research/t3_diagnostics_2026-10-16.md`).
Diagnostic only. CPU only, at most 3 single-thread processes, never MPS. NEVER touch tmux
t4a-lane0/t4a-lane1/t4a-probe/t4b-build/git-sync or files in /Users/josenunez/Projects/ml/snake-dqn-t4a.

Part 1 (now, T3): reconcile the boost numbers. For LABELS and KEEPV8 finals (seeds 10-14) and CTRL:
greedy boost-argmax rate on the arm's OWN round-62 probe trajectories vs on CTRL's trajectories,
per length bucket (<400, 400-700, 700-1000, >=1000), plus the share of the arm's own decisions in
each bucket. Also confirm which action set the training telemetry `greedy_boost_rate` counts. State
which reading holds (definition difference vs state-distribution split = caution in behaviour space).

Part 2 (after all five TWO_SIDED finals exist: watch
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4a-mac-20261016/lane{0,1}-status.txt`):
the declared spillover readout (a/b/c, per -max Q decile, cross-boost share) and the growth curve,
for TWO_SIDED, with LABELS and KEEPV8 from T3 alongside for contrast. Apply the declared
interpretation text verbatim.

Docs: append Part 1 to `docs/research/t3_diagnostics_2026-10-16.md` (new section) and write Part 2
to `docs/research/t4a_posthoc_diagnostics_2026-10-16.md`. Tests for new code; flake8.
Git: branch t4prep-20261015; commit often; after each commit `git fetch origin && git merge
origin/main`, `git push origin HEAD:main`, `git push origin t4prep-20261015`. Poll until Part 2 is
done. Final message: Part 1 reading, Part 2 table + interpretation, sha.
