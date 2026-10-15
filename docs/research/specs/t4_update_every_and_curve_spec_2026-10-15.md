# Spec: --update-every flag + T3 checkpoint-curve probes (2026-10-15)

Implementer: Codex. Read AGENTS.md and the advisor note
`docs/research/advice/2026-10-15_throughput_t4.md` (sections 1.1 and 1.2), plus the T3
pre-registration `docs/research/specs/t3_screen_prereg_2026-10-15.md`.

## Part 1: `--update-every M` (default 1 = unchanged)
In `train_m3c.py` (multi-actor path) and `learner_m3c.py` (reference and fast paths): perform one
learner update per `M x 128 x batch_scale` ingested agent transitions, same batch size, same target
sync counted in updates (document it). Default 1 must be byte-identical (run the existing default
traces/tests: tests/test_m4a_trainer.py, tests/test_perf_learner.py, tests/test_td_huber_delta.py,
tests/test_t3_rollout_labels.py). Log `update_every` and record it among the non-default options.
Add tests. Do NOT run the A/B training now (the Mac GPU is busy); write the A/B command (one seed,
M=1 vs M=2, 8M from B8_s0 with the LABELS_KEEPV8_LO flags, quiet Mac) into the doc.

## Part 2: T3 effect-vs-length curve (diagnostic, not a screen decision)
For each finished T3 run in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t3-mac-20261015/`
(and as more finish, until all 15 are done), probe `ckpt_2000k.pth`, `ckpt_4000k.pth`,
`ckpt_6000k.pth` and `final.pth` on 32 scripted worlds of round_index 64 (verify unused with the
seed audit; do NOT use 62/63, which belong to the T3 screen), no veto, the T2 probe path, CPU
rowwise, up to 4 single-thread processes (8 of 18 cores are busy; never use MPS). Also probe the 5
M6 CTRL finals and their ckpt_4000k on the same worlds as the reference. Report per arm the paired
(by seed) scripted episode/alive-frame survival and mass vs CTRL at 2M/4M/6M/8M, and whether 4M
carries >= 80% of the 8M effect. This never changes the T3 decision; say so in the doc.

Doc: `docs/research/t4_update_every_and_curve_2026-10-15.md`. Never touch tmux
t3-lane0/t3-lane1/t3-probe/git-sync or processes you did not start. Run Part 2's watcher in tmux
`t3-curve`; poll until all 15 runs are probed.
Git: branch t4prep-20261015; commit often; after each commit `git fetch origin && git merge
origin/main`, `git push origin HEAD:main`, `git push origin t4prep-20261015`.
Final message: Part 1 test results and A/B command; Part 2 curve table and the 4M verdict; commit sha.
