# Spec: T8 probes, raw-raster CNN + arrival-race (Voronoi) features (2026-10-17, diagnostics only)

Read AGENTS.md, `docs/research/advice/2026-10-17_clever_ideas.md` (ideas 1 and 3, binding), your T5/T5b code and
report (same restored bank states, same per-action label "branch died within 300", same world-grouped folds and
bootstrap). No policy training; no gate rescoring.

Probe 1, raw-raster CNN: train a small CNN from scratch (two small conv towers over the packed ego2s-b local 7x31x31
and global 4x37x37 planes + the 60 scalars + one-hot action, a few hundred thousand params, early stopping on an
inner validation split by world) to predict per-action death within 300 on the T5 bank, 5 world-grouped folds,
AUC with world-bootstrap CIs. Report vs T5 (scalars 0.726, 24 features 0.748) and T5b (fuse 0.77). Advisor's
reading: >= 0.85 -> the raster carries the signal (B8's trunk drops it); ~0.77 -> the raster lacks it.
Probe 3, arrival-race features: per action, from the restored state after the hypothetical move (opponents moving
1 cell/frame, heads as BFS sources, hero from its next cell; tail-aware passability as in T5): Voronoi area
(cells the hero reaches strictly first; log2(min(a,4096)+1)/12, uncapped), contested fraction of the action's
tail-aware region (cells some opponent reaches first or at the same time), and the hero's margin to the nearest
contested boundary cell. Unit tests on hand-built grids. Rerun the T5 GBM/LR with 60 + 24 + these; report AUC
deltas with CIs and the T1 LAD share (surviving action has larger Voronoi area than the chosen one).
Doc `docs/research/t8_probes_2026-10-17.md`. CPU only (MPS allowed for the CNN if faster), at most 2 processes in
total; NEVER touch tmux t7-phase-r or any process you did not start. Git: branch t5-20261016; commit often; after
each commit `git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t5-20261016`.
Final message: both AUC tables with CIs, the readings, sha.
