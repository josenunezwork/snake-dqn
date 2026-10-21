# Reorg proposals: scope 02 (docs/experiments/solo_food_*)

Done in this pass: created [docs/experiments/SOLO_FOOD_SERIES.md](../../../experiments/SOLO_FOOD_SERIES.md)
(78 folders + the progress map), and rewrote 59 absolute in-repo links
(`/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/...`) in 34 READMEs to relative links.
Before the rewrite these links worked only in the owner's main checkout. Zero relative links were broken.

Proposals (not executed):

1. **Move the series into one folder.** Move `docs/experiments/solo_food_*` (78 dirs plus
   `solo_food_learning_progress_2026-09-21.md`) under `docs/experiments/solo_food/`, with
   SOLO_FOOD_SERIES.md as its README. The flat `docs/experiments/` listing is dominated by this series.
   Moving it requires a `../` link rewrite in every README and in the sibling `solo_*` records that link in.
2. **Handle the ~814 absolute links into `/Users/josenunez/Projects/ml/snake-dqn-artifacts/...`.**
   They resolve only on the owner's Mac. Either add a one-line note to SOLO_FOOD_SERIES.md that evidence lives
   in the external artifacts tree, or adopt an `ARTIFACTS:` prefix convention. Do not try to fix these in place.
3. **Fix a dangling link.** `solo_food_coordinate_routing_2026-09-20/README.md` links to
   `/Users/josenunez/Projects/ml/snake-dqn/runs/solo-food-pqn-value-offset-implementation/design.md`.
   `runs/` is not tracked, so the link is dead in every checkout. Either point it at the artifacts copy or drop the link.
4. **Add an index for the sibling `solo_*` series.** Covers `solo_native_*`, `solo_ambient_*`, `solo_budget_curve`,
   `solo_exploration_floor`, etc., and links to `solo_learning_status_2026-09-20.md` and
   `research_stopping_point_2026-09-21.md`. They precede and feed this series but are out of scope 02.
5. **Add study codes to folder names, or to a front-matter line.** The codes (AM..DA) are the main
   cross-reference key inside the READMEs, but the folder names omit them. Thirteen records carry no code
   (the twelve 09-13 records and `pqn_enemy_dose`).
