# Spec: RETURN arm weight calibration + prereg amendment (2026-10-16, before any RETURN run)

Read AGENTS.md, `docs/research/advice/2026-10-16_return_arm_review.md` (binding for this step),
`docs/research/specs/t4e_prereg_2026-10-16.md`, and the return code in train_m3c.py / learner_m3c.py.

1. Telemetry (default-off flag or always-on log fields that do not change the math): per 100k,
   the gradient-norm ratio ||grad of W*return_loss|| / ||grad of td_loss|| (shared params, measured on
   the same update before clipping), `max_abs_q`, and the batch composition (anchor rows, return
   rows, agent TD rows). Default path byte-identical (existing traces/tests).
2. Calibration: from each of B8_s0 and B8_s2 finals, with the RETURN recipe and the real train bank,
   run short smokes (a few hundred updates; MPS is allowed and preferred for realism, the Mac is idle;
   if MPS is unavailable use CPU and say so) for W in {1.0, 0.2, 0.05, 0.02, 0.01}; record the ratio
   at initialisation (first ~50 updates) and its trend. Choose W so the init ratio is about 0.5
   (geometric interpolation allowed; round to 2 significant figures). Confirm the batch holds 128
   anchor rows AND ~128 return rows (not 128 split).
3. Amend the pre-registration (section "Amendment 1, before any RETURN run"): the measured W and the
   calibration table; the advisor's readouts verbatim with thresholds (held-out decisive Q-gap
   median <= -3 Q; training-set gap within 2x; held-out outside gap change within +/-0.5; (c) within
   2 points of the 0.99% drift baseline; max|Q| per 100k reported); the batch composition result;
   confirm round 68/69 seed audits exclude the bank's worlds.
4. Update `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4e-mac-20261016/lane{0,1}.sh`
   to the chosen W (still `--device mps`, unstarted); `--check` must show it.
Never touch tmux sessions or processes you did not start. Git: branch t4d-20261016; commit often;
after each commit `git fetch origin && git merge origin/main`, `git push origin HEAD:main`,
`git push origin t4d-20261016`. Final message: calibration table, chosen W, batch composition,
amended prereg path, sha.
