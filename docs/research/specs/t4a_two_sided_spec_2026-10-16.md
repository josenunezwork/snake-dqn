# Spec: T4 step 1, two-sided bank loss (2026-10-16)

Implementer: Codex. Read AGENTS.md, then `docs/research/advice/2026-10-16_t4_levers.md` section (a)
and (c) (binding design), the T3 pre-registration, `docs/research/t4_prep_tooling_2026-10-15.md`
(bank v2 prefer-nonboost) and the `--label-bank*` code in `train_m3c.py` / `learner_m3c.py`.

## Code (default-off; default path byte-identical, prove with the existing traces/tests)
- `--label-bank-counter-mode {inert, single-action}` (default inert = today). single-action: counter
  rows (B8 right) use the single-action DQfD margin with B8's action as the label
  (`Q(a_B8) >= max_{a != a_B8} Q(a) + m`, as `anchor_loss_margin`); decisive rows keep the
  surviving-set margin.
- `--label-bank-counter-ratio R` (default: use the bank as stored): sample counter and decisive
  rows so counter:decisive = R:1 in each anchor batch (upsampling counters with replacement).
- Telemetry per 100k: `bank_agree_decisive`, `bank_agree_counter` (argmax in surviving set; for
  counter rows argmax == B8's action), plus the existing fields.
- Margin: reuse `--label-bank-margin-scale`; the arm uses scale 4.75 (0.0632 x 4.75 = 0.30).
- Counter matching: report whether bank v2's counters match the decisive rows on length, frame and
  -max Q (quantiles); if badly mismatched, add `--label-bank-counter-match` that reweights counters
  by inverse-density on (length bucket, frame bucket); default off.
- Tests: default unchanged; formula tests for both loss sides; tiny CPU training smoke with the
  real bank v2.

## Pre-registration (write before any run): `docs/research/specs/t4a_prereg_2026-10-16.md`
Arm TWO_SIDED: seeds 10-14 from B8_s0..4 finals, 8M, M6 recipe (`--td-huber-delta 1.0`), anchor =
half on-policy v8 + half bank v2 (`--onpolicy-labels --label-frac 0.5 --label-bank-frac 0.5`),
lambda 0.1 throughout (`--anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1`),
`--label-bank-counter-mode single-action --label-bank-counter-ratio 1 --label-bank-margin-scale
4.75`. CTRL = M6 CTRL finals. Fresh round 65 (seed audit), 96/32/32 worlds; same GO/KILL/EXTEND
rule as T3; readout pre-registered: bank_agree_decisive and bank_agree_counter at 8M (both > 0.9
-> inputs carry the distinction; decisive < 0.8 with counter holding -> inputs are the limit).
Also write the 2-lane Mac launch script (do not start it) to
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4a-mac-20261016/lane{0,1}.sh`.

Never touch tmux sessions or processes you did not start; CPU-only tests (no MPS) while T3 trains.
Git: branch t4a-20261016; commit often; after each commit `git fetch origin && git merge
origin/main`, `git push origin HEAD:main`, `git push origin t4a-20261016`.
Final message: what changed, test results, counter-matching numbers, launch script paths, sha.
