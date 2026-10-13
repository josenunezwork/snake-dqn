# M3-C result and M3-D diagnostics — learnings (2026-10-12)

Distilled record of the M3-C training run (2026-10-08/09) and the autonomous
M3-D diagnostic series (2026-10-10..12, branch `codex/m3d-autonomous-20261010`,
archived as tag `archive/m3d-autonomous-20261010`). The branch itself
(~211 commits, ~431k lines, mostly per-study governance receipts) was not
merged. Small M3-C evidence files are copied to
[m3c_m3d_evidence/](m3c_m3d_evidence/); the full artifact tree stays under
`snake-dqn-artifacts/redesign-m3c-*` and `snake-dqn-artifacts/redesign-autonomous-20261010/`.

**Bottom line:** goal (beat frp3-s12+v8 *without* a veto) unmet. Nothing promoted.
frp3-s12 + v8 remains the served champion.

## M3-C (the actual training run)

Setup: ego2s-b raster DQN, arms A / B / L x seeds 0-4 x 20M transitions, RTX 4090
pod (recovery attempt), ~$7.9 total including the stalled A1 attempt.
Probes: 32 worlds x 3 mixes (frozen / scripted / mixed).

| Arm | No-veto mass (mean of 5 seeds) | vs incumbent+v8 (434.9) | No-veto survival (scr / mix / frz) |
|---|---|---|---|
| A | 177.4 | -257.5 | 1% / 9% / 4% |
| B | 172.9 | -262.0 | 3% / 8% / 4% |
| L | 184.6 | -250.3 | 1% / 11% / 4% |
| incumbent + v8 | 434.9 | — | 66% / 75% / 63% |

With v8 on, arm B scored 447.7 (+12.8; survival in band) and all arms were
~435-448. Neither selection entered Phase R (`m3c_m3d_evidence/retry-recovery-closeout.json`).

### Learnings

1. **No-veto policy quality is the whole problem.** The redesign student cannot
   survive on its own (1-11% vs 63-75%), consistently across 15 runs.
2. **v8 masks policy quality.** Any student + v8 lands at roughly the
   incumbent's score. Development comparisons must be scored no-veto or they
   cannot discriminate candidates.
3. **Learning was still moving.** One-seed 16-world probes: A_s0 129 -> 153 -> 223
   at 5M / 10M / 15M (`10m-probe-summary.json`, `15m-probe-summary.json`). Final
   probes use a different world bank, so a plateau at 20M is not established.

## M3-D (autonomous diagnostics, $0, Mac CPU)

~70 bounded studies (S1-S41 plus ~30 named follow-ups) on why the distilled
student does not match the teacher. All closed negative or inconclusive; ten-plus
closed on operational/admission refusals before any fit.

### Learnings worth keeping

4. **Fits TRAIN, fails transfer.** The best fixed recipe (FUSE_LINEAR + heads)
   passes all TRAIN screens and fails supported fresh-world transfer (S32/S34).
   More correction-label coverage barely helps (S7 +2.5pp, S9 +1 representative,
   S40 +4 of 49 heldout). Points to generalization, not capacity.
5. **Errors are among allowed actions.** Every critical target is reachable under
   the advisory masks; failures are not caused by masking (S13). All 179 TRAIN
   correction labels are jointly expressible by changing only the C0 advantage
   head (S14) — the frozen features carry the information.
6. **Objective conflict.** Lowering ordinary-row KL 1.0 -> 0.1 improves critical
   agreement (e.g. 17/82 -> 32/82) but drops ordinary agreement below the 98% floor
   (S18); removing the ordinary objective entirely still misses both (S16).
   Patching a teacher with corrections fights preserving its behaviour.
7. **Value head is miscalibrated.** Material systematic *underestimation* of the
   policy's own realised returns on a frozen-policy population
   (native return calibration). Anything that bootstraps from it inherits the bias.
8. **Return-based fixes did not help gameplay.** Game-return TD, critic-free
   Monte Carlo, expert-trajectory imitation, current-v8 imitation and
   outcome-cost pilots all failed to beat the release no-veto; current-v8
   student -158 vs incumbent+v8 (97.5% CI -273..-42); outcome-cost -95
   (CI -214..+25, unresolved).
9. **Compute.** One unchanged native ego-raster update is ~107 ms on Mac CPU,
   dominated by convolution — real training of this network belongs on a GPU.

### Process lesson

Autonomous agents need a bias to execute: a study that never runs a fit should
not count as progress, and per-study governance (ratification, admission,
qualification, closeout receipts) must be capped relative to the experiment.
Many M3-D studies ended on wall-clock admission or audit-inventory refusals and
produced no scientific endpoint.
