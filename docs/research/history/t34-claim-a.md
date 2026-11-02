# T34 Claim A Bare-Student Distillation and Evaluation Pipeline

## Question & Goal

Can teacher horizon-rescue labels (`teacher_v2` shadow-LAD counterfactual rollouts) be distilled into a single bare `Ego2sNet` student (plain `S1b` or `S1a` with a 4-plane $152 \times 88$ full-arena conv tower or 256-D residual head) so that it flips `>= 70%` of held-out decisive rescues to `a*` without breaching the `FINAL_PLAN.md` §8 anchor stability tripwires (`<= 5%` action disagreement vs `b8-s3` in every length decile, `<= 1.5x` boost-share ratio, and `<= 90%` single-action share)?

## Timeline of Versions & Steps

| Date | Step / Version | What Was Run / Built | Verdict / Status |
|---|---|---|---|
| 2026-10-19 | v1 ExIt fit harness & P5 tower prep (`research/t34_exit_b8_fit/`) | Built P4 shared-weight distillation loss (`KL(softmax(Q_prev/0.05) \|\| softmax(Q/0.05)) + hinge(margin=0.1) + 0.01*adv_MSE`), 1:1 searched:unsearched sampler, streaming per-epoch replay regenerator, and P5 4-plane `152x88` full-arena conv tower (`S1a`, `16-32-64-64 -> 256`, zero-init fuse columns with `max\|dQ\| == 0` vs B8). | `BUILT` |
| 2026-10-19 | v2 lossless `uint8` fit cache (`research/t34_exit_b8_fit/cache.py`) | Replaced per-epoch replay regeneration with a once-per-round hash-validated `uint8` memory-mapped cache (`12,777 B/row` = `~12.4 GB/1M rows` for `S1b`; `66,281 B/row` = `~66 GB/1M rows` for `S1a`) and v2 shadow/LAD weighting. | `BUILT` |
| 2026-10-20 | Claim A end-to-end pipeline & Mac smokes (`research/t34_claim_a/`) | Built student process fan-out (`label.py`), `ExportReplaySource` (`replay_source.py`), parallel `32 x H5000` `DevProbe` (`probe.py`), `SEED_AUDIT_CLAIM_A.json`, and ran `smoke-01`..`06` (`43.0 s` total for `smoke-06`). Identified that `smoke-01` stopped at Epoch 1 (`0/10` decisive agreement) and that 39 A5 worlds yielded only `~1` update/epoch under the 1:1 quota. | `RISK_IDENTIFIED` |
| 2026-10-21 | Claim A objective, dose, D0 & R1 amendment (`2026-10-21_t34_claim_a_amendment.md`) | Fable-approved pre-data amendment (`research/t34_claim_a_loss/`, `research/t34_claim_a_r1/`) proving the v1 shared-weight loss flipped only `0.4%` of rescues even with unconstrained free Q; introduced split-stream losses (`split`, `split_tau2`), update-dose `U* in {512, 1024, 2048, 4096}`, 9-cell D0 diagnostic, and 800-train / 200-heldout R1 plan (`r1-labels-v4`). | `APPROVED` (Conditions C1–C6) |
| 2026-10-21 | L1 D0 loss-diagnostic rerun on L4 (`d2454013`, `research/t34_claim_a_loss/`) | Executed the registered 3-fold $\times$ 9-cell D0 diagnostic (`frozen_recipe`, `frozen_{trainable,residual}`, `split_{trainable,residual}`, `split_tau2_{trainable,residual}`) on the 39 excluded A5 v2 worlds (`16.8 MB` tarball, SHA `82fd839d...`). | `KILL` |

## Final Verdict & Key Numbers

- **Final Verdict:** **`KILL`** on the Claim A projection line (2026-10-21 registered D0 rerun `d2454013`), and **`UNUSED / CLOSED_WITHOUT_CONSUMER`** on the R1 label collection package (`research/t34_claim_a_r1`) after Round 0 closed on `INTEGRITY_FAILURE` and D0 returned `KILL` ([`docs/research/specs/2026-10-21_t34_claim_a_amendment.md`](../specs/2026-10-21_t34_claim_a_amendment.md), [`research/t34_claim_a_r1/README.md`](../../../research/t34_claim_a_r1/README.md), [`docs/notes/t34-campaign.md`](../../notes/t34-campaign.md)).
- **Algebraic Lock of the v1 Shared-Weight Objective (`finding 1`):**
  - Across the 39 excluded A5 v2 worlds (**`190,779` labelled rows**: **`190,483`** `shadow_confirm`, **`225`** decisive `horizon_rescue` across `21/39` worlds, **`71`** `all_die_delay`), optimizing the v1 shared-weight loss (`KL` at $\tau = 0.05$ + `margin = 0.1` hinge + `0.01` adv-MSE sharing `sample_weight`) directly on unconstrained per-row free Q-values for 5,000 Adam steps at `lr = 1e-2` flipped only **`0.4%`** of decisive rows to `a*` (**`97.8%`** remained at bare `b8-s3`), even when decisive row weight was raised 10$\times$ ([`docs/research/specs/2026-10-21_t34_claim_a_amendment.md`](../specs/2026-10-21_t34_claim_a_amendment.md) §1).
  - Furthermore, the v1 1:1 searched:unsearched quota trained only `~600` rows/epoch (`~1` optimizer step/epoch on 39 worlds; `~60` steps/fit on 800 worlds), and `all_die_delay` rows consumed anchor quota while carrying zero loss weight ([`research/t34_claim_a/README.md`](../../../research/t34_claim_a/README.md), [`docs/research/specs/2026-10-21_t34_claim_a_amendment.md`](../specs/2026-10-21_t34_claim_a_amendment.md) §1).
- **Split-Stream Objective & Registered D0 Diagnostic (`d2454013`):**
  - Separating the batch (`1,024 = 128 decisive + 896 anchor`) into count-normalized streams (`L_A = KL + 0.01*adv_MSE` on `shadow_confirm` anchors; `L_D = tau*CE + hinge` on `horizon_rescue` rows with no self-KL) achieved **`100%` (`225/225`) free-Q flips**, passing free-Q eligibility (`>= 99%`).
  - However, when trained through neural parameters (`trainable` B8 at `2.5e-5` / `1e-4` or frozen-B8 `residual` head at `1e-4` / `1e-3` across `U* in {32, 128, 512, 1024, 2048, 4096}` updates in 3 world-stratified folds), **no corrected cell maintained the held-out anchor guards (`<= 5%` length-decile disagreement, `<= 1.5x` boost ratio, `<= 90%` one-action share) for `>= 512` updates** (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-d0-20261021/`, [`docs/notes/t34-campaign.md`](../../notes/t34-campaign.md), [`docs/research/final_run_20261019/CAMPAIGN_LOG.md`](../final_run_20261019/CAMPAIGN_LOG.md)).
- **Pipeline & Cache Measurements (`research/t34_claim_a/README.md`):**
  - Lossless `uint8` fit cache size: **`12,777` bytes/row** (`S1b`, peak RSS `1.39 GB`) and **`66,281` bytes/row** (`S1a` with 4-plane $152 \times 88$ tower, peak RSS `2.90 GB`).
  - Single-world A5 diagnostic cache replay (`2,105` kept rows): **`3.5 s`** on 1 Mac thread (`13.5 KB/row`, `max |Q_recorded - Q_B8| = 7.2e-5`; [`docs/research/specs/2026-10-21_t34_claim_a_amendment.md`](../specs/2026-10-21_t34_claim_a_amendment.md) §1).
  - Parallel `DevProbe` speed: **`4.0 ms` per decision per worker** on Mac (`smoke-06` total `43.0 s`: `15.1 s` label, `0.08 s` payload, `17.2 s` cache build, `1.8 s` fit, `8.6 s` probe; `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/`, [`research/t34_claim_a/README.md`](../../../research/t34_claim_a/README.md)).

## What Was Learned

1. **Self-KL on Decisive Rows Algebraically Cancels Rescue Supervision:** Sharing `sample_weight` between `KL(softmax(Q_prev/tau) || softmax(Q/tau))` (gradient scale `1/tau = 20` at $\tau = 0.05$) and the margin-`0.1` hinge locks `Q` to `Q_prev` on decisive states (`0.4%` flip rate even with unconstrained per-row Q). Decisive rows must never carry a self-KL term toward the failing parent policy.
2. **Feature Collision Between Decisive and Ordinary States Causes Anchor Spillover:** Even with a clean split-stream objective (`100%` free-Q flips), gradient updates on `128` decisive rows/batch through `Ego2sNet` (or a 256-D residual head) breach the `<= 5%` length-decile disagreement and `<= 1.5x` boost-share guards before reaching `512` updates, because rare horizon-rescue states overlap ordinary states in single-frame representation space.
3. **Once-Per-Round `uint8` Fit Caching Eliminates Replay Bottlenecks:** Materializing regenerated observations once per round into a hash-verified `uint8` mmap cache (`12.8 KB/row` for `S1b`) reduced multi-epoch training time to seconds and made CPU probe/replay the sole dominant cost.

## Why It Closed / Current Status

Closed on 2026-10-21 when the registered L1 D0 diagnostic (`d2454013`) returned `KILL` under §4.4 of the Claim A amendment (`neither trunk keeps the guards for >= 512 updates`), terminating the Claim A bare-student projection line (`L3`) and leaving R1 (`research/t34_claim_a_r1`) unused.

## Kept Files in This Lineage

- [`docs/research/specs/2026-10-21_t34_claim_a_amendment.md`](../specs/2026-10-21_t34_claim_a_amendment.md) — Prospective Fable-approved Claim A amendment defining the split-stream objective, update-dose `U*`, L1 D0 9-cell diagnostic, and 800+200 R1 label plan.
- [`research/t34_claim_a/README.md`](../../../research/t34_claim_a/README.md) — End-to-end Claim A student fan-out, `ExportReplaySource`, parallel `DevProbe`, seed audit, and Mac smoke measurements.
- [`research/t34_claim_a_r1/README.md`](../../../research/t34_claim_a_r1/README.md) — Package README for the 800-train / 200-heldout R1 label plan and sealer (`SEED_AUDIT_R1.json`).
