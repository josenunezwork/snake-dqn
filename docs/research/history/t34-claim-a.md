# T34 Claim A Bare-Student Distillation and Evaluation Pipeline

**Dates:** 2026-10-19 to 2026-10-21
**Verdict:** Closed `KILL` at the registered L4 `D0` diagnostic (`d2454013`) — shared-weight distillation flipped only `0.4%` of rescues, while split-stream objectives flipped `100%` of free Qs (`225/225`) yet broke held-out `b8-s3` anchor guards within `< 512` neural updates, leaving `R1` (`t34_claim_a_r1`) `UNUSED / CLOSED_WITHOUT_CONSUMER`.

## What was tried
- Built `t34_exit_b8_fit` (`S1b` bare `Ego2sNet` and `S1a` adding a zero-init 4-plane `152x88` full-arena conv tower with `max|dQ| == 0`), a once-per-round lossless `uint8` mmap cache (`12,777` B/row `S1b`, `66,281` B/row `S1a`), and parallel `32 x H5000` `DevProbe` (`4.0 ms`/decision/worker).
- Tested the v1 shared-weight objective (`KL` at `tau = 0.05` + `margin = 0.1` hinge + `0.01*adv_MSE`, `1:1` searched:unsearched quota) on `39` excluded A5 v2 worlds (`190,779` rows: `190,483` `shadow_confirm`, `225` decisive `horizon_rescue` in `21/39` worlds, `71` `all_die_delay`).
- Ratified and ran the `3`-fold × `9`-cell `D0` diagnostic (`t34_claim_a_loss`) testing split-stream losses (`L_A = KL + 0.01*adv_MSE` on `896` anchors; `L_D = tau*CE + hinge` with no self-KL on `128` decisive rows).
- Swept `trainable` B8 and frozen-B8 `residual` heads over `U* in {32, 128, 512, 1024, 2048, 4096}` updates against `>= 70%` decisive flips and anchor guards (`<= 5%` length-decile disagreement, `<= 1.5x` boost ratio, `<= 90%` single-action share).

## What happened
- **v1 shared-weight algebraic lock:** Optimizing unconstrained free per-row Q-values for `5,000` Adam steps (`lr = 1e-2`) flipped only **`0.4%`** of decisive rows to `a*` (**`97.8%`** stayed bare `b8-s3`), and the `1:1` sampler ran only `~1` update/epoch on `39` worlds.
- **Split-stream `D0` rerun (`KILL`):** Removing self-KL from decisive rows achieved **`100%` (`225/225`) free-Q flips** (`1.0` vs `0.0044` frozen, passing `>= 99%` eligibility), but **every neural cell (`trainable` and `residual`) failed the held-out anchor guards at `32..4096` updates** (none held `>= 512` updates), leaving `800+200` `R1` unused.

## Why
Applying `KL(softmax(Q_prev/0.05) || softmax(Q/0.05))` (`1/tau = 20`) on decisive rows cancelled the `0.1`-margin hinge; once split streams fixed free-Q flips, rare `horizon_rescue` states (`225 / 190,779`) overlapped ordinary states in single-frame space so gradient updates broke normal `b8-s3` anchor behavior before `512` steps.

## What to reuse / what not to repeat
- Never apply a parent-policy self-KL term on decisive teacher-override rows; separate anchor (`shadow_confirm`) and decisive (`horizon_rescue`) loss streams and drop zero-weight `all_die_delay` rows.
- Reuse once-per-round `uint8` mmap observation caches (`12,777` B/row, `1.8 s` fit) instead of per-epoch replay regeneration, and test free-Q + anchor spillover (`D0`) before funding `800+200`-world label banks (`R1`).

## Full record
Original: `git show 90b3db94:docs/research/history/t34-claim-a.md` | Primary sources: [2026-10-21_t34_claim_a_amendment.md](../specs/2026-10-21_t34_claim_a_amendment.md), [t34_claim_a/README.md](../../../research/t34_claim_a/README.md), [t34_claim_a_r1/README.md](../../../research/t34_claim_a_r1/README.md)
