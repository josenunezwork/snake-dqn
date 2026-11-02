# T35 Auxiliary Outcome-Head Probe and Two-Deviation Rescue Probe

## Question and Goal

Following the October 20, 2026 A100 strategy consultation ([`t34-a100-advice.md`](t34-a100-advice.md)), two falsifiable diagnostic probes were pre-registered under `T35` on 2026-10-20 before committing major GPU search or multi-step teacher training compute:

1. **T35 Auxiliary Outcome-Head Probe (`research/t35_outcome_head`):** Can the frozen served `b8-s3` (`Ego2sNet`) 256-D trunk representation (`fc1` post-ReLU), or the raw 60-D scalar observation input (`obs_60`), linearly predict imminent trajectory catastrophe (`y_cat = 1[t_death - t <= 32]`) on held-out rollouts better than the scalar danger baseline (`D_danger = 1 - min_{a in {0,1,2}} r_reach(a)`, the complement of the best normal-speed `v8` flood-fill reachability ratio) and current value/margin outputs (`V`, `Q_max`, `Q_margin`, `H_pi`)? If the frozen trunk already encodes upcoming traps (`AUROC(fc1) - AUROC(D_danger) >= +0.03` and `Recall@FPR10(fc1) >= 0.45`), a cheap auxiliary outcome head or selective search trigger could detect traps early; if not, the trunk is representation-blind to multi-step traps until the last moment.
2. **T35 Two-Deviation Rescue Probe (`research/t35_two_deviation`):** On natural `b8-s0`/`b8-s3` deaths where 1-deviation open-loop counterfactual sweeps (`T34` `lad_p0_cf_20261019`) failed to rescue the hero across the entire `[t_death - 200, t_death - 1]` window (`rescued == False`), can a **pair of single-step policy overrides** `(t_1, a_1), (t_2, a_2)` at two distinct lead frames (`t_1 < t_2`), returning to greedy `pi_0` at all other frames, rescue the trajectory (`survive >= 600` frames past `t_death` or reach frame `5000`)? This directly tested whether natural unrescued deaths are **2-switch traps** (which sparse multi-step open-loop deviations could solve) or **sustained multi-step control failures** requiring closed-loop step-by-step replanning.

---

## Timeline of Steps and Versions

| Date | Step / Package | What Was Run | Verdict |
|---|---|---|---|
| 2026-10-20 | `research/t35_outcome_head` (`PREREGISTRATION.md`) | Pre-registered 4-tier hierarchy of 5-fold grouped-by-episode CV probes (`T0` existing scalar baselines `D_danger`, `-V`, `-Q_max`, `-Q_margin`, `H_pi`; `T1` L2-logistic on `obs_60`; `T2` L2-logistic on frozen `fc1` 256-D; `T3` L2-logistic on `concat[fc1, obs_60, V, Q_max, Q_margin, H_pi]` 320-D) across 600 fresh `b8-s3` self-play episodes (`seeds 75000..75599`, max 3000 steps) at horizons `H in {16, 32, 64}`. | **PRE-REGISTERED** (`b9302620`, 0 Colab units, CPU-only). |
| 2026-10-20 | `research/t35_outcome_head` (`RESULT.md`) | Executed 600-episode rollout + 5-fold grouped CV (`snake-dqn-artifacts/t35-outcome-head-20261020/registered-02/result.json`, sha256 `62b644b9ba2c49e9ee6147ce822316b1c8fd2f3366c8e0f105b51fd754258b76`). Sampled `482` episodes (`360` death episodes, `122` censored survivals), `16,892` frames (`3,995` positive at `H=32`, base rate `23.65%`), wall time `158.9 s`. | **NO-GO (`representation_blind`)** — At primary `H=32`, `T0 D_danger` achieved `AUROC = 0.6592` (`Recall@FPR10 = 0.3401`); `T2 fc1 (256-D)` reached `AUROC = 0.6540` (`Delta = -0.0052` vs `+0.03` bar, `Recall@FPR10 = 0.3059` vs `0.45` bar). |
| 2026-10-20 | `research/t35_two_deviation` (`PREREGISTRATION.md`) | Pre-registered two-stage coarse-to-fine 2-deviation search (`L=180`, stride-2 `t_1` x 5 non-greedy actions = `450` Stage-A branches, top `K=12` by failure time `t_fail`, fine stride-1 `t_2` in `[max(t_1+1, t_fail-90), t_fail-1]` x 5 actions, top `M=8` full `H_confirm=600` confirmation) on the 7 natural `lad_p0_cf_20261019` unrescued events (`t_death >= 3500`), with pre-declared `EXTEND` rule to 19 natural events from `lad_c_20261019` (`N=26` pooled) if rescues landed in `{1, 2}`. | **PRE-REGISTERED** (`321ba7c7`, 0 Colab units, Mac CPU 3 workers). |
| 2026-10-20 | `research/t35_two_deviation` (`RESULT_PRIMARY.md`, consolidated here) | Ran Stage 1 primary probe on the 7 `lad_p0_cf_20261019` unrescued natural deaths (`snake-dqn-artifacts/t35-two-deviation-20261020/primary/summary-primary.json`, sha256 `69acca9d11f76cb11337c91df60110424b3f8af9348a11ab2f52b1782c428a90`, wall `12 min`, `2,048,954` row-steps). | **TRIGGERED EXTEND (`2 / 7` rescued = `28.6%`)** — Exactly 2 of 7 events rescued (`b8_s3_19@4693` and `b8_s3_9@4188`), hitting the `{1, 2}` boundary rule and automatically triggering the pre-registered 19-event extension. |
| 2026-10-20 | `research/t35_two_deviation` (`RESULT_EXTEND.md`) | Ran Stage 2 extension on all 19 `lad_c_20261019` unrescued natural deaths (`t_death >= 3500`) and pooled across all `N = 26` events (`snake-dqn-artifacts/t35-two-deviation-20261020/extend/summary-pooled.json`, sha256 `7aa2dc18634873e160da783baadbd9170565f232c1919c87113b7db34f0fa957`, `5,361,328` extend row-steps, `7,410,282` pooled row-steps). | **FAIL (`sustained_control_dominant`)** — `3 / 19` (`15.8%`) rescued on the extension cohort; `5 / 26` (`19.2%`, 95% Wilson CI `[8.5%, 37.9%]`) pooled, below the `>= 7 / 26` (`26.9%`) pass bar. |

---

## Final Verdict and Key Numbers

### 1. T35 Auxiliary Outcome-Head Probe (`research/t35_outcome_head`)

- **Verdict:** **NO-GO (`representation_blind`)** — Every pre-registered pass bar failed at the primary horizon `H = 32` (`research/t35_outcome_head/RESULT.md`; primary artifact `snake-dqn-artifacts/t35-outcome-head-20261020/registered-02/result.json`, sha256 `62b644b9ba2c49e9ee6147ce822316b1c8fd2f3366c8e0f105b51fd754258b76`).
- **Primary Horizon (`H = 32`, `N = 16,892` frames, `3,995` positive, base rate `0.2365`):**
  - **Best Tier-0 single-scalar baseline (`D_danger`):** `AUROC = 0.6592 +/- 0.0156`, `PR-AUC = 0.5174`, `Brier = 0.1453`, `Recall@FPR10 = 0.3401 +/- 0.0258`, `Recall@FPR20 = 0.4322`, lead-time profile `P(y_hat >= tau_10 | k)` rising from `0.068` at `k in [25, 32]` and `0.118` at `k in [17, 24]` to `0.223` at `k in [9, 16]` and `0.628` at `k in [1, 8]`.
  - **Other Tier-0 baselines (`H = 32`):** `-V` (`AUROC = 0.5944`, `Recall@FPR10 = 0.2582`), `-Q_max` (`AUROC = 0.5877`, `Recall@FPR10 = 0.2582`), `-Q_margin` (`AUROC = 0.5432`, `Recall@FPR10 = 0.1497`), `H_pi` (`AUROC = 0.5250`, `Recall@FPR10 = 0.1306`).
  - **Tier-1 `obs_60` linear probe (`60-D`):** `AUROC = 0.6646 +/- 0.0195` (`Delta = +0.0054` over `D_danger`), `PR-AUC = 0.4910`, `Brier = 0.1517`, `Recall@FPR10 = 0.3263 +/- 0.0211`.
  - **Tier-2 `fc1` trunk linear probe (`256-D`):** `AUROC = 0.6540 +/- 0.0157` (`Delta = -0.0052` vs `D_danger`, failing the `>= +0.030` bar), `PR-AUC = 0.4718`, `Brier = 0.1563`, `Recall@FPR10 = 0.3059 +/- 0.0198` (failing the `>= 0.45` bar and `-0.0342` below `D_danger`). Early warning in the `[9, 24]` step lead window averaged `0.1544` (`<= 0.1707` for `D_danger`).
  - **Tier-3 `concat[fc1, obs_60, scalars]` linear probe (`320-D`):** `AUROC = 0.6549 +/- 0.0157` (`Delta = -0.0043` vs `D_danger`), `PR-AUC = 0.4765`, `Brier = 0.1562`, `Recall@FPR10 = 0.3137 +/- 0.0149`.
- **Secondary Horizons (`H = 16` and `H = 64`):**
  - At `H = 16` (base rate `13.43%`): `D_danger` achieved `AUROC = 0.7282` (`Recall@FPR10 = 0.4834`), outperforming `obs_60` (`0.7240`, `Recall@FPR10 = 0.4488`), `fc1` (`0.7130`, `Recall@FPR10 = 0.4316`), and `concat_320` (`0.7155`, `Recall@FPR10 = 0.4394`).
  - At `H = 64` (base rate `40.48%`): `D_danger` achieved `AUROC = 0.6056` (`Recall@FPR10 = 0.2286`), while `obs_60` reached `0.6199`, `fc1` reached `0.6042`, and `concat_320` reached `0.6061`.
- **Death Subtype Breakdown (`H = 32`):**
  - **Wall deaths (`229` positive frames):** Predictable by all probes (`D_danger` recall `0.5633`, `obs_60` recall `0.7817`, `fc1` recall `0.6987`).
  - **Body deaths (`3,766` positive frames — `94.3%` of all imminent death frames):** Nearly invisible beyond 8 frames (`D_danger` recall `0.3266`, `obs_60` recall `0.2987`, `fc1` recall `0.2820`, `concat_320` recall `0.2892`).

### 2. T35 Two-Deviation Rescue Probe (`research/t35_two_deviation`)

- **Verdict:** **FAIL (`sustained_control_dominant`)** — Pooled rescue rate across all `N = 26` natural `1`-deviation-unrescued deaths (`t_death >= 3500`) was **`5 / 26` (`19.2%`, 95% Wilson CI `[8.5%, 37.9%]`)**, below the pre-registered `>= 7 / 26` (`26.9%`) pass bar (`research/t35_two_deviation/RESULT_EXTEND.md`; primary artifact `snake-dqn-artifacts/t35-two-deviation-20261020/primary/summary-primary.json`, sha256 `69acca9d11f76cb11337c91df60110424b3f8af9348a11ab2f52b1782c428a90`; pooled artifact `snake-dqn-artifacts/t35-two-deviation-20261020/extend/summary-pooled.json`, sha256 `7aa2dc18634873e160da783baadbd9170565f232c1919c87113b7db34f0fa957`).
- **Stage 1 Primary Cohort (`7` events from `lad_p0_cf_20261019`, `2,048,954` row-steps, `1,024,477` row-steps/rescue):**

| Event ID | Seed | `t_death` | Window `[t_start, t_end]` | Best 1-Dev Delay | Stage-2 Outcome | Best `(t_1, a_1) -> (t_2, a_2)` | Confirm Survival | Row-Steps |
|---|---|---|---|---|---|---|---|---|
| `b8_s0_21@3707` | 21 | 3707 | `[3527, 3706]` | `+46` (`t_fail=3753`) | `NOT_RESCUED` | `(3527, 0) -> (3663, 2)` (`t_fail2=3766`) | `59` (`0` cand) | `370,039` (`345 s`) |
| `b8_s0_8@4318` | 8 | 4318 | `[4138, 4317]` | `+7` (`t_fail=4325`) | `NOT_RESCUED` | `(4139, 4) -> (4240, 0)` (`t_fail2=4334`) | `16` (`0` cand) | `326,941` (`312 s`) |
| `b8_s3_19@4693` | 19 | 4693 | `[4513, 4692]` | `+304` (`t_fail=4997`) | **`RESCUED_DEPTH2`** | `(4813, 3) -> (4965, 0)` (rank 1; ranks 1..4 rescued) | `35` (censored @ `5000`, `4` cand) | `141,293` (`159 s`) |
| `b8_s3_26@3647` | 26 | 3647 | `[3467, 3646]` | `+40` (`t_fail=3687`) | `NOT_RESCUED` | `(3469, 2) -> (3676, 1)` (`t_fail2=3694`) | `47` (`0` cand) | `380,981` (`351 s`) |
| `b8_s3_6@4562` | 6 | 4562 | `[4382, 4561]` | `+3` (`t_fail=4565`) | `NOT_RESCUED` | `(4402, 2) -> (4514, 1)` (`t_fail2=4622`) | `60` (`0` cand) | `156,174` (`155 s`) |
| `b8_s3_8@4246` | 8 | 4246 | `[4066, 4245]` | `+20` (`t_fail=4266`) | `NOT_RESCUED` | `(4068, 0) -> (4250, 2)` (`t_fail2=4273`) | `27` (`0` cand) | `285,910` (`284 s`) |
| `b8_s3_9@4188` | 9 | 4188 | `[4008, 4187]` | `+48` (`t_fail=4236`) | **`RESCUED_DEPTH2`** | `(4198, 0) -> (4204, 3)` (rank 3; `1` cand) | `600` (uncensored) | `387,616` (`370 s`) |

- **Stage 2 Extension Cohort (`19` events from `lad_c_20261019`, `5,361,328` row-steps):**
  - **Rescued (`3 / 19 = 15.8%`):** `b8_s0_101@3956` via `(3790, 2) -> (4136, 4)` at rank 2 (`+194` 1-dev delay, `600` confirm survival, `18` candidates, `383,956` row-steps); `b8_s3_112@4343` via `(4163, 4) -> (4151, 3)` at rank 1 (`+212` 1-dev delay, `600` confirm survival, `1` candidate, `167,692` row-steps); `b8_s3_134@3949` via `(3861, 4) -> (3879, 5)` at rank 4 (`+29` 1-dev delay, `600` confirm survival, `1` candidate, `326,337` row-steps).
  - **Unrescued (`16 / 19 = 84.2%`):** All remaining 16 events (`b8_s0_103@4058`, `b8_s0_108@4852`, `b8_s0_112@4352`, `b8_s0_132@4672`, `b8_s0_134@4023`, `b8_s0_136@3804`, `b8_s0_155@4286`, `b8_s0_158@3908`, `b8_s3_102@4553`, `b8_s3_105@4474`, `b8_s3_110@4030`, `b8_s3_118@3711`, `b8_s3_126@4756`, `b8_s3_129@4547`, `b8_s3_156@3899`, `b8_s3_157@3717`) had `0` rescuing candidates (`confirm_survival` between `12` and `252` frames).
- **Pooled Secondary Diagnostics (`N = 26`):**
  - **Total compute:** `7,410,282` row-steps (`1,482,056` row-steps per rescued event).
  - **Strong structural predictor of 2-dev rescue:** Events where the best 1-deviation branch already delayed death by `> 150` frames (`+194`, `+212`, `+304`) rescued **`3 / 3` (`100%`)** at depth 2; events where the best 1-deviation branch delayed death by `< 60` frames rescued only **`2 / 23` (`8.7%`)**, and only via tight `6..18` frame boost-maneuver pairs (`b8_s3_9@4188` and `b8_s3_134@3949`).

---

## What Was Learned

1. **The `b8-s3` 256-D trunk is representation-blind to multi-step body traps until the last 1–8 frames:** Linearly probing `fc1` (`AUROC = 0.6540`) or even the full 320-D concatenation of `fc1`, `obs_60`, and value/entropy scalars (`AUROC = 0.6549`) failed to beat the single-scalar `v8` flood-fill reachability complement `D_danger` (`AUROC = 0.6592`). Because the ego-centric raster crops (`31x31` stride-1 and `25x25` stride-3) and 2-conv trunk do not explicitly compute multi-step topological closure beyond what `v8` reachability scalars already summarize, training an auxiliary linear/MLP outcome head on the frozen `b8-s3` trunk cannot solve early trap detection.
2. **Sparse open-loop deviations cannot rescue the majority of natural late-game deaths:** Over `80%` (`21 / 26`) of natural 1-deviation-unrescued deaths (`t_death >= 3500`) also fail under exhaustive coarse-to-fine 2-deviation search across a 180-frame lead window. When every single-deviation branch dies within `< 60` frames of the original death (`23 / 26` cases), returning control to greedy `pi_0` after one or two overrides simply steers back into the contested trajectory (`91.3%` failure rate).
3. **Implication for T34 / Watch controller design:** Together, these two negative probes ruled out (a) frozen-trunk outcome-head gating and (b) sparse 2-switch open-loop trajectory repair as standalone fixes. Surviving contested late-game body traps requires **online closed-loop step-by-step replanning** (re-evaluating world rollouts at every decision tick rather than returning immediately to `pi_0` after 1–2 overrides), which directly motivated the closed-loop depth-2 / exact-stop (`exact-stop-l2-prime`) online controllers in the subsequent Watch campaign.

---

## Why It Closed / Current Status

- **Status:** **CLOSED (Both probes completed and falsified their target hypotheses on 2026-10-20).**
- Neither probe consumed any Colab GPU units (both ran locally on Mac CPU in under 35 minutes combined wall-clock). Their negative verdicts permanently closed the frozen-trunk outcome-head and open-loop 2-deviation repair lines.

---

## Kept Files in This Lineage

- [`research/t35_outcome_head/PREREGISTRATION.md`](../../../research/t35_outcome_head/PREREGISTRATION.md) — Pre-registration for the T35 auxiliary outcome-head linear probe (`H in {16, 32, 64}`, 4 probe tiers, pass/fail bars).
- [`research/t35_outcome_head/RESULT.md`](../../../research/t35_outcome_head/RESULT.md) — Final 600-episode / 16,892-frame 5-fold grouped CV result report (`NO-GO: representation_blind`).
- [`research/t35_two_deviation/PREREGISTRATION.md`](../../../research/t35_two_deviation/PREREGISTRATION.md) — Pre-registration for the T35 two-deviation counterfactual rescue probe (`N=7` primary + `N=19` extension rule).
- [`research/t35_two_deviation/RESULT_EXTEND.md`](../../../research/t35_two_deviation/RESULT_EXTEND.md) — Final pooled `N=26` (`5 / 26 = 19.2%`) result report (`FAIL: sustained_control_dominant`).
