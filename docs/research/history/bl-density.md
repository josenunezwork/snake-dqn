# Big-Lever DENSITY-1 and DENSITY-2 Synthetic Arena Shrink vs Natural Doom-Root Yield (`bl-density`)

## Question / Goal

Test whether shrinking the arena geometry—either from episode start (`B50`, `B70m` 70% area, `B70p` 70% perimeter) or dynamically after the hero reaches mature length $L \ge 800$ (`C-P1`, shrinking to the minimum-map floor $\max(8 L_{\text{hero}}, U_{\text{total}}/\rho_{95}, 8349)$)—increases $L \ge 800$ H300 doom-root yield per simulation step by $\ge 2\times$ over natural $145 \times 83$ rollouts (`Arm A`) while preserving $\ge 0.5\times$ witnessed-avoidable fraction and remaining distributionally close (`35`-feature episode-grouped logistic regression $\text{AUC} < 0.80$).

## Timeline of Versions / Steps

| Date | Step / Amendment | What Was Run / Changed | Verdict / Outcome |
| :--- | :--- | :--- | :--- |
| 2026-10-21 | `METRICS.md` & `METRICS_AMENDMENT_01..02.md` | Defined v1 metric contract (`bl-density-development-20261021`, 72 eps across `A, B50, B70, C`); dropped constant columns in `LogisticRegression(class_weight='balanced')`; coarsened sampling stride from 50 to 200 (`24` roots/ep) after Mac preflight measured `2.184720` CPU-s per L800 H300 branch | Superseded by Round-14 Amendment 03 |
| 2026-10-22 | `METRICS_AMENDMENT_03.md` + review/validation (`f92b7d71`) | Implemented Round-14 ruling (b) (`bl-density-round14-20261022`, 90 eps: `A` 36, `B70m` 18, `B70p` 18, `C-P1` 18); dropped `B50`; added 4-root witnessed-avoidable H300 alternative checks and kill bars (`yield < 2*A`, `avoidable < 0.5*A`, or `AUC >= 0.80`). Passed 45 package + 95 transport tests; 12/12 local sparse-archive episodes matched semantic SHA | `PASS` (implementation & preflight) |
| 2026-10-22 | DENSITY-1 Pilot (`METRICS_AMENDMENT_04.md`) | Executed 90-episode DENSITY-1 pilot on Colab L4 CPU (`4,532.02` CPU-s); natural Arm A yielded only `5/36` doom-bearing episodes (`2` witnessed-avoidable roots total), motivating 492-episode DENSITY-2 replication (`bl-density-round14b-20261022`: `A` 240, `B70m` 72, `B70p` 108, `C-P1` 72) | Advanced to DENSITY-2 (`492` eps) |
| 2026-10-22 | DENSITY-2 Attempt 1 & `METRICS_AMENDMENT_05.md` | `density2-main-02-b8ab5916` verified `408/492` episodes (`skip-ledger.json` sha256 `70b252ec...`) in `10,683.46 s` before session timeout, leaving `84` missing primary episodes (`72` `C-P1` + `12` `B70p/mixed` indices `022, 025..035`) plus `12` Arm-A calibration replays (`[0.80, 1.25]` speed band) | `INCOMPLETE` (`408/492` verified) |
| 2026-10-22 | Completion Repins `02` $\to$ `03` $\to$ `04` (`04b55162`) | Repinned DENSITY-2 completion package across transport requalifications (`213f25be` $\to$ `e8e207d2` $\to$ `553af3f6`) and fixed the L4 family-canary gate to require CUDA action observations only when the job declares CUDA (`README.md`, `COMPLETION_REPIN_04.md`) | Prepared (`density2-completion-20261022-04`, `4.90 u` prior spend inside `16.50 u` series cap) |

## Final Verdict and Key Numbers

- **Mac H300 branch cost (`research/bl_density/METRICS_AMENDMENT_02.md`):** Single-thread Mac preflight measured **`2.184720` CPU-s** per $L \ge 800$ H300 branch (~`138.6` CPU-hours at stride 50), requiring stride-200 root sampling (`24` candidate roots per H5000 episode).
- **Amendment 03 validation (`f92b7d718f1415aeebb7856d9d3f1649543ace7a`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/DELIVERY_AMENDMENT_03.json`):** Passed **44/44** independent review tests (`1.48 s`), **45/45** final package tests (`1.51 s`), and **95/95** supervisor/source/transfer tests; Z-9 sparse archive was **`1,163,885` bytes** (`65.408409 s` normal upload forecast, `213.572073 s <= 300 s` fallback forecast), and **12/12** local sparse-archive episodes (`12.357 s` wall) produced identical semantic SHA-256s at `--workers 1` and `--workers 2`.
- **DENSITY-1 90-episode pilot (`research/bl_density/METRICS_AMENDMENT_04.md`):** Completed in **`4,532.02` CPU-s**. Natural `Arm A` produced only **`1/12` frozen, `2/12` mixed, and `2/12` scripted** doom-bearing episodes (`5/36` total) and **`0, 1, 1` witnessed-avoidable roots** across the three mixes. Synthetic doom-bearing episodes were **`5/18` in `B70m`**, **`3/18` in `B70p`**, and **`11/18` in `C-P1`**. Cheap-screen prefilter recall was **`0.60` (`6/10`)**; natural 95th-percentile occupancy $\rho_{95}$ was **`0.1223` (frozen), `0.1051` (mixed), `0.1221` (scripted)**; and **`27/29` (`93.1%`)** doom roots died by `DEATH_SELF` at median hero length **`L = 1186`**.
- **DENSITY-2 492-episode attempt 1 (`research/bl_density/METRICS_AMENDMENT_05.md`, `research/bl_density/COMPLETION_REPIN_04.md`):** Verified **`408 / 492` episodes** (`skip-ledger.json` sha256 `70b252ec52c7d61c62c419e77e4e837a8dc1c5d120c283a61439d186565f444c`) in **`10,683.46 s`** on 10 L4 CPU workers before hitting the wall timeout, leaving **`84` missing primary episodes** (`72` `C-P1` and `12` `B70p/mixed` indices `022, 025..035`) plus **`12` Arm-A calibration replays** (`96` completion units total, estimated at `35 min`, `180 min / 5.0 u` effective cap inside the `16.50 u` series envelope; local H8 synthetic producer canary completed `9` episodes and skipped `3` in `3.859915` CPU-s in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/density2-completion-20261022-04`).

## What Was Learned

1. Stride-200 root sampling on natural $145 \times 83$ rollouts yields very sparse $L \ge 800$ doom roots (`5/36` episodes and only `2` witnessed-avoidable roots in DENSITY-1), making small natural control arms noisy for yield-ratio denominators.
2. Post-$L \ge 800$ dynamic arena shrinking (`C-P1`, floored at $\max(8 L_{\text{hero}}, U_{\text{total}}/\rho_{95}, 8349)$ cells) substantially increases mature doom-episode incidence (`11/18` vs `5/36` natural) where `27/29` deaths are self-trap (`DEATH_SELF`, median $L = 1186$), whereas static start-of-episode shrinks (`B70m`, `B70p`) suppress early growth before snakes reach $L \ge 800$.
3. CPU-only multi-worker jobs running on Colab L4 for vCPU count must exempt CUDA action-observation checks in the family-canary gate while keeping full transport, sidecar, and verified-release checks (`research/bl_density/README.md`).

## Why It Closed / Current Status

**Closed / superseded by direct natural hindsight factories (`F0`/`D256`/`D1024`/`D4096`) and GPU counterfactual rollouts (`t34-ct-1028`).** DENSITY-2 Attempt 1 (`408/492` verified episodes) remained sealed pending completion of the last 84 `C-P1`/`B70p` units (`COMPLETION_REPIN_04.md`) when the campaign pivoted to mining exact-stop death-prefix rescues directly from natural trajectories.

## Kept Files in This Lineage

- [`research/bl_density/COMPLETION_REPIN_03.md`](../../../research/bl_density/COMPLETION_REPIN_03.md)
- [`research/bl_density/COMPLETION_REPIN_04.md`](../../../research/bl_density/COMPLETION_REPIN_04.md)
- [`research/bl_density/METRICS.md`](../../../research/bl_density/METRICS.md)
- [`research/bl_density/METRICS_AMENDMENT_01.md`](../../../research/bl_density/METRICS_AMENDMENT_01.md)
- [`research/bl_density/METRICS_AMENDMENT_02.md`](../../../research/bl_density/METRICS_AMENDMENT_02.md)
- [`research/bl_density/METRICS_AMENDMENT_03.md`](../../../research/bl_density/METRICS_AMENDMENT_03.md)
- [`research/bl_density/METRICS_AMENDMENT_04.md`](../../../research/bl_density/METRICS_AMENDMENT_04.md)
- [`research/bl_density/METRICS_AMENDMENT_05.md`](../../../research/bl_density/METRICS_AMENDMENT_05.md)
- [`research/bl_density/README.md`](../../../research/bl_density/README.md)
