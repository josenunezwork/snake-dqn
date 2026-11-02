# Big-Lever DENSITY-1 and DENSITY-2 Synthetic Arena Shrink vs Natural Doom-Root Yield (`bl-density`)

## Question / Goal

Development-only test (no efficacy, promotion, release or causal claim) of whether synthetic densification raises $L \ge 800$ H300 doom-root yield over the natural arm `A` (standard $145 \times 83$ map, five opponents): `B50`/`B70` shrink both map dimensions by $\sqrt{0.5}$/$\sqrt{0.7}$ (later `B70m` = `121x69` with five opponents, `B70p` = same map with seven opponents), and `C`/`C-P1` inject long serpentine bodies on the standard map (C-P1 uses P1's `collector._inject_long` construction). Round-14 kill rules: REJECTED on yield `< 2*A`, avoidable fraction `< 0.5*A`, classifier AUC `>= 0.80` (35-feature episode-grouped logistic regression), or a minimum-map violation, with $A_{\min}(L)=\max(8L_{\text{hero}}, U_{\text{total}}/\rho_{95}, 8349)$ (`research/bl_density/METRICS.md`, `METRICS_AMENDMENT_03.md`, `METRICS_AMENDMENT_04.md`).

## Timeline of Versions / Steps

| Date | Step / Amendment | What Was Run / Changed | Verdict / Outcome |
| :--- | :--- | :--- | :--- |
| 2026-10-21 | `METRICS.md` & `METRICS_AMENDMENT_01..02.md` | Defined v1 metric contract (`bl-density-development-20261021`, 72 eps across `A, B50, B70, C`); dropped constant columns in `LogisticRegression(class_weight='balanced')`; coarsened sampling stride from 50 to 200 (`24` roots/ep) after Mac preflight measured `2.184720` CPU-s per L800 H300 branch | Superseded by Round-14 Amendment 03 |
| 2026-10-22 | `METRICS_AMENDMENT_03.md` + review/validation (`f92b7d71`) | Implemented Round-14 ruling (b) (`bl-density-round14-20261022`, 90 eps: `A` 36, `B70m` 18, `B70p` 18, `C-P1` 18); dropped `B50`; added 4-root witnessed-avoidable H300 alternative checks and kill bars (`yield < 2*A`, `avoidable < 0.5*A`, or `AUC >= 0.80`). Passed 45 package + 95 transport tests; 12/12 local sparse-archive episodes matched semantic SHA | `PASS` (implementation & preflight) |
| 2026-10-22 | DENSITY-1 Pilot (`METRICS_AMENDMENT_04.md`) | 90-episode pilot (`4,532` CPU-s total); natural Arm A yielded only `5/36` doom-bearing episodes (`2` witnessed-avoidable roots total), motivating 492-episode DENSITY-2 replication (`bl-density-round14b-20261022`: `A` 240, `B70m` 72, `B70p` 108, `C-P1` 72) | Advanced to DENSITY-2 (`492` eps) |
| 2026-10-22 | DENSITY-2 production & `METRICS_AMENDMENT_05.md` | Production (`density2-20261022-gatefix-f1904777/production-output-02`, ten single-thread L4 CPU workers) hit a second mechanical failure (deadline reached during telemetry recovery after work finished in ~67 min); `408/492` episodes verified locally (ledger SHA256 `70b252ec...`), leaving `84` missing primary episodes (`72` `C-P1` + `12` `B70p/mixed` indices `022, 025..035`) plus `12` Arm-A calibration replays (`[0.80, 1.25]` speed band) | Outcome-blind completion authorized (`408/492` verified) |
| 2026-10-22 | Completion Repins `02` $\to$ `03` $\to$ `04` (`04b55162`) | Repinned DENSITY-2 completion package across transport requalifications (`213f25be` $\to$ `e8e207d2` $\to$ `553af3f6`) and fixed the L4 family-canary gate to require CUDA action observations only when the job declares CUDA (`README.md`, `COMPLETION_REPIN_04.md`) | Prepared (`density2-completion-20261022-04`, `4.90 u` prior spend inside `16.50 u` series cap) |

## Final Verdict and Key Numbers

- **Mac H300 branch cost (`research/bl_density/METRICS_AMENDMENT_02.md`):** an excluded Mac engineering preflight ran one L800 scripted H300 branch in **`2.184720` CPU-s**, making the original 2,412,000-decision workload too large for a `<= 90`-minute L4 session; sampling moved to stride 200 (`24` possible roots per H5000 episode).
- **Amendment 03 validation (`f92b7d718f1415aeebb7856d9d3f1649543ace7a`, `research/bl_density/VALIDATION_AMENDMENT_03.md` and `REVIEW_AMENDMENT_03.md`, removed in consolidation; see git c4c4ab41):** Passed **44/44** independent review tests (`1.48 s`), **45/45** final package tests (`1.51 s`), and **95/95** supervisor/source/transfer tests; Z-9 sparse archive was **`1,163,885` bytes** (`65.408409 s` normal upload forecast, `213.572073 s <= 300 s` fallback forecast), and **12/12** local sparse-archive episodes (`12.357 s` wall) produced identical semantic SHA-256s at `--workers 1` and `--workers 2`.
- **DENSITY-1 90-episode pilot (sizing disclosures in `research/bl_density/METRICS_AMENDMENT_04.md`):** **`4,532` CPU-s** total. Natural `Arm A` produced only **`1/12` frozen, `2/12` mixed, and `2/12` scripted** doom-bearing episodes (`5/36` total) and **`0, 1, 1` witnessed-avoidable roots** across the three mixes. Synthetic doom-bearing episodes were **`5/18` in `B70m`**, **`3/18` in `B70p`**, and **`11/18` in `C-P1`**. Cheap-screen prefilter recall was **`0.60` (`6/10`)**; natural $\rho_{95}$ was **`.122 / .105 / .122`** (frozen/mixed/scripted); and **`27/29`** pilot doom roots were `DEATH_SELF` (`L801–2094`, median **`1186`**).
- **DENSITY-2 492-episode production (`research/bl_density/COMPLETION.md`, `METRICS_AMENDMENT_05.md`, `COMPLETION_REPIN_04.md`):** `408/492` episodes verified locally (ledger SHA256 `70b252ec52c7d61c62c419e77e4e837a8dc1c5d120c283a61439d186565f444c`), missing `84` (`72` C-P1 + B70p/mixed `022, 025–035`) plus `12` non-scoring Arm-A replays = `96` completion units; completion estimate `35 min`, effective cap `5.0 u / 180 min`, series envelope `16.50 u`, prior spend `4.90 u`; local tiny H8 producer `9` completed / `3` skipped in `3.859915` CPU-s.
- **Terminal result (2026-10-22 17:16 PT; `docs/notes/big-lever-experiments.md`, `research/INDEX.md`):** **all nine smaller-map/more-opponent cells REJECTED under the minimum-map rule, identity `12/12`; density lever closed.**

## What Was Learned

1. Natural $145 \times 83$ rollouts at stride 200 yield very sparse $L \ge 800$ doom roots (pilot: `5/36` A episodes, `0/1/1` witnessed-avoidable roots), so small natural control arms give noisy yield denominators; the pilot was INSUFFICIENT.
2. Synthetic arms produced more doom-bearing pilot episodes (C-P1 `11/18`, B70m `5/18`, B70p `3/18`), but in DENSITY-2 every densified cell was REJECTED by the minimum-map rule, so none was admitted as a realistic densifier.
3. CPU-only multi-worker jobs placed on Colab L4 must not require CUDA action observations in the family-canary gate unless the job declares CUDA, while keeping transport, sidecar and verified-release checks (`research/bl_density/GATEFIX_PENDING_REQUALIFICATION.md`, removed in consolidation; see git c4c4ab41).

## Why It Closed / Current Status

**Closed — TERMINAL / REJECTED (2026-10-22).** All nine DENSITY-2 cells were rejected under the minimum-map rule, closing the density lever (`research/INDEX.md`, `docs/notes/big-lever-experiments.md`).

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
